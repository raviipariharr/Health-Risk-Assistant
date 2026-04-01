"""
ROUTE: /api/v1/analyze
========================
This is the main endpoint that ties everything together.

Request flow:
  POST /api/v1/analyze
       ↓
  1. FastAPI validates SymptomRequest (Pydantic)
       ↓
  2. NLP processor extracts keywords, severity, duration
       ↓
  3. ML scorer computes risk score + feature contributions
       ↓
  4. AI client calls Claude with NLP + ML data as context
       ↓
  5. Merge NLP + ML + AI into AnalysisResponse
       ↓
  6. Return JSON to frontend

The `async def` + `await` pattern means FastAPI won't
block while waiting for the AI API call — it handles
other requests in the meantime.
"""

import time
import uuid
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from app.models import (
    SymptomRequest, AnalysisResponse, NLPResult,
    RiskPrediction, Suggestion, ErrorResponse,
    RiskLevel, SuggestionPriority
)
from app.nlp.processor import process_symptoms
from app.ml.scorer import score_symptoms_ml, map_score_to_level, get_primary_concern
from app.ml.ai_client import get_ai_assessment


router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Analyze symptoms and return risk assessment",
    description="""
    Full NLP + ML + AI pipeline:
    1. Extract symptom keywords via NLP
    2. Score risk using weighted ML model
    3. Enhance with Claude AI reasoning
    4. Return structured risk assessment with explainability
    """
)
async def analyze_symptoms(
    request: SymptomRequest,
    x_api_key: Optional[str] = Header(default=None, description="Anthropic API key")
):
    start_time = time.monotonic()
    session_id = str(uuid.uuid4())[:8]   # short random ID for this analysis

    # --- Validate we have something to analyze ---
    combined = request.combined_symptoms
    if not combined.strip():
        raise HTTPException(
            status_code=400,
            detail="No symptoms provided. Please describe your symptoms or select from the list."
        )

    # =============================================
    # STEP 1: NLP PROCESSING
    # =============================================
    nlp_output = process_symptoms(
        symptom_text=request.symptom_text,
        selected_chips=request.selected_chips,
        age=request.age,
        sex=request.sex.value,
        duration=request.duration.value
    )

    # Build the NLPResult response object
    nlp_result = NLPResult(
        detected_keywords=nlp_output["detected_keywords"],
        symptom_summary=nlp_output["symptom_summary"],
        parsed_highlights=nlp_output["parsed_highlights"],
        severity_indicators=nlp_output["severity_indicators"],
        duration_context=nlp_output["duration_context"]
    )

    # =============================================
    # STEP 2: ML RISK SCORING
    # =============================================
    ml_score, ml_contributions = score_symptoms_ml(
        keywords=nlp_output["detected_keywords"],
        severity_indicators=nlp_output["severity_indicators"],
        duration=request.duration.value,
        age=request.age,
        sex=request.sex.value
    )

    primary_concern = get_primary_concern(
        nlp_output["detected_keywords"],
        ml_score
    )

    # =============================================
    # STEP 3: AI ASSESSMENT (Claude)
    # =============================================
    if x_api_key:
        # If API key is provided in header, use live Claude
        try:
            ai_output = await get_ai_assessment(
                combined_text=nlp_output["combined_text"],
                keywords=nlp_output["detected_keywords"],
                severity_indicators=nlp_output["severity_indicators"],
                ml_score=ml_score,
                ml_contributions=ml_contributions,
                primary_concern=primary_concern,
                duration_context=nlp_output["duration_context"],
                age=request.age,
                sex=request.sex.value,
                api_key=x_api_key
            )
            risk_data = ai_output.get("risk", {})
            suggestions_data = ai_output.get("suggestions", [])
        except Exception as e:
            # Fall back to ML-only if AI call fails
            risk_data = {}
            suggestions_data = []
    else:
        # No API key: use ML-only mode (no LLM call)
        risk_data = {}
        suggestions_data = []

    # =============================================
    # STEP 4: MERGE RESULTS
    # =============================================

    # Use AI score if available, fall back to ML score
    final_score = int(risk_data.get("score", ml_score))
    final_level = risk_data.get("level") or map_score_to_level(final_score)

    risk_prediction = RiskPrediction(
        level=RiskLevel(final_level),
        score=final_score,
        primary_concern=risk_data.get("primary_concern", primary_concern),
        explanation=risk_data.get("explanation",
            f"Based on {len(nlp_output['detected_keywords'])} detected symptoms "
            f"with a weighted ML risk score of {ml_score:.0f}/100."
        ),
        confidence=risk_data.get("confidence", "medium"),
        reasoning_steps=risk_data.get("reasoning_steps", ml_contributions),
        ml_score=ml_score,
        ai_score=risk_data.get("ai_score")
    )

    # Build suggestion objects
    suggestions = []
    for s in suggestions_data:
        suggestions.append(Suggestion(
            priority=SuggestionPriority(s.get("priority", "general")),
            icon=s.get("icon", "🩺"),
            title=s.get("title", "Follow up"),
            detail=s.get("detail", "")
        ))

    # Default suggestions if AI didn't provide any
    if not suggestions:
        if final_level == "high":
            suggestions = [
                Suggestion(priority=SuggestionPriority.urgent, icon="🏥",
                    title="Seek medical care today",
                    detail="Your symptom profile suggests you should be evaluated by a healthcare provider today.")
            ]
        suggestions.append(
            Suggestion(priority=SuggestionPriority.general, icon="📋",
                title="Track your symptoms",
                detail="Keep a log of when symptoms occur, their severity, and any triggers.")
        )

    # Calculate processing time
    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    return AnalysisResponse(
        success=True,
        session_id=session_id,
        nlp=nlp_result,
        risk=risk_prediction,
        suggestions=suggestions,
        processing_time_ms=elapsed_ms
    )
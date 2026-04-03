"""
ROUTE: /api/v1/analyze  (UPDATED for Step 5)
=============================================
Changes from Step 2:
  - scorer.py  (rule-based weights)   → classifier.py  (trained RandomForest)
  - ML output now includes probabilities, confidence, top active features
  - New /api/v1/model-info endpoint exposes model metadata
  - Explainability enriched with feature-level importances
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
from app.ml.classifier import predict as ml_predict, get_model_info
from app.ml.ai_client import get_ai_assessment

router = APIRouter()


@router.get("/model-info", summary="Get trained ML model metadata")
async def model_info():
    return get_model_info()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze symptoms — NLP + ML + AI pipeline",
)
async def analyze_symptoms(
    request: SymptomRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    start_time = time.monotonic()
    session_id = str(uuid.uuid4())[:8]

    if not request.combined_symptoms.strip():
        raise HTTPException(status_code=400, detail="No symptoms provided.")

    # STEP 2: NLP
    nlp_output = process_symptoms(
        symptom_text=request.symptom_text,
        selected_chips=request.selected_chips,
        age=request.age,
        sex=request.sex.value,
        duration=request.duration.value
    )
    nlp_result = NLPResult(
        detected_keywords=nlp_output["detected_keywords"],
        symptom_summary=nlp_output["symptom_summary"],
        parsed_highlights=nlp_output["parsed_highlights"],
        severity_indicators=nlp_output["severity_indicators"],
        duration_context=nlp_output["duration_context"]
    )

    # STEP 5: ML CLASSIFIER
    ml_result = ml_predict(
        keywords=nlp_output["detected_keywords"],
        severity_indicators=nlp_output["severity_indicators"],
        duration=request.duration.value,
        age=request.age,
    )

    # Build human-readable reasoning from ML output
    ml_reasoning = []
    if ml_result["model_available"]:
        active = ml_result["top_features"]
        if active:
            top = [f["feature"].replace("has_", "").replace("_", " ") for f in active[:3]]
            ml_reasoning.append(f"Key features detected: {', '.join(top)}")
        p = ml_result["probabilities"]
        ml_reasoning.append(
            f"Model probabilities — Low: {p['low']:.0%}, Medium: {p['medium']:.0%}, High: {p['high']:.0%}"
        )
        ml_reasoning.append(
            f"Model confidence: {ml_result['confidence']} (score {ml_result['score']}/100)"
        )
        if ml_result["top_features"]:
            feat_line = "Feature importances: " + ", ".join(
                f'{f["feature"].replace("has_","").replace("_"," ")} ({f["importance"]:.3f})'
                for f in ml_result["top_features"][:4]
            )
            ml_reasoning.append(feat_line)
    else:
        ml_reasoning = ["ML model not yet trained — run: python -m app.ml.training.train"]

    primary_concern = "Symptom profile warrants clinical evaluation"
    if ml_result["top_features"]:
        top_feat = ml_result["top_features"][0]["feature"]
        primary_concern = top_feat.replace("has_", "").replace("_", " ").title() + " identified as primary concern"

    # STEP 3: AI (if API key provided)
    risk_data, suggestions_data = {}, []
    if x_api_key:
        try:
            ai_output = await get_ai_assessment(
                combined_text=nlp_output["combined_text"],
                keywords=nlp_output["detected_keywords"],
                severity_indicators=nlp_output["severity_indicators"],
                ml_score=float(ml_result["score"]),
                ml_contributions=ml_reasoning,
                primary_concern=primary_concern,
                duration_context=nlp_output["duration_context"],
                age=request.age,
                sex=request.sex.value,
                api_key=x_api_key
            )
            risk_data = ai_output.get("risk", {})
            suggestions_data = ai_output.get("suggestions", [])
        except Exception:
            pass

    # MERGE
    final_score = int(risk_data.get("score", ml_result["score"]))
    final_level = risk_data.get("level") or ml_result["level"]
    reasoning = risk_data.get("reasoning_steps") or ml_reasoning

    risk_prediction = RiskPrediction(
        level=RiskLevel(final_level),
        score=final_score,
        primary_concern=risk_data.get("primary_concern", primary_concern),
        explanation=risk_data.get("explanation",
            f"RandomForest classifier scored {ml_result['score']}/100 → {final_level.upper()} risk. "
            f"Confidence: {ml_result['confidence']}. "
            f"Probabilities: low={ml_result['probabilities']['low']:.0%}, "
            f"medium={ml_result['probabilities']['medium']:.0%}, "
            f"high={ml_result['probabilities']['high']:.0%}."
        ),
        confidence=risk_data.get("confidence", ml_result["confidence"]),
        reasoning_steps=reasoning,
        ml_score=float(ml_result["score"]),
        ai_score=risk_data.get("ai_score"),
    )

    suggestions = [
        Suggestion(
            priority=SuggestionPriority(s.get("priority", "general")),
            icon=s.get("icon", "🩺"),
            title=s.get("title", "Follow up"),
            detail=s.get("detail", "")
        ) for s in suggestions_data
    ] or [
        Suggestion(priority=SuggestionPriority.urgent if final_level == "high" else SuggestionPriority.general,
                   icon="🏥" if final_level == "high" else "📋",
                   title="Seek care today" if final_level == "high" else "Track your symptoms",
                   detail="Your symptom profile warrants prompt evaluation." if final_level == "high"
                          else "Log when symptoms occur and share with your doctor.")
    ]

    return AnalysisResponse(
        success=True,
        session_id=session_id,
        nlp=nlp_result,
        risk=risk_prediction,
        suggestions=suggestions,
        processing_time_ms=int((time.monotonic() - start_time) * 1000)
    )
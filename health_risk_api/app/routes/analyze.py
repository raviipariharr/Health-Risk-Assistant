"""
ROUTE: /api/v1/analyze  (Extended with Vitals & Labs)
=======================================================
Pipeline:
  1. NLP — symptom keyword extraction
  2. Vitals/Labs interpreter — clinical measurement flags + ML features
  3. ML classifier — full feature vector (symptoms + measurements)
  4. AI (optional) — Claude reasoning with vitals context
  5. Merge & return
"""

import time
import uuid
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from app.models import (
    SymptomRequest, AnalysisResponse, NLPResult,
    RiskPrediction, Suggestion, ErrorResponse,
    RiskLevel, SuggestionPriority, VitalsInterpretation
)
from app.nlp.processor import process_symptoms
from app.nlp.vitals import interpret_vitals_and_labs
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
    summary="Analyze symptoms + vitals/labs — NLP + ML + AI pipeline",
)
async def analyze_symptoms(
    request: SymptomRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    start_time = time.monotonic()
    session_id = str(uuid.uuid4())[:8]

    if not request.combined_symptoms.strip():
        raise HTTPException(status_code=400, detail="No symptoms provided.")

    # STEP 1: NLP
    nlp_output = process_symptoms(
        symptom_text=request.symptom_text,
        selected_chips=request.selected_chips,
        age=request.age,
        sex=request.sex.value,
        duration=request.duration.value
    )

    # STEP 2: Vitals & Labs interpretation
    vitals_output = interpret_vitals_and_labs(
        vitals=request.vitals,
        labs=request.labs,
        age=request.age,
        sex=request.sex.value,
    )

    # Merge vitals-inferred keywords into NLP output
    enriched_keywords = list(nlp_output["detected_keywords"])
    for kw in vitals_output["inferred_keywords"]:
        if kw not in enriched_keywords:
            enriched_keywords.append(kw)

    # Merge vitals summary into symptom summary
    symptom_summary = nlp_output["symptom_summary"]
    if vitals_output["summary"]:
        symptom_summary += " " + vitals_output["summary"]

    nlp_result = NLPResult(
        detected_keywords=enriched_keywords,
        symptom_summary=symptom_summary,
        parsed_highlights=nlp_output["parsed_highlights"],
        severity_indicators=nlp_output["severity_indicators"],
        duration_context=nlp_output["duration_context"]
    )

    # STEP 3: ML Classifier — pass vitals ML features
    ml_result = ml_predict(
        keywords=enriched_keywords,
        severity_indicators=nlp_output["severity_indicators"],
        duration=request.duration.value,
        age=request.age,
        severity_boost=vitals_output["severity_boost"],
        contexts=nlp_output.get("contexts", []),
        vitals_ml_features=vitals_output["ml_features"],
    )

    # Build ML reasoning steps
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
        if vitals_output["flags"]:
            flags_str = "; ".join(vitals_output["flags"][:3])
            ml_reasoning.append(f"Measurement findings: {flags_str}")
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

    # STEP 4: AI (if API key provided)
    risk_data, suggestions_data = {}, []
    if x_api_key:
        try:
            # Build extended combined text including vitals flags
            vitals_context = ""
            if vitals_output["flags"]:
                vitals_context = "\n\nMeasurement findings: " + "; ".join(vitals_output["flags"])

            ai_output = await get_ai_assessment(
                combined_text=nlp_output["combined_text"] + vitals_context,
                keywords=enriched_keywords,
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

    # STEP 5: Merge
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
        vitals_flags=vitals_output["flags"],
    )

    suggestions = [
        Suggestion(
            priority=SuggestionPriority(s.get("priority", "general")),
            icon=s.get("icon", "🩺"),
            title=s.get("title", "Follow up"),
            detail=s.get("detail", "")
        ) for s in suggestions_data
    ] or [
        Suggestion(
            priority=SuggestionPriority.urgent if final_level == "high" else SuggestionPriority.general,
            icon="🏥" if final_level == "high" else "📋",
            title="Seek care today" if final_level == "high" else "Track your symptoms",
            detail="Your symptom profile warrants prompt evaluation." if final_level == "high"
                   else "Log when symptoms occur and share with your doctor."
        )
    ]

    vitals_summary = VitalsInterpretation(
        flags=vitals_output["flags"],
        summary=vitals_output["summary"],
    )

    return AnalysisResponse(
        success=True,
        session_id=session_id,
        nlp=nlp_result,
        risk=risk_prediction,
        suggestions=suggestions,
        vitals_summary=vitals_summary,
        processing_time_ms=int((time.monotonic() - start_time) * 1000)
    )
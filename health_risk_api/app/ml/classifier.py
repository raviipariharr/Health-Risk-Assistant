"""
STEP 5: CLASSIFIER — Feature Extraction + Prediction
======================================================
This is the bridge between NLP output and the ML model.

Two responsibilities:
  1. extract_features()
     Converts NLP processor output (keyword list) into a
     numeric feature vector that the model can understand.
     
     NLP gives us: ["chest pain", "shortness of breath", "dizziness"]
     Model needs:  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ..., 55, 3, 2, 0, 0, 1, 0]
     
     This translation step is called "feature engineering".

  2. predict()
     Loads the saved model (once, on first call)
     Runs the feature vector through the pipeline
     Returns score, risk level, confidence, and explanation

Why load the model once?
  → joblib.load() reads a file from disk — that's slow.
  → We cache it in a module-level variable (_model).
  → Second call returns instantly from memory.
  → This pattern is called "lazy loading" or "singleton".
"""

import os
import json
import joblib
import numpy as np
from typing import List, Tuple, Dict, Optional

from app.ml.training.dataset import (
    ALL_FEATURE_NAMES, SYMPTOM_FEATURES, NUMERIC_FEATURES, RISK_LABELS
)

# ---- Paths ----
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_classifier.joblib')
META_PATH  = os.path.join(MODEL_DIR, 'model_metadata.json')

# ---- Module-level cache ----
# None = not loaded yet
# Set on first call to predict()
_model    = None
_metadata = None


def _load_model():
    """
    Load the trained pipeline from disk.
    Called automatically on first prediction.
    Raises FileNotFoundError if model hasn't been trained yet.
    """
    global _model, _metadata

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Run training first:\n"
            f"  python -m app.ml.training.train"
        )

    _model = joblib.load(MODEL_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)

    print(f"[ML] Model loaded: {_metadata.get('model_type')} "
          f"(accuracy={_metadata.get('test_accuracy')})")


# ---- Keyword → feature name mapping ----
# Maps NLP keyword strings to the boolean feature column names.
# This is the "vocabulary" that connects NLP output to ML input.
KEYWORD_TO_FEATURE: Dict[str, str] = {
    "chest pain":             "has_chest_pain",
    "chest tightness":        "has_chest_pain",
    "chest pressure":         "has_chest_pain",
    "exertional chest pain":  "has_chest_pain",
    "progressive chest pain": "has_chest_pain",
    "shortness of breath":    "has_shortness_of_breath",
    "breathlessness":         "has_shortness_of_breath",
    "difficulty breathing":   "has_shortness_of_breath",
    "exertional dyspnoea":    "has_shortness_of_breath",
    "resting dyspnoea":       "has_shortness_of_breath",
    "progressive dyspnoea":   "has_shortness_of_breath",
    "breathless":             "has_shortness_of_breath",
    "palpitations":           "has_palpitations",
    "rapid heartbeat":        "has_palpitations",
    "exertional palpitations":"has_palpitations",
    "exertional syncope":     "has_palpitations",
    "heart pounding":         "has_palpitations",
    "skipped beats":          "has_palpitations",
    "irregular heartbeat":    "has_irregular_heartbeat",
    "radiating":              "has_radiating_pain",
    "radiating pain":         "has_radiating_pain",
    "left arm pain":          "has_radiating_pain",
    "jaw pain":               "has_radiating_pain",
    "confusion":              "has_confusion",
    "slurred speech":         "has_slurred_speech",
    "seizure":                "has_seizure",
    "fainting":               "has_fainting",
    "orthostatic dizziness":  "has_fainting",
    "sudden":                 "has_sudden_onset",
    "sudden onset":           "has_sudden_onset",
    "thunderclap headache":   "has_sudden_onset",
    "blood in stool":         "has_blood_in_stool",
    "rectal bleeding":        "has_blood_in_stool",
    "blood in urine":         "has_blood_in_urine",
    "weight loss":            "has_severe_weight_loss",
    "severe weight loss":     "has_severe_weight_loss",
    "fever":                  "has_fever",
    "high fever":             "has_high_fever",
    "night sweats":           "has_night_sweats",
    "chills":                 "has_night_sweats",
    "dizziness":              "has_dizziness",
    "vertigo":                "has_dizziness",
    "lightheadedness":        "has_dizziness",
    "vomiting":               "has_vomiting",
    "abdominal pain":         "has_abdominal_pain",
    "stomach ache":           "has_stomach_ache",
    "numbness":               "has_numbness",
    "tingling":               "has_numbness",
    "blurred vision":         "has_blurred_vision",
    "double vision":          "has_blurred_vision",
    "headache":               "has_headache",
    "migraine":               "has_headache",
    "fatigue":                "has_fatigue",
    "tiredness":              "has_fatigue",
    "exhaustion":             "has_fatigue",
    "nausea":                 "has_nausea",
    "cough":                  "has_cough",
    "dry cough":              "has_cough",
    "back pain":              "has_back_pain",
    "joint pain":             "has_joint_pain",
    "muscle ache":            "has_joint_pain",
    "sore throat":            "has_sore_throat",
    "rash":                   "has_rash",
}

# Severity text → numeric code
SEVERITY_TO_SCORE = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 3,
    "intense": 2,
    "unbearable": 3,
    "worst": 3,
    "excruciating": 3,
    "persistent": 2,
    "constant": 2,
    "worsening": 2,
    "sudden": 2,
    "crushing": 3,
    "stabbing": 2,
}

# Duration code mapping
DURATION_TO_CODE = {
    "hours": 0,
    "days": 1,
    "week": 2,
    "weeks": 3,
    "months": 4,
}


def extract_features(
    keywords: List[str],
    severity_indicators: List[str],
    duration: str,
    age: Optional[int],
    severity_boost: int = 0,
    contexts: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert NLP output into a numeric feature vector.
    Now accepts severity_boost (from contextual inference rules)
    and contexts (exertional, at_rest, etc.) for richer features.
    """
    contexts = contexts or []
    feat = {name: 0 for name in SYMPTOM_FEATURES}

    # Map keywords → boolean features
    for kw in keywords:
        feat_name = KEYWORD_TO_FEATURE.get(kw.lower())
        if feat_name and feat_name in feat:
            feat[feat_name] = 1

    symptom_count = sum(feat.values())

    # Severity score from indicators
    severity_score = 0
    for indicator in severity_indicators:
        s = SEVERITY_TO_SCORE.get(indicator.lower(), 0)
        severity_score = max(severity_score, s)

    # Apply boost from inference rules (e.g. chest pain + exertional → +2)
    severity_score = min(3, severity_score + severity_boost)

    # Context modifiers: exertional symptoms are higher risk
    is_exertional = 1 if "exertional" in contexts else 0
    is_at_rest    = 1 if "at_rest" in contexts else 0
    is_sudden     = 1 if "sudden_onset" in contexts else 0

    # If exertional context present with cardiac keywords, boost symptom count
    # to reflect the higher clinical significance
    if is_exertional and (feat.get("has_chest_pain") or feat.get("has_shortness_of_breath")):
        symptom_count = min(symptom_count + 2, 15)  # cap at 15

    duration_code = DURATION_TO_CODE.get(duration, 1)
    age_val = float(age) if age else 40.0

    numeric = {
        "age":                      age_val,
        "symptom_count":            symptom_count,
        "severity_score":           severity_score,
        "duration_code":            duration_code,
        "is_acute":                 1 if duration_code == 0 else 0,
        "age_over_60":              1 if age_val >= 60 else 0,
        "has_any_severe_indicator": 1 if severity_score >= 2 else 0,
    }

    full_feat = {**feat, **numeric}
    feature_vector = np.array(
        [full_feat[name] for name in ALL_FEATURE_NAMES],
        dtype=float
    ).reshape(1, -1)

    return feature_vector, full_feat


def predict(
    keywords: List[str],
    severity_indicators: List[str],
    duration: str,
    age: Optional[int],
    severity_boost: int = 0,
    contexts: Optional[List[str]] = None,
) -> Dict:
    """
    Full prediction pipeline.
    
    Returns a dict with:
      - score (0-100)
      - level ("low" | "medium" | "high")
      - probabilities { "low": 0.12, "medium": 0.31, "high": 0.57 }
      - confidence ("low" | "medium" | "high")
      - top_features: list of (feature_name, value) for active features
      - model_available: bool
    """
    global _model

    # Lazy-load the model on first prediction
    if _model is None:
        try:
            _load_model()
        except FileNotFoundError as e:
            # Model not trained yet — return a graceful fallback
            return {
                "score": 0,
                "level": "low",
                "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0},
                "confidence": "low",
                "top_features": [],
                "model_available": False,
                "error": str(e),
            }

    # Build feature vector (now includes severity_boost and contexts)
    X, feat_dict = extract_features(
        keywords, severity_indicators, duration, age,
        severity_boost=severity_boost,
        contexts=contexts,
    )

    # --- Prediction ---
    # predict() returns the most likely class: 0, 1, or 2
    pred_class = int(_model.predict(X)[0])

    # predict_proba() returns probabilities for all 3 classes
    # e.g. [0.08, 0.25, 0.67] → 67% chance high risk
    proba = _model.predict_proba(X)[0]  # shape (3,)

    # --- Score: weighted average of class probabilities ---
    # Maps [low=0, med=1, high=2] probs to 0-100 scale
    # score = (0*p_low + 50*p_med + 100*p_high) is too binary
    # Better: use the predicted class probability centred on its range
    class_midpoints = [15.0, 50.0, 85.0]
    score = float(sum(p * m for p, m in zip(proba, class_midpoints)))
    score = round(min(100, max(0, score)))

    level = RISK_LABELS[pred_class]

    # --- Confidence: how certain is the model? ---
    # max_prob close to 1.0 = very confident
    # max_prob close to 0.33 = uncertain (all classes equally likely)
    max_prob = float(proba[pred_class])
    if max_prob >= 0.75:
        confidence = "high"
    elif max_prob >= 0.50:
        confidence = "medium"
    else:
        confidence = "low"

    # --- Explainability: which features were active? ---
    # Get feature importances from the trained Random Forest
    rf = _model.named_steps['classifier']
    importances = rf.feature_importances_

    # Pair each feature with its importance AND whether it was active
    active_features = []
    for i, fname in enumerate(ALL_FEATURE_NAMES):
        val = feat_dict.get(fname, 0)
        if val > 0:  # only report active features
            active_features.append({
                "feature": fname,
                "value": float(val),
                "importance": float(round(importances[i], 4)),
            })

    # Sort by importance (highest first)
    active_features.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "score": score,
        "level": level,
        "probabilities": {
            "low":    float(round(proba[0], 3)),
            "medium": float(round(proba[1], 3)),
            "high":   float(round(proba[2], 3)),
        },
        "confidence": confidence,
        "top_features": active_features[:8],  # top 8 active features
        "model_available": True,
        "metadata": {
            "test_accuracy": _metadata.get("test_accuracy") if _metadata else None,
            "model_type": _metadata.get("model_type") if _metadata else None,
        }
    }


def get_model_info() -> Dict:
    """Return metadata about the loaded model. Used by API status endpoint."""
    global _metadata
    if _metadata:
        return _metadata
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"status": "no model trained yet"}
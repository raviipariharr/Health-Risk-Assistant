"""
CLASSIFIER — Feature Extraction + Prediction (Extended)
=========================================================
Now accepts vitals/lab ML features from the vitals interpreter
and merges them with the keyword-derived features before running
the RandomForest pipeline.

The feature vector now has three layers:
  1. Symptom boolean features (from NLP keywords)     — SYMPTOM_FEATURES
  2. Measurement boolean features (from vitals/labs)  — MEASUREMENT_FEATURES
  3. Numeric composite features                       — NUMERIC_FEATURES
"""

import os
import json
import joblib
import numpy as np
from typing import List, Optional, Dict, Tuple

from app.ml.training.dataset import (
    ALL_FEATURE_NAMES, SYMPTOM_FEATURES, MEASUREMENT_FEATURES,
    NUMERIC_FEATURES, RISK_LABELS
)

MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_classifier.joblib')
META_PATH  = os.path.join(MODEL_DIR, 'model_metadata.json')

_model    = None
_metadata = None


def _load_model():
    global _model, _metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            "Run: python -m app.ml.training.train"
        )
    _model = joblib.load(MODEL_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)
    print(f"[ML] Model loaded: {_metadata.get('model_type')} "
          f"(accuracy={_metadata.get('test_accuracy')})")


# Keyword → feature name (symptom boolean features)
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
    # Vitals-inferred keywords → measurement features
    "hypertension":           "has_hypertension",
    "elevated blood pressure":"has_hypertension",
    "severe hypertension":    "has_hypertensive_crisis",
    "cardiovascular emergency":"has_hypertensive_crisis",
    "low blood pressure":     "has_bradycardia",
    "tachycardia":            "has_tachycardia",
    "bradycardia":            "has_bradycardia",
    "cyanosis":               "has_hypoxaemia",
    "hypoxaemia":             "has_hypoxaemia",
    "high cholesterol":       "has_high_cholesterol",
    "elevated cholesterol":   "has_high_cholesterol",
    "diabetes":               "has_diabetes",
    "hyperglycaemia":         "has_diabetes",
    "hypoglycaemia":          "has_diabetes",
    "kidney failure":         "has_ckd",
    "chronic kidney disease": "has_ckd",
}

SEVERITY_TO_SCORE = {
    "mild": 1, "moderate": 2, "severe": 3, "extreme": 3,
    "intense": 2, "unbearable": 3, "worst": 3, "excruciating": 3,
    "persistent": 2, "constant": 2, "worsening": 2, "sudden": 2,
    "crushing": 3, "stabbing": 2,
}

DURATION_TO_CODE = {
    "hours": 0, "days": 1, "week": 2, "weeks": 3, "months": 4,
}


def extract_features(
    keywords: List[str],
    severity_indicators: List[str],
    duration: str,
    age: Optional[int],
    severity_boost: int = 0,
    contexts: Optional[List[str]] = None,
    vitals_ml_features: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the full feature vector including vitals/lab measurements.
    """
    contexts = contexts or []
    vitals_ml_features = vitals_ml_features or {}

    # Start with zero for all boolean symptom features
    feat: Dict[str, float] = {name: 0.0 for name in SYMPTOM_FEATURES + MEASUREMENT_FEATURES}

    # Map NLP keywords → symptom boolean features
    for kw in keywords:
        fname = KEYWORD_TO_FEATURE.get(kw.lower())
        if fname and fname in feat:
            feat[fname] = 1.0

    # Overlay vitals/lab measurement boolean features (these override if set)
    for mfeat in MEASUREMENT_FEATURES:
        if mfeat in vitals_ml_features:
            feat[mfeat] = vitals_ml_features[mfeat]

    symptom_count = sum(feat[f] for f in SYMPTOM_FEATURES)

    # Severity score
    severity_score = 0
    for indicator in severity_indicators:
        s = SEVERITY_TO_SCORE.get(indicator.lower(), 0)
        severity_score = max(severity_score, s)
    severity_score = min(3, severity_score + severity_boost)

    # Exertional context boost
    is_exertional = 1 if "exertional" in contexts else 0
    if is_exertional and (feat.get("has_chest_pain") or feat.get("has_shortness_of_breath")):
        symptom_count = min(symptom_count + 2, 15)

    duration_code = DURATION_TO_CODE.get(duration, 1)
    age_val = float(age) if age else 40.0

    # Vitals / labs composite scores from the interpreter
    vitals_sev = vitals_ml_features.get("bp_severity", 0)
    if vitals_ml_features.get("has_hypoxaemia"):
        vitals_sev += 2
    if vitals_ml_features.get("has_tachycardia"):
        vitals_sev += 1
    vitals_severity = min(3, int(vitals_sev))

    labs_sev = 0
    if vitals_ml_features.get("has_diabetes"):
        labs_sev += 2
    if vitals_ml_features.get("has_high_cholesterol"):
        labs_sev += 1
    if vitals_ml_features.get("has_ckd"):
        labs_sev += 2
    labs_severity = min(3, labs_sev)

    measurement_count = sum(1 for mf in MEASUREMENT_FEATURES if feat.get(mf, 0) > 0)

    numeric = {
        "age":                      age_val,
        "symptom_count":            symptom_count,
        "severity_score":           severity_score,
        "duration_code":            duration_code,
        "is_acute":                 1 if duration_code == 0 else 0,
        "age_over_60":              1 if age_val >= 60 else 0,
        "has_any_severe_indicator": 1 if severity_score >= 2 else 0,
        "vitals_severity_score":    vitals_severity,
        "labs_severity_score":      labs_severity,
        "measurement_count":        measurement_count,
    }

    full_feat = {**feat, **numeric}
    feature_vector = np.array(
        [full_feat.get(name, 0.0) for name in ALL_FEATURE_NAMES],
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
    vitals_ml_features: Optional[Dict[str, float]] = None,
) -> Dict:
    global _model
    if _model is None:
        try:
            _load_model()
        except FileNotFoundError as e:
            return {
                "score": 0, "level": "low",
                "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0},
                "confidence": "low", "top_features": [],
                "model_available": False, "error": str(e),
            }

    X, feat_dict = extract_features(
        keywords, severity_indicators, duration, age,
        severity_boost=severity_boost,
        contexts=contexts,
        vitals_ml_features=vitals_ml_features,
    )

    pred_class = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]

    class_midpoints = [15.0, 50.0, 85.0]
    score = float(sum(p * m for p, m in zip(proba, class_midpoints)))
    score = round(min(100, max(0, score)))
    level = RISK_LABELS[pred_class]

    max_prob = float(proba[pred_class])
    confidence = "high" if max_prob >= 0.75 else ("medium" if max_prob >= 0.50 else "low")

    rf = _model.named_steps['classifier']
    importances = rf.feature_importances_

    active_features = []
    for i, fname in enumerate(ALL_FEATURE_NAMES):
        val = feat_dict.get(fname, 0)
        if val > 0:
            active_features.append({
                "feature":    fname,
                "value":      float(val),
                "importance": float(round(importances[i], 4)),
            })
    active_features.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "score":         score,
        "level":         level,
        "probabilities": {
            "low":    float(round(proba[0], 3)),
            "medium": float(round(proba[1], 3)),
            "high":   float(round(proba[2], 3)),
        },
        "confidence":     confidence,
        "top_features":   active_features[:8],
        "model_available":True,
        "metadata": {
            "test_accuracy": _metadata.get("test_accuracy") if _metadata else None,
            "model_type":    _metadata.get("model_type") if _metadata else None,
        }
    }


def get_model_info() -> Dict:
    global _metadata
    if _metadata:
        return _metadata
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"status": "no model trained yet"}
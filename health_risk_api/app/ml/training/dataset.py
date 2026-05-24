"""
SYNTHETIC DATASET GENERATOR — Extended with Vitals & Lab Features
===================================================================
Added numeric measurement features so the RandomForest can learn
from real clinical values, not just symptom keywords.

New feature groups:
  Vitals features  — has_hypertension, has_hypertensive_crisis,
                     has_tachycardia, has_hypoxaemia, has_fever_vital,
                     bp_severity, heart_rate_norm, spo2_norm
  Lab features     — has_high_cholesterol, has_low_hdl, has_diabetes,
                     has_prediabetes, has_ckd, chol_score, glucose_score
  Body metrics     — has_obesity, bmi_norm

The oracle label function also uses these when available.
"""

import numpy as np
import pandas as pd
import random
from typing import Tuple

SYMPTOM_FEATURES = [
    # High-risk cardiovascular
    "has_chest_pain",
    "has_shortness_of_breath",
    "has_palpitations",
    "has_irregular_heartbeat",
    "has_radiating_pain",

    # High-risk neurological
    "has_confusion",
    "has_slurred_speech",
    "has_seizure",
    "has_fainting",
    "has_sudden_onset",

    # High-risk general
    "has_blood_in_stool",
    "has_blood_in_urine",
    "has_severe_weight_loss",

    # Medium-risk
    "has_fever",
    "has_high_fever",
    "has_night_sweats",
    "has_dizziness",
    "has_vomiting",
    "has_abdominal_pain",
    "has_numbness",
    "has_blurred_vision",

    # Lower-risk
    "has_headache",
    "has_fatigue",
    "has_nausea",
    "has_cough",
    "has_back_pain",
    "has_joint_pain",
    "has_sore_throat",
    "has_rash",
    "has_stomach_ache",
]

# ---- NEW: Vitals & Lab boolean features ----
MEASUREMENT_FEATURES = [
    # Vitals
    "has_hypertension",         # BP stage 1+
    "has_hypertensive_crisis",  # BP ≥ 180/120
    "has_tachycardia",          # HR > 100
    "has_bradycardia",          # HR < 60
    "has_hypoxaemia",           # SpO2 < 94
    "has_tachypnoea",           # RR ≥ 20
    "has_fever_vital",          # Temp ≥ 37.5
    "has_obesity",              # BMI ≥ 30

    # Labs
    "has_high_cholesterol",     # total ≥ 5.0 or LDL ≥ 3.0
    "has_low_hdl",              # HDL < 1.0
    "has_diabetes",             # HbA1c ≥ 6.5 or fasting glucose ≥ 7.0
    "has_prediabetes",          # HbA1c 5.7-6.4 or fasting 5.6-6.9
    "has_ckd",                  # eGFR < 60
]

NUMERIC_FEATURES = [
    "age",
    "symptom_count",
    "severity_score",
    "duration_code",
    "is_acute",
    "age_over_60",
    "has_any_severe_indicator",
    # New composite scores (0-3 scale)
    "vitals_severity_score",    # sum of vital sign severity levels (capped at 3)
    "labs_severity_score",      # sum of lab abnormality severity levels (capped at 3)
    "measurement_count",        # how many measurements are abnormal
]

ALL_FEATURE_NAMES = SYMPTOM_FEATURES + MEASUREMENT_FEATURES + NUMERIC_FEATURES

RISK_LABELS = {0: "low", 1: "medium", 2: "high"}


def oracle_label(row: dict) -> int:
    # Immediate high-risk: classic cardiac/neuro emergencies
    if row["has_chest_pain"] and row["has_shortness_of_breath"]:
        return 2
    if row["has_chest_pain"] and row["has_radiating_pain"]:
        return 2
    if row["has_slurred_speech"] or row["has_seizure"]:
        return 2
    if row["has_sudden_onset"] and row["has_chest_pain"]:
        return 2
    if row["has_blood_in_stool"] or row["has_blood_in_urine"]:
        return 2
    if row["has_confusion"] and row["age"] >= 60:
        return 2

    # Hypertensive crisis alone is high risk
    if row["has_hypertensive_crisis"]:
        return 2

    # Chest pain + hypertension = high risk
    if row["has_chest_pain"] and row["has_hypertension"]:
        return 2

    # Hypoxaemia + any respiratory/cardiac symptom
    if row["has_hypoxaemia"] and (row["has_shortness_of_breath"] or row["has_chest_pain"]):
        return 2

    # Severe combined measurement abnormality
    if row["vitals_severity_score"] >= 3 and row["labs_severity_score"] >= 2:
        return 2

    # Severe symptoms
    if row["severity_score"] >= 3 and row["symptom_count"] >= 3:
        return 2
    if row["has_chest_pain"] and row["severity_score"] >= 2:
        return 2
    if row["has_palpitations"] and row["has_dizziness"] and row["age"] >= 60:
        return 2

    # ---- Medium risk ----
    if row["symptom_count"] >= 4:
        return 1
    if row["has_fever"] and row["has_shortness_of_breath"]:
        return 1
    if row["has_high_fever"] or row["has_fever_vital"]:
        return 1
    if row["has_night_sweats"] and row["has_severe_weight_loss"]:
        return 1
    if row["has_abdominal_pain"] and row["severity_score"] >= 2:
        return 1
    if row["has_dizziness"] and row["has_fainting"]:
        return 1
    if row["has_numbness"] or row["has_blurred_vision"]:
        return 1
    if row["has_vomiting"] and row["has_fever"]:
        return 1
    if row["age"] >= 65 and row["symptom_count"] >= 2:
        return 1
    if row["has_headache"] and row["severity_score"] >= 2:
        return 1

    # Measurement-based medium risk
    if row["has_hypertension"] and row["has_high_cholesterol"]:
        return 1
    if row["has_diabetes"] and row["symptom_count"] >= 2:
        return 1
    if row["has_tachycardia"] and row["has_palpitations"]:
        return 1
    if row["measurement_count"] >= 4:
        return 1
    if row["vitals_severity_score"] >= 2:
        return 1
    if row["labs_severity_score"] >= 2:
        return 1

    return 0


def generate_patient(rng: random.Random, target_risk: int = None) -> dict:
    row = {}
    age = rng.randint(18, 85)
    row["age"] = age
    row["age_over_60"] = 1 if age >= 60 else 0

    duration_weights = [0.15, 0.35, 0.25, 0.15, 0.10]
    row["duration_code"] = rng.choices(range(5), weights=duration_weights)[0]
    row["is_acute"] = 1 if row["duration_code"] == 0 else 0

    if target_risk == 2:
        sev_weights = [0.05, 0.15, 0.35, 0.45]
    elif target_risk == 1:
        sev_weights = [0.10, 0.40, 0.35, 0.15]
    else:
        sev_weights = [0.40, 0.40, 0.15, 0.05]
    row["severity_score"] = rng.choices([0, 1, 2, 3], weights=sev_weights)[0]
    row["has_any_severe_indicator"] = 1 if row["severity_score"] >= 2 else 0

    high_risk_p   = 0.6 if target_risk == 2 else (0.15 if target_risk == 1 else 0.03)
    medium_risk_p = 0.5 if target_risk == 1 else (0.3  if target_risk == 2 else 0.08)
    low_risk_p    = 0.5 if target_risk == 0 else (0.2  if target_risk == 1 else 0.1)

    probs = {
        "has_chest_pain":         high_risk_p,
        "has_shortness_of_breath":high_risk_p * 0.8,
        "has_palpitations":       high_risk_p * 0.5,
        "has_irregular_heartbeat":high_risk_p * 0.3,
        "has_radiating_pain":     high_risk_p * 0.6,
        "has_confusion":          high_risk_p * 0.4,
        "has_slurred_speech":     high_risk_p * 0.2,
        "has_seizure":            high_risk_p * 0.15,
        "has_fainting":           high_risk_p * 0.4,
        "has_sudden_onset":       high_risk_p * 0.7,
        "has_blood_in_stool":     high_risk_p * 0.3,
        "has_blood_in_urine":     high_risk_p * 0.25,
        "has_severe_weight_loss": high_risk_p * 0.3,
        "has_fever":              medium_risk_p,
        "has_high_fever":         medium_risk_p * 0.4,
        "has_night_sweats":       medium_risk_p * 0.5,
        "has_dizziness":          medium_risk_p * 0.6,
        "has_vomiting":           medium_risk_p * 0.5,
        "has_abdominal_pain":     medium_risk_p * 0.6,
        "has_numbness":           medium_risk_p * 0.4,
        "has_blurred_vision":     medium_risk_p * 0.4,
        "has_headache":           low_risk_p,
        "has_fatigue":            low_risk_p * 0.9,
        "has_nausea":             low_risk_p * 0.7,
        "has_cough":              low_risk_p * 0.8,
        "has_back_pain":          low_risk_p * 0.7,
        "has_joint_pain":         low_risk_p * 0.6,
        "has_sore_throat":        low_risk_p * 0.8,
        "has_rash":               low_risk_p * 0.4,
        "has_stomach_ache":       low_risk_p * 0.6,
    }

    for feat in SYMPTOM_FEATURES:
        row[feat] = 1 if rng.random() < probs[feat] else 0

    # ---- Measurement features ----
    # Probability of having abnormal measurements increases with risk
    meas_p = 0.5 if target_risk == 2 else (0.25 if target_risk == 1 else 0.08)

    meas_probs = {
        "has_hypertension":        meas_p,
        "has_hypertensive_crisis": meas_p * 0.2,
        "has_tachycardia":         meas_p * 0.4,
        "has_bradycardia":         meas_p * 0.1,
        "has_hypoxaemia":          meas_p * 0.3,
        "has_tachypnoea":          meas_p * 0.3,
        "has_fever_vital":         meas_p * 0.5,
        "has_obesity":             0.3 if target_risk != 0 else 0.15,
        "has_high_cholesterol":    meas_p * 0.6,
        "has_low_hdl":             meas_p * 0.4,
        "has_diabetes":            meas_p * 0.4,
        "has_prediabetes":         meas_p * 0.5 if target_risk == 1 else meas_p * 0.2,
        "has_ckd":                 meas_p * 0.3,
    }

    for feat in MEASUREMENT_FEATURES:
        row[feat] = 1 if rng.random() < meas_probs[feat] else 0

    # Vitals severity composite (0-3)
    vitals_sev = (
        row["has_hypertensive_crisis"] * 3
        + row["has_hypertension"] * 1
        + row["has_hypoxaemia"] * 2
        + row["has_tachycardia"] * 1
        + row["has_tachypnoea"] * 1
    )
    row["vitals_severity_score"] = min(3, vitals_sev)

    # Labs severity composite (0-3)
    labs_sev = (
        row["has_high_cholesterol"] * 1
        + row["has_diabetes"] * 2
        + row["has_low_hdl"] * 1
        + row["has_ckd"] * 2
    )
    row["labs_severity_score"] = min(3, labs_sev)

    row["measurement_count"] = sum(row[f] for f in MEASUREMENT_FEATURES)
    row["symptom_count"] = sum(row[f] for f in SYMPTOM_FEATURES)

    row["risk_label"] = oracle_label(row)
    return row


def generate_dataset(n_samples: int = 4000, random_seed: int = 42) -> pd.DataFrame:
    rng = random.Random(random_seed)
    records = []
    per_class = n_samples // 3

    for risk_level in [0, 1, 2]:
        count = 0
        attempts = 0
        while count < per_class and attempts < per_class * 20:
            patient = generate_patient(rng, target_risk=risk_level)
            if patient["risk_label"] == risk_level:
                records.append(patient)
                count += 1
            attempts += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"Generated {len(df)} samples")
    print(f"Label distribution:\n{df['risk_label'].value_counts().sort_index()}")
    return df
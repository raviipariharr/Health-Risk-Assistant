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

NUMERIC_FEATURES = [
    "age",                  # 1–100
    "symptom_count",        # how many symptoms present
    "severity_score",       # 0–3: 0=none, 1=mild, 2=moderate, 3=severe
    "duration_code",        # 0=hours, 1=days, 2=week, 3=weeks, 4=months
    "is_acute",             # 1 if duration_code == 0 (hours)
    "age_over_60",          # 1 if age >= 60 (higher baseline risk)
    "has_any_severe_indicator",  # 1 if severity_score >= 2
]

ALL_FEATURE_NAMES = SYMPTOM_FEATURES + NUMERIC_FEATURES

RISK_LABELS = {0: "low", 1: "medium", 2: "high"}


def oracle_label(row: dict) -> int:
    """
    The ground truth rule engine.
    Determines the "true" risk level for a synthetic patient.
    This replaces human doctor labeling for our synthetic data.

    Returns 0 (low), 1 (medium), 2 (high)
    """
    # Immediate high-risk signals
    if (row["has_chest_pain"] and row["has_shortness_of_breath"]):
        return 2
    if (row["has_chest_pain"] and row["has_radiating_pain"]):
        return 2
    if row["has_slurred_speech"] or row["has_seizure"]:
        return 2
    if (row["has_sudden_onset"] and row["has_chest_pain"]):
        return 2
    if row["has_blood_in_stool"] or row["has_blood_in_urine"]:
        return 2
    if row["has_confusion"] and row["age"] >= 60:
        return 2

    # High-risk with severity
    if row["severity_score"] >= 3 and row["symptom_count"] >= 3:
        return 2
    if row["has_chest_pain"] and row["severity_score"] >= 2:
        return 2
    if row["has_palpitations"] and row["has_dizziness"] and row["age"] >= 60:
        return 2

    # Medium risk
    if row["symptom_count"] >= 4:
        return 1
    if row["has_fever"] and row["has_shortness_of_breath"]:
        return 1
    if row["has_high_fever"]:
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

    return 0  # low risk


def generate_patient(rng: random.Random, target_risk: int = None) -> dict:
    """
    Generate one synthetic patient record.
    
    If target_risk is given, we bias sampling to produce
    that risk level (helps balance the dataset).
    """
    row = {}

    age = rng.randint(18, 85)
    row["age"] = age
    row["age_over_60"] = 1 if age >= 60 else 0

    # Duration
    duration_weights = [0.15, 0.35, 0.25, 0.15, 0.10]  # hours..months
    row["duration_code"] = rng.choices(range(5), weights=duration_weights)[0]
    row["is_acute"] = 1 if row["duration_code"] == 0 else 0

    # Severity
    if target_risk == 2:
        sev_weights = [0.05, 0.15, 0.35, 0.45]
    elif target_risk == 1:
        sev_weights = [0.10, 0.40, 0.35, 0.15]
    else:
        sev_weights = [0.40, 0.40, 0.15, 0.05]
    row["severity_score"] = rng.choices([0, 1, 2, 3], weights=sev_weights)[0]
    row["has_any_severe_indicator"] = 1 if row["severity_score"] >= 2 else 0

    # High-risk symptom probabilities depend on target_risk
    high_risk_p   = 0.6 if target_risk == 2 else (0.15 if target_risk == 1 else 0.03)
    medium_risk_p = 0.5 if target_risk == 1 else (0.3 if target_risk == 2 else 0.08)
    low_risk_p    = 0.5 if target_risk == 0 else (0.2 if target_risk == 1 else 0.1)

    probs = {
        "has_chest_pain": high_risk_p,
        "has_shortness_of_breath": high_risk_p * 0.8,
        "has_palpitations": high_risk_p * 0.5,
        "has_irregular_heartbeat": high_risk_p * 0.3,
        "has_radiating_pain": high_risk_p * 0.6,
        "has_confusion": high_risk_p * 0.4,
        "has_slurred_speech": high_risk_p * 0.2,
        "has_seizure": high_risk_p * 0.15,
        "has_fainting": high_risk_p * 0.4,
        "has_sudden_onset": high_risk_p * 0.7,
        "has_blood_in_stool": high_risk_p * 0.3,
        "has_blood_in_urine": high_risk_p * 0.25,
        "has_severe_weight_loss": high_risk_p * 0.3,
        "has_fever": medium_risk_p,
        "has_high_fever": medium_risk_p * 0.4,
        "has_night_sweats": medium_risk_p * 0.5,
        "has_dizziness": medium_risk_p * 0.6,
        "has_vomiting": medium_risk_p * 0.5,
        "has_abdominal_pain": medium_risk_p * 0.6,
        "has_numbness": medium_risk_p * 0.4,
        "has_blurred_vision": medium_risk_p * 0.4,
        "has_headache": low_risk_p,
        "has_fatigue": low_risk_p * 0.9,
        "has_nausea": low_risk_p * 0.7,
        "has_cough": low_risk_p * 0.8,
        "has_back_pain": low_risk_p * 0.7,
        "has_joint_pain": low_risk_p * 0.6,
        "has_sore_throat": low_risk_p * 0.8,
        "has_rash": low_risk_p * 0.4,
        "has_stomach_ache": low_risk_p * 0.6,
    }

    for feat in SYMPTOM_FEATURES:
        row[feat] = 1 if rng.random() < probs[feat] else 0

    row["symptom_count"] = sum(row[f] for f in SYMPTOM_FEATURES)

    # Get oracle label (may differ from target_risk — that's the noise)
    row["risk_label"] = oracle_label(row)

    return row


def generate_dataset(n_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset.
    
    Why balanced?
    → If 90% of samples are "low risk", the model learns to predict
      "low" for everything and still gets 90% accuracy — useless!
    → Balanced classes force the model to actually learn each category.
    """
    rng = random.Random(random_seed)
    records = []

    # Generate equal thirds for each risk level
    per_class = n_samples // 3

    for risk_level in [0, 1, 2]:
        count = 0
        attempts = 0
        while count < per_class and attempts < per_class * 20:
            patient = generate_patient(rng, target_risk=risk_level)
            # Accept sample if oracle agrees with intended label
            # (rejects noise — gives cleaner training signal)
            if patient["risk_label"] == risk_level:
                records.append(patient)
                count += 1
            attempts += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Generated {len(df)} samples")
    print(f"Label distribution:\n{df['risk_label'].value_counts().sort_index()}")

    return df
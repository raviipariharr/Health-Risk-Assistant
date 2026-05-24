"""
VITALS & LABS INTERPRETER
===========================
Converts raw numeric measurements into:
  1. Clinical flags  — human-readable findings ("Hypertension Stage 2")
  2. Feature dict    — boolean/float features for the ML classifier
  3. Severity boost  — integer added to severity_score when readings are abnormal
  4. Keywords        — symptom keywords inferred from the numbers
                       (feeds into the same NLP pipeline)

All thresholds follow published clinical guidelines:
  Blood pressure : ESC/AHA 2023
  Cholesterol    : NICE/ACC-AHA 2019
  Blood glucose  : WHO / ADA 2023
  SpO2           : BTS / WHO
  BMI            : WHO
  Temperature    : standard febrile thresholds
"""

from typing import Optional, Dict, List, Tuple


# =============================================
# BLOOD PRESSURE CLASSIFICATION
# =============================================

def interpret_blood_pressure(
    systolic: Optional[int],
    diastolic: Optional[int]
) -> Tuple[str, int, List[str]]:
    """
    Returns (label, severity_score 0-3, inferred_keywords).
    """
    if systolic is None or diastolic is None:
        return "unknown", 0, []

    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive crisis", 3, ["severe hypertension", "cardiovascular emergency"]
    if systolic >= 160 or diastolic >= 100:
        return "Hypertension Stage 2", 2, ["hypertension", "elevated blood pressure"]
    if systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 1", 1, ["hypertension", "elevated blood pressure"]
    if systolic >= 130 or diastolic >= 80:
        return "Elevated blood pressure", 1, ["elevated blood pressure"]
    if systolic < 90 or diastolic < 60:
        return "Hypotension", 2, ["low blood pressure", "dizziness"]
    return "Normal", 0, []


# =============================================
# HEART RATE CLASSIFICATION
# =============================================

def interpret_heart_rate(hr: Optional[int]) -> Tuple[str, int, List[str]]:
    if hr is None:
        return "unknown", 0, []
    if hr > 150:
        return "Severe tachycardia", 3, ["rapid heartbeat", "palpitations"]
    if hr > 100:
        return "Tachycardia", 2, ["rapid heartbeat"]
    if hr < 40:
        return "Severe bradycardia", 3, ["irregular heartbeat", "fainting"]
    if hr < 60:
        return "Bradycardia", 1, ["irregular heartbeat"]
    return "Normal", 0, []


# =============================================
# SpO2 CLASSIFICATION
# =============================================

def interpret_spo2(spo2: Optional[float]) -> Tuple[str, int, List[str]]:
    if spo2 is None:
        return "unknown", 0, []
    if spo2 < 90:
        return "Severe hypoxaemia", 3, ["shortness of breath", "difficulty breathing", "cyanosis"]
    if spo2 < 94:
        return "Mild hypoxaemia", 2, ["shortness of breath"]
    if spo2 < 96:
        return "Low-normal SpO2", 1, []
    return "Normal", 0, []


# =============================================
# TEMPERATURE CLASSIFICATION
# =============================================

def interpret_temperature(temp_c: Optional[float]) -> Tuple[str, int, List[str]]:
    if temp_c is None:
        return "unknown", 0, []
    if temp_c >= 40.0:
        return "Hyperpyrexia", 3, ["high fever", "fever"]
    if temp_c >= 38.5:
        return "High fever", 2, ["high fever", "fever"]
    if temp_c >= 37.5:
        return "Low-grade fever", 1, ["fever"]
    if temp_c < 35.0:
        return "Hypothermia", 3, ["weakness", "confusion"]
    return "Normal", 0, []


# =============================================
# RESPIRATORY RATE CLASSIFICATION
# =============================================

def interpret_respiratory_rate(rr: Optional[int]) -> Tuple[str, int, List[str]]:
    if rr is None:
        return "unknown", 0, []
    if rr >= 30:
        return "Severe tachypnoea", 3, ["shortness of breath", "difficulty breathing"]
    if rr >= 20:
        return "Tachypnoea", 2, ["shortness of breath"]
    if rr < 8:
        return "Bradypnoea", 3, ["difficulty breathing"]
    return "Normal", 0, []


# =============================================
# CHOLESTEROL CLASSIFICATION (mmol/L)
# =============================================

def interpret_cholesterol(
    total: Optional[float],
    ldl:   Optional[float],
    hdl:   Optional[float],
    trig:  Optional[float]
) -> Tuple[List[str], int, List[str]]:
    flags = []
    severity = 0
    keywords = []

    if total is not None:
        if total >= 7.5:
            flags.append(f"Very high total cholesterol ({total:.1f} mmol/L)")
            severity = max(severity, 3)
            keywords.append("high cholesterol")
        elif total >= 5.0:
            flags.append(f"Borderline-high total cholesterol ({total:.1f} mmol/L)")
            severity = max(severity, 1)
            keywords.append("elevated cholesterol")

    if ldl is not None:
        if ldl >= 5.0:
            flags.append(f"Very high LDL cholesterol ({ldl:.1f} mmol/L)")
            severity = max(severity, 3)
        elif ldl >= 3.0:
            flags.append(f"Elevated LDL cholesterol ({ldl:.1f} mmol/L)")
            severity = max(severity, 1)

    if hdl is not None:
        if hdl < 1.0:
            flags.append(f"Low HDL cholesterol ({hdl:.1f} mmol/L) — cardiovascular risk factor")
            severity = max(severity, 2)
        elif hdl > 1.6:
            flags.append(f"Protective HDL level ({hdl:.1f} mmol/L)")

    if trig is not None:
        if trig >= 5.6:
            flags.append(f"Very high triglycerides ({trig:.1f} mmol/L) — pancreatitis risk")
            severity = max(severity, 3)
            keywords.append("hypertriglyceridaemia")
        elif trig >= 2.3:
            flags.append(f"Elevated triglycerides ({trig:.1f} mmol/L)")
            severity = max(severity, 1)

    return flags, severity, keywords


# =============================================
# GLUCOSE / HbA1c CLASSIFICATION
# =============================================

def interpret_glucose(
    fasting_glucose: Optional[float],
    hba1c:           Optional[float]
) -> Tuple[List[str], int, List[str]]:
    flags = []
    severity = 0
    keywords = []

    if fasting_glucose is not None:
        if fasting_glucose >= 11.1:
            flags.append(f"Diabetic-range glucose ({fasting_glucose:.1f} mmol/L fasting)")
            severity = max(severity, 3)
            keywords.append("hyperglycaemia")
        elif fasting_glucose >= 7.0:
            flags.append(f"Fasting hyperglycaemia ({fasting_glucose:.1f} mmol/L) — diabetes threshold")
            severity = max(severity, 2)
            keywords.append("elevated blood glucose")
        elif fasting_glucose >= 5.6:
            flags.append(f"Impaired fasting glucose ({fasting_glucose:.1f} mmol/L) — pre-diabetes range")
            severity = max(severity, 1)
        elif fasting_glucose < 3.9:
            flags.append(f"Hypoglycaemia ({fasting_glucose:.1f} mmol/L)")
            severity = max(severity, 3)
            keywords.append("hypoglycaemia")

    if hba1c is not None:
        if hba1c >= 10.0:
            flags.append(f"Severely elevated HbA1c ({hba1c:.1f}%) — very poor glycaemic control")
            severity = max(severity, 3)
        elif hba1c >= 6.5:
            flags.append(f"HbA1c {hba1c:.1f}% — diagnostic of diabetes")
            severity = max(severity, 2)
            keywords.append("diabetes")
        elif hba1c >= 5.7:
            flags.append(f"HbA1c {hba1c:.1f}% — pre-diabetes range")
            severity = max(severity, 1)

    return flags, severity, keywords


# =============================================
# BMI CALCULATION
# =============================================

def calculate_bmi(
    weight_kg: Optional[float],
    height_cm: Optional[float]
) -> Tuple[Optional[float], str, int]:
    if weight_kg is None or height_cm is None:
        return None, "unknown", 0
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi >= 40:
        return bmi, "Morbid obesity", 3
    if bmi >= 35:
        return bmi, "Severe obesity", 2
    if bmi >= 30:
        return bmi, "Obese", 2
    if bmi >= 25:
        return bmi, "Overweight", 1
    if bmi < 18.5:
        return bmi, "Underweight", 1
    return bmi, "Normal weight", 0


# =============================================
# KIDNEY FUNCTION (creatinine / eGFR)
# =============================================

def interpret_kidney(
    creatinine: Optional[float],
    egfr:       Optional[float]
) -> Tuple[List[str], int, List[str]]:
    flags = []
    severity = 0
    keywords = []

    if egfr is not None:
        if egfr < 15:
            flags.append(f"eGFR {egfr:.0f} — kidney failure (Stage 5 CKD)")
            severity = max(severity, 3)
            keywords.append("kidney failure")
        elif egfr < 30:
            flags.append(f"eGFR {egfr:.0f} — severe CKD (Stage 4)")
            severity = max(severity, 3)
            keywords.append("chronic kidney disease")
        elif egfr < 45:
            flags.append(f"eGFR {egfr:.0f} — moderate-severe CKD (Stage 3b)")
            severity = max(severity, 2)
        elif egfr < 60:
            flags.append(f"eGFR {egfr:.0f} — moderate CKD (Stage 3a)")
            severity = max(severity, 1)

    if creatinine is not None:
        if creatinine > 200:
            flags.append(f"Elevated creatinine ({creatinine:.0f} µmol/L)")
            severity = max(severity, 2)

    return flags, severity, keywords


# =============================================
# MASTER INTERPRETER — Called by processor.py
# =============================================

def interpret_vitals_and_labs(
    vitals,   # VitalsInput | None
    labs,     # LabsInput   | None
    age: Optional[int] = None,
    sex: str = "other",
) -> Dict:
    """
    Run all interpreters and aggregate results.

    Returns:
      flags           : list of human-readable clinical findings
      severity_boost  : int (0-10) added to the ML severity_score
      inferred_keywords: list of symptom keywords derived from measurements
      ml_features     : dict of float features for the ML classifier
      summary         : short human-readable paragraph
    """
    all_flags = []
    severity_boost = 0
    inferred_keywords = []
    ml_features = {}

    # ---- Vitals ----
    if vitals:
        # Blood pressure
        bp_label, bp_sev, bp_kw = interpret_blood_pressure(
            vitals.systolic_bp, vitals.diastolic_bp
        )
        if bp_label not in ("unknown", "Normal"):
            all_flags.append(f"BP: {bp_label} ({vitals.systolic_bp}/{vitals.diastolic_bp} mmHg)")
        severity_boost += bp_sev
        inferred_keywords += bp_kw
        ml_features["bp_severity"] = float(bp_sev)
        ml_features["systolic_bp"] = float(vitals.systolic_bp or 120)
        ml_features["diastolic_bp"] = float(vitals.diastolic_bp or 80)
        ml_features["has_hypertension"] = 1.0 if bp_sev >= 1 else 0.0
        ml_features["has_hypertensive_crisis"] = 1.0 if bp_sev >= 3 else 0.0

        # Heart rate
        hr_label, hr_sev, hr_kw = interpret_heart_rate(vitals.heart_rate)
        if hr_label not in ("unknown", "Normal"):
            all_flags.append(f"HR: {hr_label} ({vitals.heart_rate} bpm)")
        severity_boost += hr_sev
        inferred_keywords += hr_kw
        ml_features["heart_rate"] = float(vitals.heart_rate or 75)
        ml_features["has_tachycardia"] = 1.0 if (vitals.heart_rate or 75) > 100 else 0.0
        ml_features["has_bradycardia"] = 1.0 if (vitals.heart_rate or 75) < 60 else 0.0

        # SpO2
        spo2_label, spo2_sev, spo2_kw = interpret_spo2(vitals.spo2)
        if spo2_label not in ("unknown", "Normal", "Low-normal SpO2"):
            all_flags.append(f"SpO2: {spo2_label} ({vitals.spo2:.0f}%)")
        severity_boost += spo2_sev
        inferred_keywords += spo2_kw
        ml_features["spo2"] = float(vitals.spo2 or 98)
        ml_features["has_hypoxaemia"] = 1.0 if spo2_sev >= 2 else 0.0

        # Temperature
        temp_label, temp_sev, temp_kw = interpret_temperature(vitals.temperature_celsius)
        if temp_label not in ("unknown", "Normal"):
            all_flags.append(f"Temperature: {temp_label} ({vitals.temperature_celsius:.1f}°C)")
        severity_boost += temp_sev
        inferred_keywords += temp_kw
        ml_features["temperature"] = float(vitals.temperature_celsius or 37.0)
        ml_features["has_fever_vital"] = 1.0 if temp_sev >= 1 else 0.0

        # Respiratory rate
        rr_label, rr_sev, rr_kw = interpret_respiratory_rate(vitals.respiratory_rate)
        if rr_label not in ("unknown", "Normal"):
            all_flags.append(f"Respiratory rate: {rr_label} ({vitals.respiratory_rate} bpm)")
        severity_boost += rr_sev
        inferred_keywords += rr_kw
        ml_features["respiratory_rate"] = float(vitals.respiratory_rate or 16)
        ml_features["has_tachypnoea"] = 1.0 if rr_sev >= 2 else 0.0

        # BMI
        bmi, bmi_label, bmi_sev = calculate_bmi(vitals.weight_kg, vitals.height_cm)
        if bmi is not None and bmi_label not in ("unknown", "Normal weight"):
            all_flags.append(f"BMI: {bmi_label} ({bmi:.1f} kg/m²)")
        ml_features["bmi"] = float(bmi or 22.0)
        ml_features["has_obesity"] = 1.0 if (bmi or 22) >= 30 else 0.0

    # ---- Labs ----
    if labs:
        # Cholesterol
        chol_flags, chol_sev, chol_kw = interpret_cholesterol(
            labs.total_cholesterol, labs.ldl_cholesterol,
            labs.hdl_cholesterol, labs.triglycerides
        )
        all_flags += chol_flags
        severity_boost += chol_sev
        inferred_keywords += chol_kw
        ml_features["total_cholesterol"] = float(labs.total_cholesterol or 5.0)
        ml_features["ldl_cholesterol"]   = float(labs.ldl_cholesterol or 3.0)
        ml_features["hdl_cholesterol"]   = float(labs.hdl_cholesterol or 1.3)
        ml_features["triglycerides"]     = float(labs.triglycerides or 1.5)
        ml_features["has_high_cholesterol"] = 1.0 if chol_sev >= 2 else 0.0
        ml_features["has_low_hdl"] = 1.0 if (labs.hdl_cholesterol or 1.5) < 1.0 else 0.0

        # Glucose / HbA1c
        gluc_flags, gluc_sev, gluc_kw = interpret_glucose(
            labs.fasting_glucose, labs.hba1c
        )
        all_flags += gluc_flags
        severity_boost += gluc_sev
        inferred_keywords += gluc_kw
        ml_features["fasting_glucose"] = float(labs.fasting_glucose or 5.0)
        ml_features["hba1c"]           = float(labs.hba1c or 5.4)
        ml_features["has_diabetes"]    = 1.0 if gluc_sev >= 2 else 0.0
        ml_features["has_prediabetes"] = 1.0 if gluc_sev == 1 else 0.0

        # Kidney
        kidney_flags, kidney_sev, kidney_kw = interpret_kidney(
            labs.creatinine, labs.egfr
        )
        all_flags += kidney_flags
        severity_boost += kidney_sev
        inferred_keywords += kidney_kw
        ml_features["egfr"]       = float(labs.egfr or 90.0)
        ml_features["creatinine"] = float(labs.creatinine or 80.0)
        ml_features["has_ckd"]    = 1.0 if kidney_sev >= 1 else 0.0

    # Cap severity_boost at 10
    severity_boost = min(severity_boost, 10)

    # Deduplicate inferred keywords
    inferred_keywords = list(dict.fromkeys(inferred_keywords))

    # Build summary
    if all_flags:
        summary = f"Measurements analysis: {len(all_flags)} finding(s) detected. " + \
                  " | ".join(all_flags[:4])
    else:
        summary = "All provided measurements within normal ranges."

    return {
        "flags":              all_flags,
        "severity_boost":     severity_boost,
        "inferred_keywords":  inferred_keywords,
        "ml_features":        ml_features,
        "summary":            summary,
    }
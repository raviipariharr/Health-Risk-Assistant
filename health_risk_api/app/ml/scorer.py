"""
STEP 3: ML RISK SCORING MODULE
================================
This is a rule-based + weighted ML scoring system.

In a production system you'd train a proper classifier
(e.g. scikit-learn RandomForest on labeled symptom datasets).
Here we use a transparent weighted scoring approach so you can
see EXACTLY how the score is calculated — great for learning.

The ML model does two things:
  1. Produces a numeric risk score (0–100)
  2. Identifies which features drove that score (explainability)

This score is passed TO the LLM as context, so the AI can
reason about it alongside its own understanding of the symptoms.

Real-world extension:
  - Replace `score_symptoms_ml()` with a joblib-loaded sklearn model
  - Train on datasets like MIMIC-III or synthetic symptom datasets
  - Use TF-IDF vectorization of symptom text as features
"""

from typing import List, Dict, Tuple


# ---- Risk weight tables ----
# Each symptom keyword has a base risk weight (0.0 to 1.0)
# Higher = more likely to indicate a serious condition

SYMPTOM_RISK_WEIGHTS: Dict[str, float] = {
    # High-risk cardiovascular
    "chest pain": 0.90,
    "chest tightness": 0.85,
    "shortness of breath": 0.75,
    "palpitations": 0.65,
    "irregular heartbeat": 0.80,
    "rapid heartbeat": 0.60,
    "radiating": 0.80,

    # High-risk neurological
    "sudden": 0.85,
    "confusion": 0.80,
    "slurred speech": 0.95,
    "seizure": 0.95,
    "fainting": 0.75,
    "blurred vision": 0.65,
    "numbness": 0.65,
    "weakness": 0.65,

    # Medium-risk general
    "fever": 0.50,
    "high fever": 0.70,
    "night sweats": 0.60,
    "weight loss": 0.65,
    "blood in stool": 0.85,
    "blood in urine": 0.80,
    "rectal bleeding": 0.85,

    # Lower-risk but notable
    "headache": 0.35,
    "migraine": 0.40,
    "dizziness": 0.45,
    "vertigo": 0.45,
    "nausea": 0.30,
    "vomiting": 0.45,
    "fatigue": 0.30,
    "tiredness": 0.25,
    "back pain": 0.30,
    "joint pain": 0.30,
    "cough": 0.25,
    "sore throat": 0.20,
    "runny nose": 0.15,
    "rash": 0.35,
    "stomach ache": 0.25,
    "abdominal pain": 0.45,
    "constipation": 0.20,
    "diarrhea": 0.30,
}

# Severity modifiers — multiply the base score
SEVERITY_MULTIPLIERS: Dict[str, float] = {
    "severe": 1.4,
    "extreme": 1.5,
    "intense": 1.3,
    "unbearable": 1.5,
    "worst": 1.4,
    "excruciating": 1.5,
    "sudden": 1.5,
    "sudden onset": 1.5,
    "crushing": 1.4,
    "stabbing": 1.3,
    "persistent": 1.2,
    "constant": 1.2,
    "getting worse": 1.3,
    "worsening": 1.3,
    "passed out": 1.4,
    "blacked out": 1.4,
    "collapsed": 1.5,
    "can't breathe": 1.5,
    "cannot breathe": 1.5,
    "difficulty breathing": 1.4,
}

# Duration modifiers
DURATION_MODIFIERS: Dict[str, float] = {
    "hours": 1.2,       # acute = potentially more urgent
    "days": 1.0,
    "week": 1.0,
    "weeks": 0.85,      # chronic symptoms slightly less acute risk
    "months": 0.80,
}

# Age-based risk adjustment
# (older patients have higher baseline cardiovascular/cancer risk)
def age_modifier(age: int | None) -> float:
    if age is None:
        return 1.0
    if age < 18:
        return 0.85
    if age < 40:
        return 1.0
    if age < 60:
        return 1.1
    if age < 75:
        return 1.25
    return 1.35


def score_symptoms_ml(
    keywords: List[str],
    severity_indicators: List[str],
    duration: str,
    age: int | None,
    sex: str
) -> Tuple[float, List[str]]:
    """
    Core ML scoring function.

    Returns:
      - raw_score: float 0–100
      - feature_contributions: list of strings explaining what drove the score

    Algorithm:
      1. Look up each detected keyword's base risk weight
      2. Apply severity multiplier (max of all detected severity terms)
      3. Apply duration modifier
      4. Apply age modifier
      5. Combine using weighted average (not just sum, to avoid runaway scores)
      6. Generate explanation strings
    """
    if not keywords:
        return 0.0, ["No recognizable symptom keywords detected."]

    # Step 1: Collect base weights for detected keywords
    keyword_scores = {}
    for kw in keywords:
        weight = SYMPTOM_RISK_WEIGHTS.get(kw, 0.20)  # default 0.20 for unknown
        keyword_scores[kw] = weight

    # Step 2: Severity multiplier (take the highest one)
    sev_mult = 1.0
    top_severity = None
    for sev in severity_indicators:
        mult = SEVERITY_MULTIPLIERS.get(sev, 1.0)
        if mult > sev_mult:
            sev_mult = mult
            top_severity = sev

    # Step 3: Duration modifier
    dur_mod = DURATION_MODIFIERS.get(duration, 1.0)

    # Step 4: Age modifier
    age_mod = age_modifier(age)

    # Step 5: Compute weighted score
    # We use "max + weighted average" to:
    #   - Capture the single most alarming symptom
    #   - Also consider the symptom burden (many symptoms = higher risk)
    max_score = max(keyword_scores.values())
    avg_score = sum(keyword_scores.values()) / len(keyword_scores)

    # Blend: 60% worst symptom, 40% average burden
    blended = (0.60 * max_score) + (0.40 * avg_score)

    # Apply modifiers
    raw_score = blended * sev_mult * dur_mod * age_mod

    # Cap at 1.0 before converting to 0–100
    raw_score = min(raw_score, 1.0)
    final_score = round(raw_score * 100)

    # Step 6: Generate feature contribution explanations
    contributions = []

    # Sort keywords by their contribution (highest first)
    top_kwds = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    for kw, w in top_kwds:
        pct = round(w * 100)
        contributions.append(f'"{kw}" carries a {pct}% base risk weight')

    if top_severity and sev_mult > 1.0:
        pct_increase = round((sev_mult - 1.0) * 100)
        contributions.append(
            f'Severity indicator "{top_severity}" increased risk by {pct_increase}%'
        )

    if age and age >= 60:
        contributions.append(
            f"Age {age} adds a {round((age_mod - 1.0) * 100)}% risk modifier"
        )

    if dur_mod != 1.0:
        dir_str = "elevated" if dur_mod > 1.0 else "reduced"
        contributions.append(
            f"Symptom duration ({duration}) {dir_str} acute risk"
        )

    return float(final_score), contributions


def map_score_to_level(score: float) -> str:
    """Convert numeric score to Low/Medium/High label."""
    if score >= 65:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def get_primary_concern(keywords: List[str], score: float) -> str:
    """
    Pick the most likely primary concern based on top-weighted symptoms.
    In a real system, this would use a multi-label classifier.
    """
    # Map high-weight symptoms to concern categories
    concern_map = {
        "chest pain": "Possible cardiovascular issue — requires urgent evaluation",
        "chest tightness": "Possible cardiovascular issue — requires urgent evaluation",
        "shortness of breath": "Respiratory or cardiovascular compromise",
        "slurred speech": "Possible neurological emergency (stroke symptoms)",
        "seizure": "Neurological emergency",
        "confusion": "Acute neurological or metabolic disturbance",
        "blood in stool": "Gastrointestinal bleeding — requires prompt evaluation",
        "blood in urine": "Urinary tract issue or kidney concern",
        "sudden": "Acute onset symptoms requiring urgent assessment",
        "weight loss": "Unintentional weight loss — warrants further investigation",
        "fever": "Possible infectious or inflammatory process",
        "headache": "Primary headache disorder or secondary cause",
        "fatigue": "Non-specific systemic symptom with broad differential",
        "nausea": "Gastrointestinal or systemic process",
    }
    # Find highest-weight keyword and return its concern
    for kw in sorted(keywords, key=lambda k: SYMPTOM_RISK_WEIGHTS.get(k, 0), reverse=True):
        if kw in concern_map:
            return concern_map[kw]

    return "Non-specific symptoms warranting clinical evaluation"
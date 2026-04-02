"""
ADVANCED NLP PROCESSOR — Contextual Symptom Understanding
===========================================================

The Problem (what you noticed):
  "chest pain during workout at gym" → extracted nothing → LOW risk  ❌
  "Chest pain" chip selected       → extracted chest pain → HIGH risk ✓

  Why? The old processor ONLY looked for exact medical terms.
  It didn't understand:
    - Layman phrases ("my heart was pounding", "felt tight in chest")
    - Context triggers ("during workout", "after running", "at rest")
    - Implied symptoms ("couldn't catch my breath" = shortness of breath)
    - Symptom + context = different risk  (chest pain at rest vs during exercise)

The Solution — 5 upgrades:
  1. SYNONYM EXPANSION  — maps everyday language → medical terms
       "pounding heart"    → palpitations
       "tight chest"       → chest tightness
       "can't catch breath"→ shortness of breath

  2. CONTEXT DETECTION — extracts WHERE/WHEN symptoms happen
       "during workout", "at rest", "after eating", "lying down"
       Context changes risk: chest pain during exertion → much higher cardiac risk

  3. CONTEXTUAL INFERENCE — symptom + context → inferred symptoms
       "chest pain" + "during workout" → also infer "exertional chest pain" (high risk flag)
       "dizziness" + "standing up"     → infer "orthostatic hypotension"

  4. INTENSITY LANGUAGE — natural severity phrases
       "a little", "mild"      → severity 1
       "quite bad", "pretty"   → severity 2
       "unbearable", "worst"   → severity 3

  5. BODY PART MAPPING — anatomical descriptions → symptoms
       "left arm pain"   → radiating pain (cardiac red flag)
       "behind my eyes"  → headache location
       "in my jaw"       → jaw pain (cardiac symptom)
"""

import re
from typing import List, Tuple, Dict, Optional
from app.nlp.linguistic import run_linguistic_analysis


# =============================================
# 1. MEDICAL KEYWORD DICTIONARY (unchanged base)
# =============================================
SYMPTOM_KEYWORDS: Dict[str, List[str]] = {
    "cardiovascular": [
        "chest pain", "chest tightness", "palpitations", "heart racing",
        "shortness of breath", "breathlessness", "irregular heartbeat",
        "swollen ankles", "swollen feet", "rapid heartbeat", "jaw pain",
        "left arm pain", "exertional chest pain", "chest pressure",
        "heart pounding", "skipped beats"
    ],
    "neurological": [
        "headache", "migraine", "dizziness", "vertigo", "confusion",
        "memory loss", "numbness", "tingling", "weakness", "seizure",
        "fainting", "blurred vision", "double vision", "slurred speech",
        "lightheaded", "lightheadedness"
    ],
    "respiratory": [
        "cough", "dry cough", "wet cough", "wheezing", "difficulty breathing",
        "shortness of breath", "chest congestion", "sore throat",
        "runny nose", "nasal congestion", "loss of smell", "breathless"
    ],
    "gastrointestinal": [
        "nausea", "vomiting", "diarrhea", "constipation", "stomach ache",
        "abdominal pain", "bloating", "acid reflux", "heartburn",
        "loss of appetite", "blood in stool", "rectal bleeding"
    ],
    "general": [
        "fever", "fatigue", "tiredness", "exhaustion", "weight loss",
        "weight gain", "night sweats", "chills", "loss of appetite",
        "general weakness", "malaise"
    ],
    "musculoskeletal": [
        "back pain", "joint pain", "muscle ache", "muscle pain", "stiffness",
        "swelling", "arthritis", "neck pain", "shoulder pain", "knee pain"
    ],
    "skin": [
        "rash", "itching", "hives", "redness", "bruising",
        "yellowing", "jaundice", "skin lesion", "discoloration"
    ],
    "urinary": [
        "frequent urination", "painful urination", "blood in urine",
        "difficulty urinating", "urinary urgency"
    ]
}

ALL_KEYWORDS = [kw for kws in SYMPTOM_KEYWORDS.values() for kw in kws]


# =============================================
# 2. SYNONYM / LAYMAN PHRASE EXPANSION
#    Maps how people ACTUALLY describe symptoms
#    → the medical term the model understands
# =============================================
SYNONYM_MAP: Dict[str, str] = {
    # --- Chest / Heart ---
    r"tight(ness)? (in|around|on) (my )?(chest|heart)": "chest tightness",
    r"(my )?chest (feels? )?(tight|heavy|squeezed|compressed|constricted)": "chest tightness",
    r"(felt|feel|feeling) (a )?(tightness|heaviness|pressure|squeezing) (in|on|around) (my )?(chest|heart)": "chest tightness",
    r"pressure (in|on|around) (my )?chest": "chest pressure",
    r"chest (pressure|heaviness|squeezing|crushing)": "chest pressure",
    r"pain in (my )?(chest|heart area|left side)": "chest pain",
    r"(my )?heart (is |was )?(pounding|racing|beating fast|going crazy|hammering|thumping)": "palpitations",
    r"(my )?heart (skipped?|missed?) (a )?beat": "palpitations",
    r"(my )?heart (feels?|felt) (like it('s| is| was))? (going to (explode|burst)|out of control)": "palpitations",
    r"(feel|felt|feeling) (my )?heart(beat| beat| beating| pounding)": "palpitations",
    r"(my )?(left arm|arm) (is |was |feels? )?(numb|numb|tingling|heavy|aching|painful|hurting)": "left arm pain",
    r"pain (shooting |radiating )?(down |into |to )?(my )?(left arm|arm|jaw|shoulder)": "radiating pain",
    r"(jaw|teeth) (pain|ache|hurting|discomfort)": "jaw pain",

    # --- Breathing ---
    r"(can('t| not)|cannot|couldn('t| not)) (catch|get) (my |enough )?(breath|air)": "shortness of breath",
    r"(out of|running out of) breath": "shortness of breath",
    r"(hard|difficult|struggling) to breathe": "shortness of breath",
    r"(feel|felt|feeling) (breathless|winded|suffocated)": "shortness of breath",
    r"(gasping|gasped) (for breath|for air)": "shortness of breath",
    r"breath(ing)? (feels? )?(short|labored|difficult|shallow|heavy)": "shortness of breath",
    r"winded (after|from|during|when)": "shortness of breath",

    # --- Dizziness / Head ---
    r"(feel|felt|feeling) (dizzy|woozy|faint|lightheaded|light.headed|giddy)": "dizziness",
    r"(room|everything) (was |is |felt )?(spinning|going round|tilting)": "vertigo",
    r"(head|everything) (was |is )?(spinning|whirling)": "vertigo",
    r"(almost |nearly )?(passed out|blacked out|fainted|collapsed|fell)": "fainting",
    r"(nearly |almost )?(lost consciousness|went unconscious)": "fainting",
    r"(saw |seeing |vision went) (spots|stars|black|dark|blurry)": "blurred vision",
    r"(eyes|vision) (went |went |is |are )?(blurry|blurred|fuzzy|dark)": "blurred vision",
    r"(had a |got a )?(throbbing|pounding|splitting|terrible|bad|horrible) headache": "headache",
    r"(head|skull) (was |is |feels? )?(throbbing|pounding|splitting)": "headache",

    # --- Fatigue / Energy ---
    r"(feel|felt|feeling) (exhausted|drained|wiped out|run down|burned? out|dead tired|knackered)": "fatigue",
    r"(no |lack of |lost my )energy": "fatigue",
    r"(can('t| not)|cannot|couldn('t| not)) (get out of bed|function|do anything)": "fatigue",
    r"(so|very|extremely) (tired|weak|lethargic)": "fatigue",

    # --- Nausea / Stomach ---
    r"(feel|felt|feeling|been) (sick|nauseous|queasy|nauseated)": "nausea",
    r"(want(ed)? to|about to|going to|felt like) (throw up|vomit|be sick)": "nausea",
    r"(threw|been) (up|sick)": "vomiting",
    r"(upset|churning|cramping|hurting|aching) stomach": "stomach ache",
    r"stomach (cramps?|spasms?|knots?)": "abdominal pain",
    r"(pain|ache|cramps?) in (my )?(stomach|belly|abdomen|gut|tummy)": "abdominal pain",

    # --- Fever / Temperature ---
    r"(running|have|had|got) a (temperature|fever|high temp)": "fever",
    r"(feel|felt|feeling) (hot|burning up|feverish|boiling)": "fever",
    r"(body|skin) (feels? )?(hot|burning|on fire)": "fever",
    r"(shivering|shaking|trembling) (with cold|uncontrollably)": "chills",
    r"(chills?|cold sweats?|night sweats?)": "night sweats",

    # --- Miscellaneous ---
    r"(coughing|been coughing) (a lot|constantly|non.?stop|all night)": "cough",
    r"(throat|neck) (is |feels? )?(sore|painful|scratchy|raw|tight)": "sore throat",
    r"(weak|weakness) (in |of )?(my )?(arms?|legs?|hands?|body|muscles?)": "weakness",
    r"(muscles?|body) (feels? )?(weak|sore|achy|stiff)": "muscle ache",
    r"(joints?|knees?|hips?|elbows?|wrists?) (are |feel |feels? )?(sore|aching|painful|swollen|stiff)": "joint pain",
    r"(back|spine|lower back) (is |feels? )?(sore|aching|painful|killing me|hurting)": "back pain",
    r"(skin|body) (has |with )?(a )?(rash|spots|hives|bumps|redness)": "rash",
}


# =============================================
# 3. CONTEXT TRIGGERS — WHERE / WHEN / HOW
#    These don't map to symptoms but change risk
# =============================================
CONTEXT_PATTERNS: Dict[str, List[str]] = {
    "exertional": [
        r"during (workout|exercise|gym|running|jogging|cardio|training|sport|football|basketball|cycling|swimming)",
        r"(while|when) (working out|exercising|running|jogging|at the gym|playing sport|lifting|training)",
        r"after (workout|exercise|gym|running|jogging|physical activity|training)",
        r"(at|in) (the )?gym",
        r"(on exertion|with exertion|upon exertion)",
        r"(climbing stairs?|walking fast|walking uphill)",
        r"during (physical activity|physical effort|strenuous activity)",
        r"(went for a (run|jog|walk|cycle|swim))",
        r"(playing|played) (sport|football|basketball|tennis|squash|badminton)",
    ],
    "at_rest": [
        r"(at rest|while resting|lying down|in bed|sitting still|doing nothing)",
        r"(woke me up|woke up|waking up) (at night|from sleep|with)",
        r"(middle of the night|at night|during sleep|while sleeping)",
    ],
    "positional": [
        r"(when|while) (lying|laying) (down|flat|on my back|on my side)",
        r"(when|while) (standing up|getting up|sitting up)",
        r"(bending|leaning) (over|forward|down)",
    ],
    "after_eating": [
        r"(after|following) (eating|meals?|food|lunch|dinner|breakfast)",
        r"(when|while) (eating|drinking)",
    ],
    "sudden_onset": [
        r"(came on |started )?(suddenly|all of a sudden|out of nowhere|without warning)",
        r"(just started|brand new|never had (this|it) before)",
        r"(hit me|came on) (suddenly|like a (bolt|wave|rush))",
    ],
    "progressive": [
        r"(getting|been getting|has been getting) (worse|more severe|stronger|more frequent)",
        r"(worsening|progressing|spreading|increasing)",
        r"(worse (than|than ever)|never been this bad)",
    ],
}

# =============================================
# 4. CONTEXTUAL INFERENCE RULES
#    symptom + context → infer additional symptoms / raise severity
#    Each rule: (required_keyword, required_context, inferred_keyword, severity_boost)
# =============================================
INFERENCE_RULES = [
    # Chest pain during exertion is a red flag for cardiac ischaemia
    ("chest pain",          "exertional",   "exertional chest pain",    2),
    ("chest tightness",     "exertional",   "exertional chest pain",    2),
    ("chest pressure",      "exertional",   "exertional chest pain",    2),

    # Shortness of breath during exertion vs at rest — both notable
    ("shortness of breath", "exertional",   "exertional dyspnoea",      1),
    ("shortness of breath", "at_rest",      "resting dyspnoea",         2),

    # Palpitations during exertion can indicate arrhythmia
    ("palpitations",        "exertional",   "exertional palpitations",  1),

    # Dizziness on standing = orthostatic hypotension
    ("dizziness",           "positional",   "orthostatic dizziness",    1),

    # Fainting during exertion = serious
    ("fainting",            "exertional",   "exertional syncope",       3),

    # Headache at rest, sudden onset = red flag
    ("headache",            "sudden_onset", "thunderclap headache",     2),

    # Progressive symptoms are higher risk than static
    ("chest pain",          "progressive",  "progressive chest pain",   2),
    ("headache",            "progressive",  "progressive headache",     1),
    ("shortness of breath", "progressive",  "progressive dyspnoea",     2),
]

# =============================================
# 5. INTENSITY / SEVERITY LANGUAGE
# =============================================
SEVERITY_INDICATORS = [
    # Explicit severity words
    "severe", "extreme", "intense", "unbearable", "worst", "excruciating",
    "crushing", "stabbing", "radiating", "persistent", "constant",
    "getting worse", "worsening", "can't breathe", "cannot breathe",
    "passed out", "blacked out", "collapsed", "difficulty breathing",

    # Natural language severity phrases
    "really bad", "very bad", "so bad", "pretty bad", "quite bad",
    "really painful", "very painful", "extremely painful",
    "really severe", "very severe",
    "worst ever", "never felt this bad", "never been this bad",
    "sudden", "sudden onset", "all of a sudden", "out of nowhere",
]

# Maps intensity phrases to a severity score modifier
INTENSITY_MODIFIERS: Dict[str, float] = {
    r"(a little|slightly|bit of a|mild|minor|slight)": 0.5,       # mild
    r"(moderate|moderate|medium|some|somewhat|fairly)": 1.0,       # baseline
    r"(quite|rather|pretty|significantly|noticeably)": 1.5,        # elevated
    r"(very|really|extremely|terribly|badly|so)": 2.0,             # high
    r"(severe|unbearable|worst|excruciating|agonising|agonizing)": 3.0,  # max
}

# =============================================
# 6. BODY PART → SYMPTOM MAPPING
#    "left arm hurts" → infer "left arm pain" (cardiac red flag)
# =============================================
BODY_PART_PATTERNS = [
    # Left arm / jaw during chest complaints = cardiac referral
    (r"(pain|ache|discomfort|numb|tingling|heavy).{0,20}(left arm|arm|jaw|neck|shoulder)", "left arm pain"),
    (r"(left arm|jaw|neck|shoulder).{0,20}(pain|ache|hurt|numb|tingle|heavy)", "left arm pain"),
]

# =============================================
# NEGATION WORDS
# =============================================
NEGATION_WORDS = [
    "no", "not", "without", "don't have", "dont have", "no sign of",
    "haven't had", "havent had", "never had", "absence of", "denies",
    "no longer", "stopped", "went away", "resolved"
]


# =============================================
# CORE NLP FUNCTIONS
# =============================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # Abbreviation expansion
    abbreviations = {
        r'\bsob\b': 'shortness of breath',
        r'\bcp\b': 'chest pain',
        r'\bha\b': 'headache',
        r'\bn/v\b': 'nausea and vomiting',
        r'\bsob\b': 'shortness of breath',
        r'\bbp\b': 'blood pressure',
        r'\bhr\b': 'heart rate',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    return text


def detect_negations(text: str) -> List[Tuple[int, int]]:
    negated_spans = []
    for neg_word in NEGATION_WORDS:
        pattern = rf'\b{re.escape(neg_word)}\b(.{{0,50}})'
        for match in re.finditer(pattern, text):
            negated_spans.append((match.start(), match.end()))
    return negated_spans


def is_in_negated_span(pos: int, negated_spans: List[Tuple[int, int]]) -> bool:
    return any(start <= pos <= end for start, end in negated_spans)


def expand_synonyms(text: str) -> Tuple[str, List[str]]:
    """
    Replace layman phrases with medical terms.
    Returns (expanded_text, list_of_inferred_terms).

    Example:
      "I couldn't catch my breath at the gym"
      → "I shortness of breath at the gym"
      + inferred: ["shortness of breath"]
    """
    expanded = text
    inferred = []

    for pattern, medical_term in SYNONYM_MAP.items():
        if re.search(pattern, expanded, re.IGNORECASE):
            inferred.append(medical_term)
            # Replace the phrase in text so keyword extractor also finds it
            expanded = re.sub(pattern, f" {medical_term} ", expanded, flags=re.IGNORECASE)

    return expanded, inferred


def detect_contexts(text: str) -> List[str]:
    """
    Identify contextual triggers in the text.
    Returns list of context keys present.

    Example: "chest pain during workout" → ["exertional"]
    """
    found_contexts = []
    normalized = normalize_text(text)

    for context_key, patterns in CONTEXT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                found_contexts.append(context_key)
                break  # one match per context key is enough

    return list(set(found_contexts))


def apply_inference_rules(
    keywords: List[str],
    contexts: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Apply inference rules: symptom + context → inferred symptoms.

    Returns:
      - enriched keyword list (original + inferred)
      - list of inference explanations (for UI display)
      - severity boost integer (how much to add to severity score)
    """
    enriched = list(keywords)
    explanations = []
    total_boost = 0

    for required_kw, required_ctx, inferred_kw, boost in INFERENCE_RULES:
        if required_kw in keywords and required_ctx in contexts:
            if inferred_kw not in enriched:
                enriched.append(inferred_kw)
                explanations.append(
                    f'"{required_kw}" + context "{required_ctx}" → inferred "{inferred_kw}"'
                )
                total_boost += boost

    return enriched, explanations, total_boost


def detect_body_part_symptoms(text: str, negated_spans: List[Tuple[int, int]]) -> List[str]:
    """
    Infer symptoms from anatomical descriptions.
    "left arm pain" → "left arm pain" (cardiac red flag)
    """
    normalized = normalize_text(text)
    found = []
    for pattern, symptom in BODY_PART_PATTERNS:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            if not is_in_negated_span(match.start(), negated_spans):
                if symptom not in found:
                    found.append(symptom)
    return found


def extract_keywords(text: str) -> List[str]:
    """
    Base keyword extraction (dictionary lookup).
    Same as before — synonyms are pre-expanded before this runs.
    """
    normalized = normalize_text(text)
    negated_spans = detect_negations(normalized)
    found = []

    sorted_keywords = sorted(ALL_KEYWORDS, key=len, reverse=True)
    for keyword in sorted_keywords:
        pattern = rf'\b{re.escape(keyword)}\b'
        for match in re.finditer(pattern, normalized):
            if not is_in_negated_span(match.start(), negated_spans):
                if keyword not in found:
                    found.append(keyword)
    return found


def detect_severity_indicators(text: str) -> List[str]:
    normalized = normalize_text(text)
    found = []
    for indicator in SEVERITY_INDICATORS:
        if re.search(rf'\b{re.escape(indicator)}\b', normalized):
            found.append(indicator)
    return found


def compute_intensity_score(text: str) -> float:
    """
    Compute a 0–3 intensity score from natural language.
    Used to enrich the severity_score feature for ML.
    """
    normalized = normalize_text(text)
    max_score = 1.0  # default: baseline
    for pattern, score in INTENSITY_MODIFIERS.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            max_score = max(max_score, score)
    return max_score


def normalize_duration(duration_code: str) -> str:
    mapping = {
        "hours": "a few hours (acute onset)",
        "days": "1–3 days (subacute)",
        "week": "about a week",
        "weeks": "2–4 weeks (chronic)",
        "months": "several months (chronic/recurring)"
    }
    return mapping.get(duration_code, duration_code)


def generate_highlighted_html(original_text: str, keywords: List[str]) -> str:
    """Wrap detected keywords in HTML <span class='kw'> for frontend."""
    if not keywords or not original_text:
        return original_text

    text_lower = original_text.lower()
    highlight_positions = set()

    for keyword in keywords:
        pattern = rf'\b{re.escape(keyword)}\b'
        for match in re.finditer(pattern, text_lower):
            for i in range(match.start(), match.end()):
                highlight_positions.add(i)

    result = []
    in_highlight = False
    for i, char in enumerate(original_text):
        if i in highlight_positions and not in_highlight:
            result.append("<span class='kw'>")
            in_highlight = True
        elif i not in highlight_positions and in_highlight:
            result.append("</span>")
            in_highlight = False
        result.append(char)
    if in_highlight:
        result.append("</span>")

    return "".join(result)


def generate_symptom_summary(
    keywords: List[str],
    severity_indicators: List[str],
    contexts: List[str],
    inferences: List[str],
    duration: str,
    age: Optional[int],
    sex: str
) -> str:
    parts = []

    if keywords:
        kw_str = ", ".join(keywords[:5])
        parts.append(f"Key symptoms identified: {kw_str}.")

    if contexts:
        ctx_str = ", ".join(c.replace("_", " ") for c in contexts)
        parts.append(f"Context detected: {ctx_str}.")

    if inferences:
        parts.append(f"Inferred: {inferences[0]}.")

    if severity_indicators:
        si_str = ", ".join(severity_indicators[:2])
        parts.append(f"Severity indicators: {si_str}.")

    if age:
        parts.append(f"Patient is {age} years old ({sex}).")

    parts.append(f"Duration: {duration}.")
    return " ".join(parts)


# =============================================
# MAIN PIPELINE FUNCTION
# =============================================

def process_symptoms(
    symptom_text: str,
    selected_chips: List[str],
    age: Optional[int],
    sex: str,
    duration: str
) -> dict:
    """
    Full advanced NLP pipeline.

    Flow:
      raw text
        → synonym expansion   (layman → medical terms)
        → keyword extraction  (dictionary lookup on expanded text)
        → body part inference (left arm pain → radiating pain)
        → context detection   (during workout, at rest, sudden onset)
        → inference rules     (chest pain + exertional → exertional chest pain)
        → severity detection  (severity words + intensity modifiers)
        → HTML highlighting   (for frontend display)
        → summary generation
    """
    # Combine chips + free text
    combined_raw = ". ".join(selected_chips + ([symptom_text] if symptom_text else []))

    # STEP 1: Synonym expansion
    expanded_text, synonym_inferences = expand_synonyms(combined_raw)

    # STEP 2: Base keyword extraction (on expanded text)
    base_keywords = extract_keywords(expanded_text)

    # STEP 3: Body part symptom inference
    negated_spans = detect_negations(normalize_text(expanded_text))
    body_part_symptoms = detect_body_part_symptoms(expanded_text, negated_spans)

    # Merge synonym inferences + body part inferences into keywords
    all_keywords = list(base_keywords)
    for term in synonym_inferences + body_part_symptoms:
        if term not in all_keywords:
            all_keywords.append(term)

    # STEP 4: spaCy LINGUISTIC ANALYSIS
    # Lemmatization, POS tagging, body-symptom pairs, onset patterns, quantities
    ling = run_linguistic_analysis(symptom_text, combined_raw)

    # Merge linguistic keywords into all_keywords
    for term in ling["additional_keywords"]:
        if term not in all_keywords:
            all_keywords.append(term)

    # STEP 5: Context detection
    contexts = detect_contexts(combined_raw)

    # Merge onset patterns into contexts
    onset = ling.get("onset_patterns", {})
    if onset.get("sudden"):
        contexts = list(set(contexts + ["sudden_onset"]))
    if onset.get("progressive"):
        contexts = list(set(contexts + ["progressive"]))

    # STEP 6: Inference rules (keyword + context → new keyword)
    enriched_keywords, inference_explanations, severity_boost = apply_inference_rules(
        all_keywords, contexts
    )

    # STEP 7: Severity — combine all severity signals
    severity_indicators = detect_severity_indicators(combined_raw)
    intensity_score = compute_intensity_score(combined_raw)

    # Boost severity from linguistic adjective analysis
    ling_severity = ling.get("severity_from_adj", 0)
    combined_severity_boost = severity_boost + (1 if ling_severity >= 3 else 0)

    # Boost from quantities (pain scale ≥ 7, tachycardia)
    quantities = ling.get("quantities", {})
    if quantities.get("severe_pain") or quantities.get("severe_tachycardia"):
        combined_severity_boost += 1

    # STEP 8: Duration
    duration_str = normalize_duration(duration)

    # STEP 9: Highlighted HTML (highlight on original text only)
    highlighted = generate_highlighted_html(symptom_text, enriched_keywords)

    # STEP 10: Summary (now includes linguistic findings)
    summary = generate_symptom_summary(
        enriched_keywords, severity_indicators, contexts,
        inference_explanations, duration_str, age, sex
    )
    if ling["linguistic_summary"]:
        summary += " " + ling["linguistic_summary"] + "."

    return {
        "detected_keywords":      enriched_keywords,
        "base_keywords":          base_keywords,
        "synonym_inferences":     synonym_inferences,
        "body_part_symptoms":     body_part_symptoms,
        "inference_explanations": inference_explanations,
        "severity_indicators":    severity_indicators,
        "contexts":               contexts,
        "onset_patterns":         onset,
        "quantities":             quantities,
        "intensity_score":        intensity_score,
        "severity_boost":         combined_severity_boost,
        "ling_severity":          ling_severity,
        "linguistic_summary":     ling["linguistic_summary"],
        "symptom_summary":        summary,
        "parsed_highlights":      highlighted,
        "duration_context":       duration_str,
        "combined_text":          combined_raw,
        "keyword_count":          len(enriched_keywords),
        "has_severe_indicators":  len(severity_indicators) > 0,
    }
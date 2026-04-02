"""
STEP 2: NLP PROCESSING MODULE
================================
This is where raw symptom text gets turned into structured data.

NLP Pipeline:
  1. Text normalization (lowercase, clean punctuation)
  2. Medical keyword extraction (dictionary lookup + regex)
  3. Severity indicator detection ("severe", "sudden", "worst ever")
  4. Duration normalization
  5. Negation detection ("no fever", "no headache" → ignored)
  6. Highlight generation (HTML with <span> tags for the UI)

Why not just send raw text to the LLM?
  - Pre-processing catches structured info reliably (numbers, negations)
  - It's faster and cheaper (less tokens to LLM)
  - The ML model (Step 3) needs structured features, not raw text
  - Explainability: we can show exactly what was extracted
"""

import re
from typing import List, Tuple, Dict


# ---- Medical keyword dictionary ----
# Organized by body system — this is our "vocabulary"
SYMPTOM_KEYWORDS: Dict[str, List[str]] = {
    "cardiovascular": [
        "chest pain", "chest tightness", "palpitations", "heart racing",
        "shortness of breath", "breathlessness", "irregular heartbeat",
        "swollen ankles", "swollen feet", "rapid heartbeat"
    ],
    "neurological": [
        "headache", "migraine", "dizziness", "vertigo", "confusion",
        "memory loss", "numbness", "tingling", "weakness", "seizure",
        "fainting", "blurred vision", "double vision", "slurred speech"
    ],
    "respiratory": [
        "cough", "dry cough", "wet cough", "wheezing", "difficulty breathing",
        "shortness of breath", "chest congestion", "sore throat",
        "runny nose", "nasal congestion", "loss of smell"
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
        "rash", "itching", "hives", "redness", "swelling", "bruising",
        "yellowing", "jaundice", "skin lesion", "discoloration"
    ],
    "urinary": [
        "frequent urination", "painful urination", "blood in urine",
        "difficulty urinating", "urinary urgency"
    ]
}

# Flatten all keywords into one list for easy matching
ALL_KEYWORDS = [kw for kws in SYMPTOM_KEYWORDS.values() for kw in kws]

# High-severity indicator words — raise the risk score
SEVERITY_INDICATORS = [
    "severe", "extreme", "intense", "unbearable", "worst", "excruciating",
    "sudden", "sudden onset", "crushing", "stabbing", "radiating",
    "persistent", "constant", "getting worse", "worsening",
    "can't breathe", "cannot breathe", "difficulty breathing",
    "passed out", "blacked out", "collapsed"
]

# Negation words — if a symptom follows these, it should be ignored
NEGATION_WORDS = [
    "no", "not", "without", "don't have", "dont have", "no sign of",
    "haven't had", "havent had", "never had", "absence of", "denies"
]


def normalize_text(text: str) -> str:
    """
    Clean the text for processing.
    - Lowercase
    - Normalize whitespace
    - Expand common abbreviations
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces
    text = re.sub(r'[^\w\s\.,\-]', '', text)  # remove special chars

    # Expand abbreviations
    abbreviations = {
        r'\bsob\b': 'shortness of breath',
        r'\bcp\b': 'chest pain',
        r'\bha\b': 'headache',
        r'\bn/v\b': 'nausea and vomiting',
        r'\bha\b': 'headache',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)

    return text


def detect_negations(text: str) -> List[Tuple[int, int]]:
    """
    Find spans in the text that are negated.
    Returns list of (start, end) character positions to ignore.

    Example: "no headache" → the word "headache" is in a negated span
    """
    negated_spans = []
    for neg_word in NEGATION_WORDS:
        pattern = rf'\b{re.escape(neg_word)}\b(.{{0,40}})'
        for match in re.finditer(pattern, text):
            # Mark the 40 chars after the negation word as negated
            negated_spans.append((match.start(), match.end()))
    return negated_spans


def is_in_negated_span(pos: int, negated_spans: List[Tuple[int, int]]) -> bool:
    """Check if a text position falls within a negated region."""
    return any(start <= pos <= end for start, end in negated_spans)


def extract_keywords(text: str) -> List[str]:
    """
    Core NLP function: find medical symptom keywords in text.

    Algorithm:
    1. Normalize the text
    2. Find negated regions (skip symptoms inside them)
    3. For each keyword in our dictionary, check if it appears
    4. Return unique matches in order of appearance
    """
    normalized = normalize_text(text)
    negated_spans = detect_negations(normalized)
    found = []

    # Sort by length (longest first) to prefer "chest pain" over "pain"
    sorted_keywords = sorted(ALL_KEYWORDS, key=len, reverse=True)

    for keyword in sorted_keywords:
        pattern = rf'\b{re.escape(keyword)}\b'
        for match in re.finditer(pattern, normalized):
            pos = match.start()
            if not is_in_negated_span(pos, negated_spans):
                if keyword not in found:
                    found.append(keyword)

    return found


def detect_severity_indicators(text: str) -> List[str]:
    """
    Look for words/phrases that indicate symptom severity.
    These are important features for risk scoring.
    """
    normalized = normalize_text(text)
    found = []
    for indicator in SEVERITY_INDICATORS:
        if re.search(rf'\b{re.escape(indicator)}\b', normalized):
            found.append(indicator)
    return found


def normalize_duration(duration_code: str) -> str:
    """Convert the dropdown code into a human-readable string for the model."""
    mapping = {
        "hours": "a few hours (acute onset)",
        "days": "1–3 days (subacute)",
        "week": "about a week",
        "weeks": "2–4 weeks (chronic)",
        "months": "several months (chronic/recurring)"
    }
    return mapping.get(duration_code, duration_code)


def generate_highlighted_html(original_text: str, keywords: List[str]) -> str:
    """
    Wrap detected keywords in HTML <span> tags for the frontend UI.
    Handles overlapping matches correctly.
    """
    if not keywords or not original_text:
        return original_text

    text_lower = original_text.lower()
    # Track which character positions should be highlighted
    highlight_positions = set()

    for keyword in keywords:
        pattern = rf'\b{re.escape(keyword)}\b'
        for match in re.finditer(pattern, text_lower):
            for i in range(match.start(), match.end()):
                highlight_positions.add(i)

    # Rebuild text with <span> tags around highlighted regions
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


def generate_symptom_summary(keywords: List[str], severity_indicators: List[str],
                              duration: str, age: int | None, sex: str) -> str:
    """Build a plain-language summary of what was detected."""
    parts = []

    if keywords:
        kw_str = ", ".join(keywords[:5])
        parts.append(f"Key symptoms identified: {kw_str}.")

    if severity_indicators:
        si_str = ", ".join(severity_indicators[:3])
        parts.append(f"Severity indicators noted: {si_str}.")

    if age:
        parts.append(f"Patient is {age} years old ({sex}).")

    parts.append(f"Symptoms have been present for {duration}.")

    return " ".join(parts)


# ---- Main NLP pipeline function ----

def process_symptoms(
    symptom_text: str,
    selected_chips: List[str],
    age: int | None,
    sex: str,
    duration: str
) -> dict:
    """
    Full NLP pipeline. Takes raw input, returns structured data.
    This is called by the route handler before the ML/AI step.
    """
    # Combine chips and free text
    combined = ". ".join(selected_chips + ([symptom_text] if symptom_text else []))

    # Run NLP steps
    keywords = extract_keywords(combined)
    severity = detect_severity_indicators(combined)
    duration_str = normalize_duration(duration)
    highlighted = generate_highlighted_html(symptom_text, keywords)
    summary = generate_symptom_summary(keywords, severity, duration_str, age, sex)

    return {
        "detected_keywords": keywords,
        "symptom_summary": summary,
        "parsed_highlights": highlighted,
        "severity_indicators": severity,
        "duration_context": duration_str,
        "combined_text": combined,     # used by ML and AI steps
        "keyword_count": len(keywords),
        "has_severe_indicators": len(severity) > 0
    }
"""
STEP 4: LINGUISTIC NLP PIPELINE
=================================
Why spaCy + NLTK on top of our regex processor?

Our regex processor (processor.py) is PATTERN-BASED:
  - Fast and reliable for known phrases
  - But misses: "the ache in my left side", "been having episodes"
  - Can't understand sentence structure or word relationships

spaCy adds LINGUISTIC understanding:
  1. TOKENIZATION     — splits text into words/punctuation properly
                        "can't" → ["ca", "n't"]  not ["can't"]

  2. LEMMATIZATION    — reduces words to their root form
                        "aching" → "ache"
                        "breathless" → "breathless"
                        "episodes" → "episode"
                        This means we catch "aching" even if dict has "ache"

  3. POS TAGGING      — identifies noun/verb/adjective/adverb
                        "sharp [ADJ] pain [NOUN]" → symptom descriptor
                        "suddenly [ADV] started [VERB]" → onset pattern

  4. DEPENDENCY PARSE — understands relationships between words
                        "pain in my LEFT ARM" — "arm" is noun, "left" modifies it
                        "NO chest pain"       — "no" negates "chest pain"

  5. SPAN DETECTION   — finds multi-word medical entities
                        "shortness of breath" = one clinical concept

  6. MEDICAL NER      — our custom Named Entity Recognition
                        Labels spans as: SYMPTOM, BODY_PART, SEVERITY,
                        CONTEXT, NEGATION, DURATION

Together these make the pipeline much more robust to:
  - Spelling variations ("breathlessness", "breathless", "short of breath")
  - Grammatical variations ("I ache" vs "aching" vs "the ache")
  - Complex negation ("no real pain to speak of")
  - Descriptors that modify symptoms ("dull ache" vs "sharp pain")
"""

import re
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token, Span
from typing import List, Dict, Tuple, Optional

# =============================================
# MEDICAL LEXICON
# All the vocabulary the linguistic pipeline knows
# =============================================

# Symptoms with their lemma forms (base form the lemmatizer produces)
SYMPTOM_LEMMAS: Dict[str, str] = {
    # Cardiovascular
    "pain": "chest pain",          # context-dependent (check nearby tokens)
    "ache": "pain",
    "tightness": "chest tightness",
    "pressure": "chest pressure",
    "palpitation": "palpitations",
    "pounding": "palpitations",
    "racing": "rapid heartbeat",
    "flutter": "palpitations",
    "breathless": "shortness of breath",
    "breathlessness": "shortness of breath",
    "wheeze": "wheezing",
    "wheezing": "wheezing",
    # Neurological
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "lightheaded": "dizziness",
    "faint": "fainting",
    "blackout": "fainting",
    "migraine": "migraine",
    "numb": "numbness",
    "numbness": "numbness",
    "tingle": "tingling",
    "tingling": "tingling",
    "confusion": "confusion",
    "confused": "confusion",
    "seizure": "seizure",
    # General
    "fever": "fever",
    "fatigue": "fatigue",
    "tired": "fatigue",
    "exhaustion": "fatigue",
    "exhausted": "fatigue",
    "nausea": "nausea",
    "nauseous": "nausea",
    "vomit": "vomiting",
    "vomiting": "vomiting",
    "cough": "cough",
    "coughing": "cough",
    "rash": "rash",
    "swelling": "swelling",
    "swollen": "swelling",
}

# Body parts that signal symptom location
BODY_PARTS = {
    "chest", "heart", "arm", "leg", "head", "stomach", "abdomen",
    "back", "neck", "shoulder", "jaw", "throat", "eye", "ear",
    "knee", "hip", "ankle", "wrist", "elbow", "side", "left", "right"
}

# Severity adjectives — these modify symptoms
SEVERITY_ADJECTIVES = {
    "mild": 1, "slight": 1, "minor": 1, "little": 1,
    "moderate": 2, "medium": 2, "fair": 2,
    "severe": 3, "extreme": 3, "intense": 3, "terrible": 3,
    "sharp": 2, "dull": 1, "crushing": 3, "stabbing": 3,
    "throbbing": 2, "burning": 2, "aching": 1, "squeezing": 3,
    "radiating": 2, "shooting": 2,
}

# Onset/temporal adverbs
ONSET_WORDS = {
    "suddenly", "suddenly", "abruptly", "rapidly", "gradually",
    "slowly", "recently", "constantly", "intermittently", "occasionally"
}

# Negation tokens (spaCy dependency label "neg" also catches these)
NEGATION_TOKENS = {
    "no", "not", "never", "without", "absent", "deny", "denies",
    "negative", "rule", "out", "free"
}


# =============================================
# CUSTOM spaCy PIPELINE COMPONENTS
# We register these as @Language.component
# so spaCy runs them in sequence
# =============================================

@Language.component("medical_entity_ruler")
def medical_entity_ruler(doc: Doc) -> Doc:
    """
    Custom NER component: labels medical spans in the doc.
    
    Labels we assign:
      SYMPTOM   — a clinical symptom
      BODY_PART — anatomical location
      SEVERITY  — severity descriptor
      CONTEXT   — temporal/activity context
      NEGATION  — negation marker
    
    spaCy runs this after tokenization.
    We read token text/lemma and set doc.ents.
    """
    new_ents = []
    i = 0
    tokens = [t for t in doc]

    while i < len(tokens):
        tok = tokens[i]
        text_lower = tok.text.lower()
        lemma_lower = tok.lemma_.lower() if tok.lemma_ != "-PRON-" else text_lower

        # Check for NEGATION
        if text_lower in NEGATION_TOKENS:
            span = doc[tok.i : tok.i + 1]
            span.label_ = "NEGATION"  # won't work directly on Span — use ents list
            new_ents.append((tok.i, tok.i + 1, "NEGATION"))

        # Check for BODY PART
        elif text_lower in BODY_PARTS or lemma_lower in BODY_PARTS:
            new_ents.append((tok.i, tok.i + 1, "BODY_PART"))

        # Check for SEVERITY adjective
        elif text_lower in SEVERITY_ADJECTIVES or lemma_lower in SEVERITY_ADJECTIVES:
            new_ents.append((tok.i, tok.i + 1, "SEVERITY"))

        # Check for SYMPTOM via lemma
        elif lemma_lower in SYMPTOM_LEMMAS or text_lower in SYMPTOM_LEMMAS:
            new_ents.append((tok.i, tok.i + 1, "SYMPTOM"))

        i += 1

    # Set entities (spaCy requires non-overlapping spans)
    filtered = _filter_overlapping_spans(doc, new_ents)
    doc.ents = filtered
    return doc


def _filter_overlapping_spans(doc: Doc, ents: List[Tuple]) -> List[Span]:
    """Remove overlapping entity spans, keeping the longer one."""
    spans = []
    for s, e, label in ents:
        if s < e <= len(doc):
            span = doc[s:e]
            # Correct spaCy API: create span with label via doc.char_span or Span()
            labeled = Span(doc, s, e, label=label)
            spans.append(labeled)
    return spacy.util.filter_spans(spans)


# =============================================
# PIPELINE BUILDER
# =============================================

_nlp_pipeline = None  # module-level cache


def get_pipeline() -> Language:
    """
    Build and return the spaCy pipeline (lazy-loaded).
    Uses spacy.blank('en') — no model download needed.
    We add:
      - sentencizer   (splits into sentences)
      - our custom medical_entity_ruler
    """
    global _nlp_pipeline
    if _nlp_pipeline is not None:
        return _nlp_pipeline

    nlp = spacy.blank("en")

    # Sentencizer: splits text into sentences by punctuation
    # Needed so we can analyse each sentence for negation scope
    nlp.add_pipe("sentencizer")

    # Our custom medical entity labeler
    nlp.add_pipe("medical_entity_ruler")

    _nlp_pipeline = nlp
    return nlp


# =============================================
# LINGUISTIC ANALYSIS FUNCTIONS
# =============================================

def tokenize_and_lemmatize(text: str) -> List[Dict]:
    """
    Run spaCy tokenization on text.
    Returns list of token dicts with: text, lemma, pos, is_negated, entity_label

    This is the core of what spaCy gives us over simple regex:
    - Lemmatization: "aching" → "ache", "breathless" → "breathless"
    - Proper tokenization: handles punctuation, contractions
    """
    nlp = get_pipeline()
    doc = nlp(text.lower())

    tokens = []
    for tok in doc:
        tokens.append({
            "text":    tok.text,
            "lemma":   tok.lemma_,
            "pos":     tok.pos_,      # NOUN, VERB, ADJ, ADV, ...
            "is_stop": tok.is_stop,   # common words like "the", "a", "is"
            "is_punct": tok.is_punct,
            "ent_type": tok.ent_type_ if tok.ent_type_ else None,
        })
    return tokens


def extract_symptom_descriptors(text: str) -> List[Dict]:
    """
    Find (adjective, symptom) pairs — e.g. ("sharp", "chest pain").
    
    This is linguistic analysis: POS tagging lets us find
    adjectives that precede nouns, which tells us severity
    without the user using medical words.
    
    "sharp pain" → {symptom: "chest pain", severity_adj: "sharp", score: 2}
    "dull ache"  → {symptom: "pain",       severity_adj: "dull",  score: 1}
    """
    nlp = get_pipeline()
    doc = nlp(text.lower())
    descriptors = []

    tokens = list(doc)
    for i, tok in enumerate(tokens):
        lemma = tok.lemma_.lower()

        # If this token is a symptom word
        if lemma in SYMPTOM_LEMMAS or tok.text in SYMPTOM_LEMMAS:
            symptom = SYMPTOM_LEMMAS.get(lemma) or SYMPTOM_LEMMAS.get(tok.text)

            # Look back up to 3 tokens for a severity adjective
            sev_adj = None
            sev_score = 1
            for j in range(max(0, i - 3), i):
                prev = tokens[j]
                prev_lemma = prev.lemma_.lower()
                if prev_lemma in SEVERITY_ADJECTIVES:
                    sev_adj = prev.text
                    sev_score = SEVERITY_ADJECTIVES[prev_lemma]
                    break
                elif prev.text.lower() in SEVERITY_ADJECTIVES:
                    sev_adj = prev.text
                    sev_score = SEVERITY_ADJECTIVES[prev.text.lower()]
                    break

            # Check if negated (look back for negation word)
            is_negated = False
            for j in range(max(0, i - 4), i):
                if tokens[j].text.lower() in NEGATION_TOKENS:
                    is_negated = True
                    break

            descriptors.append({
                "symptom":      symptom,
                "severity_adj": sev_adj,
                "severity_score": sev_score,
                "is_negated":   is_negated,
                "source_text":  tok.text,
            })

    return descriptors


def extract_body_symptom_pairs(text: str) -> List[str]:
    """
    Find BODY_PART + SYMPTOM combinations in the doc.
    
    "pain in my left arm" → ["left arm pain"]
    "tightness in chest"  → ["chest tightness"]
    "ache behind eyes"    → ["headache"] (eyes → head region)
    
    This catches anatomical descriptions that exact keyword matching misses.
    """
    nlp = get_pipeline()
    doc = nlp(text.lower())

    body_symptom_map = {
        ("chest", "pain"):        "chest pain",
        ("chest", "tightness"):   "chest tightness",
        ("chest", "pressure"):    "chest pressure",
        ("chest", "ache"):        "chest pain",
        ("chest", "aching"):      "chest pain",
        ("chest", "hurt"):        "chest pain",
        ("chest", "hurting"):     "chest pain",
        ("chest", "sore"):        "chest pain",
        ("chest", "discomfort"):  "chest pain",
        ("heart", "pounding"):    "palpitations",
        ("heart", "racing"):      "rapid heartbeat",
        ("heart", "flutter"):     "palpitations",
        ("heart", "beating"):     "palpitations",
        ("arm", "pain"):          "left arm pain",
        ("arm", "numb"):          "left arm pain",
        ("arm", "ache"):          "left arm pain",
        ("arm", "aching"):        "left arm pain",
        ("arm", "tingle"):        "left arm pain",
        ("arm", "tingling"):      "left arm pain",
        ("arm", "heavy"):         "left arm pain",
        ("jaw", "pain"):          "jaw pain",
        ("jaw", "ache"):          "jaw pain",
        ("jaw", "tight"):         "jaw pain",
        ("head", "pain"):         "headache",
        ("head", "ache"):         "headache",
        ("head", "aching"):       "headache",
        ("head", "throbbing"):    "headache",
        ("stomach", "pain"):      "abdominal pain",
        ("stomach", "ache"):      "stomach ache",
        ("stomach", "aching"):    "stomach ache",
        ("abdomen", "pain"):      "abdominal pain",
        ("back", "pain"):         "back pain",
        ("back", "ache"):         "back pain",
        ("back", "aching"):       "back pain",
        ("throat", "pain"):       "sore throat",
        ("throat", "sore"):       "sore throat",
        ("knee", "pain"):         "joint pain",
        ("joint", "pain"):        "joint pain",
        ("shoulder", "pain"):     "shoulder pain",
        ("neck", "pain"):         "neck pain",
        ("side", "pain"):         "abdominal pain",
        ("side", "ache"):         "abdominal pain",
    }

    tokens = list(doc)
    found = []
    used = set()

    for i, tok in enumerate(tokens):
        tok_lemma = tok.lemma_.lower()
        tok_text  = tok.text.lower()

        # If this is a body part token
        if tok_text in BODY_PARTS or tok_lemma in BODY_PARTS:
            body = tok_lemma if tok_lemma in BODY_PARTS else tok_text

            # Look within ±4 tokens for a symptom word
            window = tokens[max(0, i-4): min(len(tokens), i+5)]
            for nearby in window:
                if nearby.i == tok.i:
                    continue
                nearby_lemma = nearby.lemma_.lower()
                key = (body, nearby_lemma)
                if key in body_symptom_map and key not in used:
                    # Check not negated
                    neg = any(
                        tokens[j].text.lower() in NEGATION_TOKENS
                        for j in range(max(0, i-3), i)
                    )
                    if not neg:
                        found.append(body_symptom_map[key])
                        used.add(key)

    return found


def extract_onset_patterns(text: str) -> Dict:
    """
    Use POS tagging to find temporal/onset adverbs.
    
    "suddenly started" → sudden_onset: True
    "gradually getting worse" → progressive: True
    "woke me up" → nocturnal: True
    
    Returns dict of onset flags for context enrichment.
    """
    nlp = get_pipeline()
    doc = nlp(text.lower())

    onset = {
        "sudden": False,
        "progressive": False,
        "nocturnal": False,
        "episodic": False,
        "chronic": False,
    }

    text_lower = text.lower()

    # Sudden onset patterns
    if re.search(r'\b(suddenly|all of a sudden|out of nowhere|abruptly|'
                 r'came on suddenly|without warning|bolt|struck)\b', text_lower):
        onset["sudden"] = True

    # Progressive
    if re.search(r'\b(getting worse|worsening|progressing|spreading|'
                 r'increasingly|more and more|building up|escalating)\b', text_lower):
        onset["progressive"] = True

    # Nocturnal (night-time)
    if re.search(r'\b(night|woke|woken|waking|sleep|overnight|'
                 r'in bed|lying down|2am|3am|midnight)\b', text_lower):
        onset["nocturnal"] = True

    # Episodic (comes and goes)
    if re.search(r'\b(comes? and goes?|on and off|intermittent|'
                 r'episodes?|attacks?|spells?|sometimes|occasionally)\b', text_lower):
        onset["episodic"] = True

    # Chronic
    if re.search(r'\b(weeks?|months?|years?|long.?term|chronic|'
                 r'for a while|ongoing|persistent|recurring)\b', text_lower):
        onset["chronic"] = True

    return onset


def extract_quantity_expressions(text: str) -> Dict:
    """
    Extract numeric/quantitative information from text.
    
    "37.8 temperature" → fever: True
    "heart rate 120"   → tachycardia flag
    "10/10 pain"       → severe pain
    "pain scale 8"     → severe pain
    """
    findings = {}

    # Temperature (fever)
    temp_match = re.search(r'(\d+\.?\d*)\s*(degrees?|°|celsius|fahrenheit|c\b|f\b)', text.lower())
    if temp_match:
        val = float(temp_match.group(1))
        if val > 37.5 or (val > 99.5 and val < 110):  # Celsius or Fahrenheit
            findings["fever_detected"] = True
            findings["temperature"] = val

    # Pain scale
    pain_scale = re.search(r'(pain|ache|discomfort).{0,20}(\d+)\s*(/\s*10|out of 10)', text.lower())
    if pain_scale:
        score = int(pain_scale.group(2))
        findings["pain_scale"] = score
        if score >= 7:
            findings["severe_pain"] = True

    # Heart rate
    hr_match = re.search(r'(heart rate|pulse|hr|bpm).{0,10}(\d+)', text.lower())
    if hr_match:
        hr = int(hr_match.group(2))
        findings["heart_rate"] = hr
        if hr > 100:
            findings["tachycardia"] = True
        if hr > 150:
            findings["severe_tachycardia"] = True

    # Duration in text ("for 3 days", "past 2 weeks")
    dur_match = re.search(
        r'(for|past|last|over)\s+(\d+)\s+(hour|day|week|month)',
        text.lower()
    )
    if dur_match:
        num = int(dur_match.group(2))
        unit = dur_match.group(3)
        findings["explicit_duration"] = f"{num} {unit}(s)"

    return findings


# =============================================
# MAIN LINGUISTIC PIPELINE FUNCTION
# Called by the main processor to enrich NLP output
# =============================================

def run_linguistic_analysis(text: str, combined_text: str) -> Dict:
    """
    Runs all linguistic analysis on the symptom text.
    
    Returns a dict that enriches the base NLP output from processor.py:
      - additional_keywords  : new symptoms found via lemmatization/POS
      - severity_from_adj    : severity score derived from adjectives
      - body_symptom_pairs   : anatomical descriptions → symptoms
      - onset_patterns       : sudden/progressive/nocturnal/episodic/chronic
      - quantities           : temperature, pain scale, heart rate
      - linguistic_summary   : human-readable explanation of what was found
    """
    if not text or not text.strip():
        return {
            "additional_keywords": [],
            "severity_from_adj": 0,
            "body_symptom_pairs": [],
            "onset_patterns": {},
            "quantities": {},
            "linguistic_summary": "",
        }

    # Run all linguistic functions
    descriptors   = extract_symptom_descriptors(combined_text)
    body_pairs    = extract_body_symptom_pairs(combined_text)
    onset         = extract_onset_patterns(combined_text)
    quantities    = extract_quantity_expressions(combined_text)

    # Extract additional keywords from linguistic analysis
    additional_keywords = []

    # From symptom descriptors (non-negated ones)
    for d in descriptors:
        if not d["is_negated"] and d["symptom"] not in additional_keywords:
            additional_keywords.append(d["symptom"])

    # From body-part + symptom pairs
    for bp in body_pairs:
        if bp not in additional_keywords:
            additional_keywords.append(bp)

    # Add inferences from quantities
    if quantities.get("fever_detected"):
        if "fever" not in additional_keywords:
            additional_keywords.append("fever")
    if quantities.get("tachycardia"):
        if "rapid heartbeat" not in additional_keywords:
            additional_keywords.append("rapid heartbeat")
    if quantities.get("severe_tachycardia"):
        if "palpitations" not in additional_keywords:
            additional_keywords.append("palpitations")

    # Compute severity from adjectives
    # Take the maximum severity score found across all descriptor pairs
    sev_scores = [d["severity_score"] for d in descriptors if not d["is_negated"]]
    if quantities.get("severe_pain"):
        sev_scores.append(3)
    if quantities.get("pain_scale"):
        scale = quantities["pain_scale"]
        sev_scores.append(1 if scale < 4 else 2 if scale < 7 else 3)
    severity_from_adj = max(sev_scores) if sev_scores else 0

    # Build human-readable linguistic summary
    summary_parts = []
    if descriptors:
        non_neg = [d for d in descriptors if not d["is_negated"]]
        if non_neg:
            desc_strs = [
                f'{d["severity_adj"] + " " if d["severity_adj"] else ""}{d["source_text"]}'
                for d in non_neg[:3]
            ]
            summary_parts.append(f"Linguistic analysis found: {', '.join(desc_strs)}")

    if body_pairs:
        summary_parts.append(f"Anatomical symptoms: {', '.join(body_pairs[:3])}")

    active_onset = [k for k, v in onset.items() if v]
    if active_onset:
        summary_parts.append(f"Onset pattern: {', '.join(active_onset)}")

    if quantities:
        q_parts = []
        if "temperature" in quantities:
            q_parts.append(f"temp {quantities['temperature']}°")
        if "pain_scale" in quantities:
            q_parts.append(f"pain {quantities['pain_scale']}/10")
        if "heart_rate" in quantities:
            q_parts.append(f"HR {quantities['heart_rate']}bpm")
        if q_parts:
            summary_parts.append(f"Measurements: {', '.join(q_parts)}")

    return {
        "additional_keywords": additional_keywords,
        "severity_from_adj":   severity_from_adj,
        "body_symptom_pairs":  body_pairs,
        "onset_patterns":      onset,
        "quantities":          quantities,
        "linguistic_summary":  ". ".join(summary_parts),
        "descriptor_count":    len([d for d in descriptors if not d["is_negated"]]),
    }
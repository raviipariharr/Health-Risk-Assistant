"""
AI INTEGRATION MODULE
======================
This module calls the Anthropic API (Claude) with the
pre-processed NLP output and ML score as context.

Key design decisions:
  1. We do NOT send raw user text to the LLM
     — Instead we send structured NLP output
     — This reduces prompt injection risks and hallucination

  2. We give the LLM the ML score as a "prior"
     — The LLM can agree, disagree, or refine it
     — This is "LLM-as-a-reasoner over structured features"

  3. We ask for structured JSON output
     — Easier to render in the frontend
     — Forces the model to reason step by step (CoT in JSON)

  4. The LLM adds:
     — Plain-language explanation of the risk
     — Reasoning steps (explainability)
     — Personalized suggestions
"""

import json
import httpx
from typing import List


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def build_prompt(
    combined_text: str,
    keywords: List[str],
    severity_indicators: List[str],
    ml_score: float,
    ml_contributions: List[str],
    primary_concern: str,
    duration_context: str,
    age: int | None,
    sex: str
) -> str:
    """
    Build a carefully structured prompt.

    Notice how we're NOT just sending "analyze this patient":
    - We give the model the NLP-extracted keywords (grounded facts)
    - We give it the ML score (quantitative prior)
    - We give it the feature contributions (why the ML thinks what it does)
    - We ask it to reason step by step and explain itself
    """

    keywords_str = ", ".join(keywords) if keywords else "none clearly identified"
    severity_str = ", ".join(severity_indicators) if severity_indicators else "none"
    contributions_str = "\n".join(f"  - {c}" for c in ml_contributions)

    return f"""You are a clinical AI assistant analyzing pre-processed patient symptom data.

=== NLP-EXTRACTED DATA ===
Original symptom description: "{combined_text}"
Detected symptom keywords: {keywords_str}
Severity indicators found: {severity_str}
Duration context: {duration_context}
Patient: {age or 'unknown'} years old, {sex}

=== ML MODEL OUTPUT ===
Risk score (0-100): {ml_score:.0f}
Primary concern identified: {primary_concern}
Score contributions:
{contributions_str}

=== YOUR TASK ===
Using the NLP data and ML score above as your foundation, produce a clinical AI assessment.
You may adjust the risk level up or down from the ML score if the clinical picture warrants it.

Respond ONLY with valid JSON (no markdown, no backticks, no preamble):

{{
  "risk": {{
    "level": "low" | "medium" | "high",
    "score": <integer 0-100>,
    "primary_concern": "<plain-language primary concern>",
    "explanation": "<2-3 sentences: what these symptoms suggest and why this risk level>",
    "confidence": "low" | "medium" | "high",
    "reasoning_steps": [
      "<step 1: what key symptoms were present>",
      "<step 2: how severity/duration factors affected assessment>",
      "<step 3: why this risk level vs the ML score>"
    ],
    "ml_score": {ml_score},
    "ai_score": <your independent score as integer>
  }},
  "suggestions": [
    {{
      "priority": "urgent" | "important" | "general",
      "icon": "🏥" | "💊" | "🛌" | "🥗" | "🚶" | "💧" | "🩺",
      "title": "<short action title>",
      "detail": "<1-2 sentence specific guidance>"
    }}
  ]
}}

Rules:
- Give 3-5 suggestions ordered by priority
- urgent = seek medical care now or today
- important = act within a few days
- general = lifestyle/monitoring advice
- Be clinically accurate but use plain language
- Do NOT invent symptoms not mentioned by the patient"""


async def get_ai_assessment(
    combined_text: str,
    keywords: List[str],
    severity_indicators: List[str],
    ml_score: float,
    ml_contributions: List[str],
    primary_concern: str,
    duration_context: str,
    age: int | None,
    sex: str,
    api_key: str
) -> dict:
    """
    Async function to call the Anthropic API.

    Uses httpx (async HTTP client) instead of requests (sync)
    so FastAPI can handle other requests while waiting for the LLM.

    Returns parsed JSON dict with 'risk' and 'suggestions' keys.
    """
    prompt = build_prompt(
        combined_text, keywords, severity_indicators,
        ml_score, ml_contributions, primary_concern,
        duration_context, age, sex
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        response.raise_for_status()

    data = response.json()
    raw_text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )

    # Strip markdown fences if the model included them
    clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(clean)
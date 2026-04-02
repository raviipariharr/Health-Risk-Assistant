"""
DATA MODELS — Pydantic Schemas
================================
Pydantic is FastAPI's secret weapon for data validation.

When a request comes in, FastAPI automatically:
  1. Parses the JSON body
  2. Validates each field (type, range, required/optional)
  3. Returns a clear 422 error if something is wrong
  4. Converts to a Python object you can use directly

Think of these as "contracts" between the frontend and backend.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ---- Enums: fixed sets of allowed values ----

class BiologicalSex(str, Enum):
    male = "male"
    female = "female"
    other = "other"

class SymptomDuration(str, Enum):
    hours = "hours"
    days = "days"
    week = "week"
    weeks = "weeks"
    months = "months"

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class SuggestionPriority(str, Enum):
    urgent = "urgent"
    important = "important"
    general = "general"


# ---- REQUEST model: what the frontend sends ----

class SymptomRequest(BaseModel):
    """
    This is what the frontend POSTs to /api/v1/analyze

    Field(...) means required.
    Field(default, ...) means optional with a default value.
    """
    symptom_text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Free-text symptom description"
    )
    selected_chips: List[str] = Field(
        default=[],
        description="Pre-selected symptom chips from UI"
    )
    age: Optional[int] = Field(
        default=None,
        ge=1,       # ge = greater than or equal to
        le=120,     # le = less than or equal to
        description="Patient age in years"
    )
    sex: BiologicalSex = Field(
        default=BiologicalSex.other,
        description="Biological sex for risk context"
    )
    duration: SymptomDuration = Field(
        default=SymptomDuration.days,
        description="How long symptoms have been present"
    )

    # Custom validator: clean up the text input
    @field_validator("symptom_text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()

    # Computed property: combine chips + text into one string
    @property
    def combined_symptoms(self) -> str:
        parts = self.selected_chips + ([self.symptom_text] if self.symptom_text else [])
        return ". ".join(parts)


# ---- RESPONSE sub-models ----

class NLPResult(BaseModel):
    """Output of the NLP processing step"""
    detected_keywords: List[str]
    symptom_summary: str
    parsed_highlights: str       # HTML with <span> highlights
    severity_indicators: List[str]   # words like "severe", "sudden", "persistent"
    duration_context: str        # normalized duration string

class RiskPrediction(BaseModel):
    """Output of the ML + AI risk scoring step"""
    level: RiskLevel
    score: int = Field(ge=0, le=100)
    primary_concern: str
    explanation: str
    confidence: str
    reasoning_steps: List[str]
    ml_score: Optional[float] = None     # score from local ML model
    ai_score: Optional[float] = None     # score from LLM

class Suggestion(BaseModel):
    """A single recommendation"""
    priority: SuggestionPriority
    icon: str
    title: str
    detail: str

class AnalysisResponse(BaseModel):
    """The full response returned to the frontend"""
    success: bool = True
    session_id: str                      # unique ID for this analysis
    nlp: NLPResult
    risk: RiskPrediction
    suggestions: List[Suggestion]
    processing_time_ms: Optional[int] = None
    disclaimer: str = (
        "This analysis is for informational purposes only and does not "
        "constitute medical advice. Always consult a qualified healthcare professional."
    )

class ErrorResponse(BaseModel):
    """Returned when something goes wrong"""
    success: bool = False
    error: str
    detail: Optional[str] = None
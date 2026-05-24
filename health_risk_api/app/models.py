"""
DATA MODELS — Pydantic Schemas
================================
Extended with vital signs and lab values so the ML pipeline
can use real clinical measurements, not just symptom keywords.

New fields added to SymptomRequest:
  Vitals  — blood pressure (systolic/diastolic), heart rate, SpO2,
             respiratory rate, body temperature
  Labs    — total cholesterol, LDL, HDL, triglycerides, blood glucose
             (fasting), HbA1c, BMI

Each field is Optional — the frontend collects only what is
relevant to the user's selected focus area (cardiac, metabolic, etc.)
The ML feature extractor handles None values gracefully.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class BiologicalSex(str, Enum):
    male   = "male"
    female = "female"
    other  = "other"

class SymptomDuration(str, Enum):
    hours  = "hours"
    days   = "days"
    week   = "week"
    weeks  = "weeks"
    months = "months"

class RiskLevel(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"

class SuggestionPriority(str, Enum):
    urgent    = "urgent"
    important = "important"
    general   = "general"


# ---- REQUEST model ----

class VitalsInput(BaseModel):
    """Real-time vital signs entered by the user."""
    systolic_bp:        Optional[int]   = Field(None, ge=50,  le=300,  description="Systolic blood pressure (mmHg)")
    diastolic_bp:       Optional[int]   = Field(None, ge=30,  le=200,  description="Diastolic blood pressure (mmHg)")
    heart_rate:         Optional[int]   = Field(None, ge=20,  le=300,  description="Heart rate (bpm)")
    spo2:               Optional[float] = Field(None, ge=50.0,le=100.0,description="Oxygen saturation (%)")
    respiratory_rate:   Optional[int]   = Field(None, ge=5,   le=60,   description="Breaths per minute")
    temperature_celsius:Optional[float] = Field(None, ge=30.0,le=45.0, description="Body temperature (°C)")
    weight_kg:          Optional[float] = Field(None, ge=1.0, le=400.0,description="Body weight (kg)")
    height_cm:          Optional[float] = Field(None, ge=50.0,le=250.0,description="Height (cm)")


class LabsInput(BaseModel):
    """Laboratory / blood test values entered by the user."""
    total_cholesterol:  Optional[float] = Field(None, ge=0.0, le=20.0,  description="Total cholesterol (mmol/L)")
    ldl_cholesterol:    Optional[float] = Field(None, ge=0.0, le=15.0,  description="LDL cholesterol (mmol/L)")
    hdl_cholesterol:    Optional[float] = Field(None, ge=0.0, le=10.0,  description="HDL cholesterol (mmol/L)")
    triglycerides:      Optional[float] = Field(None, ge=0.0, le=20.0,  description="Triglycerides (mmol/L)")
    fasting_glucose:    Optional[float] = Field(None, ge=0.0, le=50.0,  description="Fasting blood glucose (mmol/L)")
    hba1c:              Optional[float] = Field(None, ge=0.0, le=20.0,  description="HbA1c (%)")
    creatinine:         Optional[float] = Field(None, ge=0.0, le=200.0, description="Creatinine (µmol/L)")
    egfr:               Optional[float] = Field(None, ge=0.0, le=200.0, description="eGFR (mL/min/1.73m²)")


class SymptomRequest(BaseModel):
    symptom_text:   str = Field(..., min_length=1, max_length=1000)
    selected_chips: List[str] = Field(default=[])
    age:            Optional[int] = Field(default=None, ge=1, le=120)
    sex:            BiologicalSex = Field(default=BiologicalSex.other)
    duration:       SymptomDuration = Field(default=SymptomDuration.days)
    vitals:         Optional[VitalsInput] = Field(default=None, description="Optional vital signs")
    labs:           Optional[LabsInput]   = Field(default=None, description="Optional lab results")
    focus_area:     Optional[str] = Field(default=None, description="cardiac | metabolic | respiratory | general")

    @field_validator("symptom_text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()

    @property
    def combined_symptoms(self) -> str:
        parts = self.selected_chips + ([self.symptom_text] if self.symptom_text else [])
        return ". ".join(parts)


# ---- RESPONSE sub-models ----

class VitalsInterpretation(BaseModel):
    """Flagged clinical findings from vitals/labs."""
    flags:   List[str] = []   # e.g. ["Hypertension Stage 2", "Hyperglycaemia"]
    summary: str       = ""


class NLPResult(BaseModel):
    detected_keywords:   List[str]
    symptom_summary:     str
    parsed_highlights:   str
    severity_indicators: List[str]
    duration_context:    str

class RiskPrediction(BaseModel):
    level:           RiskLevel
    score:           int = Field(ge=0, le=100)
    primary_concern: str
    explanation:     str
    confidence:      str
    reasoning_steps: List[str]
    ml_score:        Optional[float] = None
    ai_score:        Optional[float] = None
    vitals_flags:    List[str] = []

class Suggestion(BaseModel):
    priority: SuggestionPriority
    icon:     str
    title:    str
    detail:   str

class AnalysisResponse(BaseModel):
    success:            bool = True
    session_id:         str
    nlp:                NLPResult
    risk:               RiskPrediction
    suggestions:        List[Suggestion]
    vitals_summary:     Optional[VitalsInterpretation] = None
    processing_time_ms: Optional[int] = None
    disclaimer: str = (
        "This analysis is for informational purposes only and does not "
        "constitute medical advice. Always consult a qualified healthcare professional."
    )

class ErrorResponse(BaseModel):
    success: bool = False
    error:   str
    detail:  Optional[str] = None
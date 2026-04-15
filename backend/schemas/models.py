"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class PatientInput(BaseModel):
    """Input schema for single patient prediction."""
    age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    bp: Optional[float] = Field(None, ge=0, le=200, description="Blood pressure mm/Hg")
    sg: Optional[str] = Field(None, description="Specific gravity (1.005-1.025)")
    al: Optional[str] = Field(None, description="Albumin (0-5)")
    su: Optional[str] = Field(None, description="Sugar (0-5)")
    rbc: Optional[str] = Field(None, description="Red blood cells (normal/abnormal)")
    pc: Optional[str] = Field(None, description="Pus cell (normal/abnormal)")
    pcc: Optional[str] = Field(None, description="Pus cell clumps (present/notpresent)")
    ba: Optional[str] = Field(None, description="Bacteria (present/notpresent)")
    bgr: Optional[float] = Field(None, ge=0, le=600, description="Blood glucose random mg/dl")
    bu: Optional[float] = Field(None, ge=0, le=500, description="Blood urea mg/dl")
    sc: Optional[float] = Field(None, ge=0, le=100, description="Serum creatinine mg/dl")
    sod: Optional[float] = Field(None, ge=0, le=200, description="Sodium mEq/L")
    pot: Optional[float] = Field(None, ge=0, le=100, description="Potassium mEq/L")
    hemo: Optional[float] = Field(None, ge=0, le=20, description="Hemoglobin gms")
    pcv: Optional[float] = Field(None, ge=0, le=100, description="Packed cell volume")
    wbcc: Optional[float] = Field(None, ge=0, le=30000, description="White blood cell count")
    rbcc: Optional[float] = Field(None, ge=0, le=10, description="Red blood cell count")
    htn: Optional[str] = Field(None, description="Hypertension (yes/no)")
    dm: Optional[str] = Field(None, description="Diabetes mellitus (yes/no)")
    cad: Optional[str] = Field(None, description="Coronary artery disease (yes/no)")
    appet: Optional[str] = Field(None, description="Appetite (good/poor)")
    pe: Optional[str] = Field(None, description="Pedal edema (yes/no)")
    ane: Optional[str] = Field(None, description="Anemia (yes/no)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 48, "bp": 80, "sg": "1.020", "al": "1", "su": "0",
            "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
            "bgr": 121, "bu": 36, "sc": 1.2, "sod": 135, "pot": 4.0,
            "hemo": 15.4, "pcv": 44, "wbcc": 7800, "rbcc": 5.2,
            "htn": "yes", "dm": "yes", "cad": "no", "appet": "good",
            "pe": "no", "ane": "no",
        }]
    }}


class SingleModelPrediction(BaseModel):
    """Prediction from a single model."""
    model_name: str = Field(..., description="Name of the model")
    prediction: int = Field(..., description="0=No CKD, 1=CKD")
    probability: float = Field(..., ge=0, le=1, description="Probability of CKD")
    risk_level: str = Field(..., description="Low / Medium / High")
    confidence: str = Field(..., description="Confidence level")


class PredictionResponse(BaseModel):
    """Response containing predictions from ALL trained models."""
    best_model_name: str = Field(..., description="Name of the best model")
    final_prediction: int = Field(..., description="Final verdict from best model: 0=No CKD, 1=CKD")
    final_probability: float = Field(..., ge=0, le=1, description="Probability from best model")
    final_risk_level: str = Field(..., description="Risk level from best model")
    all_predictions: List[SingleModelPrediction] = Field(..., description="Predictions from all models")


class TrainResponse(BaseModel):
    """Response after training the pipeline."""
    status: str
    best_model: str
    models_trained: List[str]
    evaluation_results: Dict[str, Dict[str, float]]
    plots_generated: List[str]


class MetricsResponse(BaseModel):
    """Response for evaluation metrics."""
    results: Dict[str, Dict[str, float]]
    best_model: str
    feature_selection: Dict[str, List[str]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: Optional[str] = None

"""
FastAPI main application for CKD prediction service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import logging
import joblib
import pandas as pd
from typing import Dict, Any, List
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CKD Ensemble Classifier API",
    description="API for Chronic Kidney Disease prediction using ensemble ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class PatientData(BaseModel):
    """Patient data model for CKD prediction."""

    # Demographics
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    bp: float = Field(..., ge=0, le=200, description="Blood pressure (mm/Hg)")
    sg: float = Field(..., ge=0, le=2, description="Specific gravity")
    al: float = Field(..., ge=0, le=5, description="Albumin")
    su: float = Field(..., ge=0, le=5, description="Sugar")
    rbc: str = Field(..., description="Red blood cells (normal/present)")
    pc: str = Field(..., description="Pus cells (normal/present)")
    pcc: str = Field(..., description="Pus cell clumps (notpresent/present)")
    ba: str = Field(..., description="Bacteria (notpresent/present)")
    bgr: float = Field(..., ge=0, le=500, description="Blood glucose random (mgs/dl)")
    bu: float = Field(..., ge=0, le=500, description="Blood urea (mgs/dl)")
    sc: float = Field(..., ge=0, le=100, description="Serum creatinine (mgs/dl)")
    sod: float = Field(..., ge=0, le=200, description="Sodium (mEq/L)")
    pot: float = Field(..., ge=0, le=100, description="Potassium (mEq/L)")
    hemo: float = Field(..., ge=0, le=20, description="Hemoglobin (gms)")
    pcv: float = Field(..., ge=0, le=100, description="Packed cell volume")
    wc: float = Field(..., ge=0, le=50000, description="White blood cell count")
    rc: float = Field(..., ge=0, le=10, description="Red blood cell count")
    htn: str = Field(..., description="Hypertension (yes/no)")
    dm: str = Field(..., description="Diabetes mellitus (yes/no)")
    cad: str = Field(..., description="Coronary artery disease (yes/no)")
    appet: str = Field(..., description="Appetite (good/poor)")
    pe: str = Field(..., description="Pedal edema (yes/no)")
    ane: str = Field(..., description="Anemia (yes/no)")

    @validator('rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane')
    def validate_categorical_fields(cls, v):
        valid_values = {
            'rbc': ['normal', 'abnormal'],
            'pc': ['normal', 'abnormal'],
            'pcc': ['notpresent', 'present'],
            'ba': ['notpresent', 'present'],
            'htn': ['yes', 'no'],
            'dm': ['yes', 'no'],
            'cad': ['yes', 'no'],
            'appet': ['good', 'poor'],
            'pe': ['yes', 'no'],
            'ane': ['yes', 'no']
        }
        field_name = cls.__name__.lower()
        if field_name in valid_values and v not in valid_values[field_name]:
            raise ValueError(f"Invalid value for {field_name}. Must be one of {valid_values[field_name]}")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction: int = Field(..., description="Prediction (0: No CKD, 1: CKD)")
    probability: float = Field(..., ge=0, le=1, description="Probability of CKD")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: str = Field(..., description="Confidence level based on probability")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to model file")
    preprocessor_path: str = Field(..., description="Path to preprocessor file")


# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None


def load_model_and_preprocessor():
    """Load the model and preprocessor from disk."""
    global model, preprocessor, feature_names

    try:
        model_path = "backend/models/final_ckd_model.pkl"
        preprocessor_path = "backend/models/preprocessor.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names()

        logger.info("Model and preprocessor loaded successfully")
        logger.info(f"Feature names: {feature_names}")

    except Exception as e:
        logger.error(f"Error loading model/preprocessor: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    load_model_and_preprocessor()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path="backend/models/final_ckd_model.pkl",
        preprocessor_path="backend/models/preprocessor.pkl"
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CKD Ensemble Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_patient_data(patient_data: PatientData):
    """
    Predict CKD risk for a patient based on input features.

    Args:
        patient_data: Patient medical data

    Returns:
        Prediction results with probability and risk level
    """
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert patient data to DataFrame
        patient_dict = patient_data.dict()
        df = pd.DataFrame([patient_dict])

        # Ensure columns are in correct order
        df = df[feature_names]

        # Preprocess
        df_processed = preprocessor.transform(df)

        # Make prediction
        prediction = model.predict(df_processed)[0]
        prediction_proba = model.predict_proba(df_processed)[0, 1]

        # Determine risk level
        if prediction_proba < 0.3:
            risk_level = "Low"
        elif prediction_proba < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Determine confidence
        if abs(prediction_proba - 0.5) > 0.3:
            confidence = "High"
        elif abs(prediction_proba - 0.5) > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(prediction_proba),
            risk_level=risk_level,
            confidence=confidence
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(patient_data_list: List[PatientData]):
    """
    Predict CKD risk for multiple patients in batch.

    Args:
        patient_data_list: List of patient medical data

    Returns:
        List of prediction results
    """
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert to DataFrame
        patient_dicts = [patient.dict() for patient in patient_data_list]
        df = pd.DataFrame(patient_dicts)

        # Ensure columns are in correct order
        df = df[feature_names]

        # Preprocess
        df_processed = preprocessor.transform(df)

        # Make predictions
        predictions = model.predict(df_processed)
        prediction_probas = model.predict_proba(df_processed)[:, 1]

        # Format results
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            if proba < 0.3:
                risk_level = "Low"
            elif proba < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            if abs(proba - 0.5) > 0.3:
                confidence = "High"
            elif abs(proba - 0.5) > 0.1:
                confidence = "Medium"
            else:
                confidence = "Low"

            results.append(PredictionResponse(
                prediction=int(pred),
                probability=float(proba),
                risk_level=risk_level,
                confidence=confidence
            ))

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/features")
async def get_feature_names():
    """Get the list of required feature names for prediction."""
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Features not loaded")
    return {"feature_names": feature_names}


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "backend.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
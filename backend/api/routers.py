"""
API routers — /train, /predict, /metrics, /health
"""

import os
import json
import logging
import pandas as pd
import joblib
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..schemas.models import (
    PatientInput,
    PredictionResponse,
    SingleModelPrediction,
    TrainResponse,
    MetricsResponse,
    HealthResponse,
)
from ..ml_core.preprocess import CKDPreprocessor, NOMINAL_COLS, NUMERICAL_COLS
from ..ml_core.pipeline import CKDPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# ──────────────────────────────────────────────────────────────
# Global state (loaded on startup or after /train)
# ──────────────────────────────────────────────────────────────
_state = {
    'model': None,
    'all_models': {},          # name → model object (ALL trained models)
    'preprocessor': None,
    'selected_features': None,
    'model_name': None,
    'summary': None,
}

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


def load_model_artifacts():
    """Load saved models, preprocessor, and feature list from disk."""
    try:
        model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        features_path = os.path.join(MODELS_DIR, 'selected_features.json')
        summary_path = os.path.join(MODELS_DIR, 'pipeline_summary.json')

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            _state['model'] = joblib.load(model_path)
            _state['preprocessor'] = CKDPreprocessor.load(preprocessor_path)
            logger.info(f"Best model loaded from {model_path}")

            # Load ALL individual model files (model_*.joblib)
            _state['all_models'] = {}
            for fname in os.listdir(MODELS_DIR):
                if fname.startswith('model_') and fname.endswith('.joblib'):
                    name = fname.replace('model_', '').replace('.joblib', '')
                    _state['all_models'][name] = joblib.load(
                        os.path.join(MODELS_DIR, fname)
                    )
            logger.info(f"Loaded {len(_state['all_models'])} models: {list(_state['all_models'].keys())}")

            if os.path.exists(features_path):
                with open(features_path) as f:
                    feat_data = json.load(f)
                _state['selected_features'] = feat_data.get('selected_features', [])

            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    _state['summary'] = json.load(f)
                _state['model_name'] = _state['summary'].get('best_model', 'Unknown')
        else:
            logger.warning("No trained model found. Run /train first.")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@router.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "title": "CKD Ensemble Classifier API",
        "version": "2.0.0",
        "paper": "Rahman et al., 2024",
        "endpoints": ["/health", "/train", "/predict", "/metrics", "/plots/"],
    }


@router.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_state['model'] is not None,
        preprocessor_loaded=_state['preprocessor'] is not None,
        model_name=_state.get('model_name'),
    )


@router.post("/train", response_model=TrainResponse, tags=["pipeline"])
async def train_pipeline():
    """
    Trigger the full ML training pipeline.
    This will preprocess data, train models, evaluate, and save artifacts.
    """
    try:
        logger.info("Training pipeline triggered via API...")
        pipeline = CKDPipeline(
            models_dir=MODELS_DIR,
            plots_dir=PLOTS_DIR,
        )
        summary = pipeline.run()

        # Reload artifacts
        load_model_artifacts()

        # List generated plots
        plots = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]

        return TrainResponse(
            status="success",
            best_model=summary['best_model'],
            models_trained=summary['models_trained'],
            evaluation_results=summary['evaluation_results'],
            plots_generated=plots,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(patient: PatientInput):
    """
    Predict CKD risk for a single patient using ALL trained models.
    Returns predictions from every model plus the best model's final verdict.
    """
    if _state['preprocessor'] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run /train first or wait for model to load.",
        )

    models_to_predict = _state.get('all_models', {})
    if not models_to_predict and _state['model'] is not None:
        # Fallback: only the best model is loaded
        models_to_predict = {_state.get('model_name', 'BestModel'): _state['model']}

    if not models_to_predict:
        raise HTTPException(status_code=503, detail="No models loaded. Run /train first.")

    try:
        # Build a DataFrame row from the input
        data = patient.model_dump()
        df = pd.DataFrame([data])

        # Preprocess using the fitted preprocessor
        X_processed = _state['preprocessor'].transform(df)

        # Apply feature selection if available
        if _state['selected_features']:
            available = [f for f in _state['selected_features'] if f in X_processed.columns]
            X_processed = X_processed[available]

        # Predict with ALL models
        all_predictions = []
        best_model_name = _state.get('model_name', 'Unknown')

        for name, model in models_to_predict.items():
            try:
                pred = int(model.predict(X_processed)[0])
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(X_processed)[0, 1])
                else:
                    prob = float(pred)

                # Risk level
                if prob < 0.3:
                    risk = "Low"
                elif prob < 0.7:
                    risk = "Medium"
                else:
                    risk = "High"

                # Confidence
                if abs(prob - 0.5) > 0.3:
                    conf = "High"
                elif abs(prob - 0.5) > 0.15:
                    conf = "Medium"
                else:
                    conf = "Low"

                all_predictions.append(SingleModelPrediction(
                    model_name=name,
                    prediction=pred,
                    probability=round(prob, 4),
                    risk_level=risk,
                    confidence=conf,
                ))
            except Exception as model_err:
                logger.warning(f"Model {name} prediction failed: {model_err}")

        if not all_predictions:
            raise HTTPException(status_code=500, detail="All model predictions failed.")

        # Get best model's prediction for final verdict
        best_pred = next(
            (p for p in all_predictions if p.model_name == best_model_name),
            all_predictions[0],
        )

        return PredictionResponse(
            best_model_name=best_model_name,
            final_prediction=best_pred.prediction,
            final_probability=best_pred.probability,
            final_risk_level=best_pred.risk_level,
            all_predictions=all_predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse, tags=["evaluation"])
async def get_metrics():
    """Return evaluation metrics for all trained models."""
    if _state['summary'] is None:
        # Try loading from disk
        summary_path = os.path.join(MODELS_DIR, 'pipeline_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                _state['summary'] = json.load(f)
        else:
            raise HTTPException(
                status_code=404,
                detail="No metrics found. Run /train first.",
            )

    return MetricsResponse(
        results=_state['summary']['evaluation_results'],
        best_model=_state['summary']['best_model'],
        feature_selection=_state['summary'].get('feature_selection', {}),
    )


@router.get("/plots-list", tags=["evaluation"])
async def list_plots():
    """List all available plot files."""
    if not os.path.exists(PLOTS_DIR):
        return {"plots": []}
    plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
    return {"plots": plots}

"""
API routers — /train, /predict/{variant}, /metrics, /health
Supports 3 variants: all_features, rfe, boruta
"""

import os
import json
import logging
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException

from ..schemas.models import (
    PatientInput,
    PredictionResponse,
    SingleModelPrediction,
    TrainResponse,
    MetricsResponse,
    VariantMetricsResponse,
    HealthResponse,
    VariantEnum,
)
from ..ml_core.preprocess import CKDPreprocessor
from ..ml_core.pipeline import CKDPipeline, VARIANTS

logger = logging.getLogger(__name__)
router = APIRouter()

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────
_state = {
    'preprocessor': None,
    'variants': {},   # variant_name → { 'models': {name→model}, 'features': [...], 'best_model': str }
    'summary': None,
}

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))


def load_model_artifacts():
    """Load all model artifacts from all 3 variant directories."""
    try:
        # Load shared preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            _state['preprocessor'] = CKDPreprocessor.load(preprocessor_path)
            logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning("✗ No preprocessor found.")
            return

        # Load summary
        summary_path = os.path.join(MODELS_DIR, 'pipeline_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                _state['summary'] = json.load(f)

        # Load each variant
        total_models = 0
        for variant in VARIANTS:
            variant_dir = os.path.join(MODELS_DIR, variant)
            features_path = os.path.join(variant_dir, 'features.json')

            if not os.path.exists(variant_dir) or not os.path.exists(features_path):
                logger.warning(f"  ✗ Variant '{variant}' not found — skipping")
                continue

            # Load features
            with open(features_path) as f:
                feat_data = json.load(f)

            # Load all models
            models = {}
            for fname in sorted(os.listdir(variant_dir)):
                if fname.startswith('model_') and fname.endswith('.joblib'):
                    name = fname.replace('model_', '').replace('.joblib', '')
                    models[name] = joblib.load(os.path.join(variant_dir, fname))

            # Get best model name from summary
            best_model = 'Unknown'
            if _state['summary'] and 'variants' in _state['summary']:
                variant_info = _state['summary']['variants'].get(variant, {})
                best_model = variant_info.get('best_model', 'Unknown')

            _state['variants'][variant] = {
                'models': models,
                'features': feat_data.get('features', []),
                'n_features': feat_data.get('n_features', 0),
                'best_model': best_model,
            }

            total_models += len(models)
            logger.info(f"  ✓ {variant}: {len(models)} models loaded "
                         f"({feat_data.get('n_features', '?')} features, best={best_model})")

        logger.info(f"Total: {len(_state['variants'])} variants, {total_models} models loaded")

    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@router.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "title": "CKD Ensemble Classifier API",
        "version": "3.0.0",
        "paper": "Rahman et al., 2024",
        "variants": VARIANTS,
        "endpoints": [
            "/health", "/train",
            "/predict/all_features", "/predict/rfe", "/predict/boruta",
            "/metrics", "/metrics/{variant}",
            "/plots/{variant}/{filename}", "/plots-list",
        ],
    }


@router.get("/health", response_model=HealthResponse, tags=["info"])
async def health_check():
    """Health check with loaded variant info."""
    variants_loaded = list(_state['variants'].keys())
    total = sum(len(v['models']) for v in _state['variants'].values())
    return HealthResponse(
        status="healthy",
        variants_loaded=variants_loaded,
        total_models=total,
        preprocessor_loaded=_state['preprocessor'] is not None,
    )


@router.post("/train", response_model=TrainResponse, tags=["pipeline"])
async def train_pipeline():
    """Trigger the full ML training pipeline (3 variants × 8 models)."""
    try:
        logger.info("Training pipeline triggered via API...")
        pipeline = CKDPipeline(
            models_dir=MODELS_DIR,
            plots_dir=PLOTS_DIR,
        )
        summary = pipeline.run()

        # Reload all artifacts
        load_model_artifacts()

        # Build response
        variant_results = {}
        for v_name, v_data in summary.get('variants', {}).items():
            variant_results[v_name] = {
                'n_features': v_data['n_features'],
                'models_trained': v_data['models_trained'],
                'best_model': v_data['best_model'],
            }

        total = sum(len(v['models_trained']) for v in summary.get('variants', {}).values())

        return TrainResponse(
            status="success",
            variants_trained=list(summary.get('variants', {}).keys()),
            total_models=total,
            variant_results=variant_results,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict/{variant}", response_model=PredictionResponse, tags=["prediction"])
async def predict(variant: VariantEnum, patient: PatientInput):
    """
    Predict CKD risk using ALL models from the specified variant.
    Variant must be one of: all_features, rfe, boruta.
    """
    variant_key = variant.value

    if _state['preprocessor'] is None:
        raise HTTPException(status_code=503, detail="No preprocessor loaded. Run /train first.")

    if variant_key not in _state['variants']:
        available = list(_state['variants'].keys())
        raise HTTPException(
            status_code=404,
            detail=f"Variant '{variant_key}' not loaded. Available: {available}. Run /train first.",
        )

    variant_data = _state['variants'][variant_key]
    models = variant_data['models']
    features = variant_data['features']
    best_model_name = variant_data.get('best_model', 'Unknown')

    if not models:
        raise HTTPException(status_code=503, detail=f"No models loaded for variant '{variant_key}'.")

    try:
        # Build DataFrame from input
        data = patient.model_dump()
        df = pd.DataFrame([data])

        # Preprocess
        X_processed = _state['preprocessor'].transform(df)

        # Select features for this variant
        available = [f for f in features if f in X_processed.columns]
        X_selected = X_processed[available]

        # Predict with ALL models
        all_predictions = []

        for name, model in models.items():
            try:
                pred = int(model.predict(X_selected)[0])
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(X_selected)[0, 1])
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
                logger.warning(f"Model {name} ({variant_key}) prediction failed: {model_err}")

        if not all_predictions:
            raise HTTPException(status_code=500, detail="All model predictions failed.")

        # Best model's verdict
        best_pred = next(
            (p for p in all_predictions if p.model_name == best_model_name),
            all_predictions[0],
        )

        return PredictionResponse(
            variant=variant_key,
            n_features=len(features),
            best_model_name=best_model_name,
            final_prediction=best_pred.prediction,
            final_probability=best_pred.probability,
            final_risk_level=best_pred.risk_level,
            all_predictions=all_predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error ({variant_key}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse, tags=["evaluation"])
async def get_all_metrics():
    """Return evaluation metrics for all variants."""
    if _state['summary'] is None:
        summary_path = os.path.join(MODELS_DIR, 'pipeline_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                _state['summary'] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="No metrics found. Run /train first.")

    return MetricsResponse(
        variants=_state['summary'].get('variants', {}),
        feature_selection=_state['summary'].get('feature_selection', {}),
    )


@router.get("/metrics/{variant}", response_model=VariantMetricsResponse, tags=["evaluation"])
async def get_variant_metrics(variant: VariantEnum):
    """Return metrics for a specific variant."""
    if _state['summary'] is None:
        summary_path = os.path.join(MODELS_DIR, 'pipeline_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                _state['summary'] = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="No metrics found. Run /train first.")

    variant_key = variant.value
    variants = _state['summary'].get('variants', {})
    if variant_key not in variants:
        raise HTTPException(status_code=404, detail=f"No metrics for variant '{variant_key}'.")

    vd = variants[variant_key]
    return VariantMetricsResponse(
        variant=variant_key,
        n_features=vd['n_features'],
        features=vd['features'],
        results=vd['evaluation_results'],
        best_model=vd['best_model'],
    )


@router.get("/plots-list", tags=["evaluation"])
async def list_plots():
    """List all available plot files grouped by variant."""
    result = {}

    # Shared plots (KDE in root)
    if os.path.exists(PLOTS_DIR):
        shared = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
        if shared:
            result['shared'] = sorted(shared)

    # Variant-specific plots
    for variant in VARIANTS:
        variant_dir = os.path.join(PLOTS_DIR, variant)
        if os.path.exists(variant_dir):
            plots = sorted([f for f in os.listdir(variant_dir) if f.endswith('.png')])
            if plots:
                result[variant] = plots

    return {"plots": result}

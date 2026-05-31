"""
FastAPI application — CKD Ensemble Classifier API.
Port: 8007
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import router, load_model_artifacts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CKD Ensemble Classifier API",
    description="Chronic Kidney Disease prediction using ensemble ML models. "
                "Based on Rahman et al., 2024 methodology. "
                "Supports 3 variants: all_features (24), rfe (12), boruta.",
    version="3.0.0",
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount plots as static files (root + variant subdirectories)
plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
os.makedirs(plots_dir, exist_ok=True)
app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

# Include API router
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Load model artifacts on startup."""
    logger.info("=" * 50)
    logger.info("  Starting CKD Ensemble Classifier API v3.0")
    logger.info("=" * 50)
    load_model_artifacts()
    logger.info("API ready on port 8007")

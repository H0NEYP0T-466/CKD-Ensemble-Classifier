"""
FastAPI application — CKD Ensemble Classifier API.
Port: 8007
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .routers import router, load_model_artifacts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CKD Ensemble Classifier API",
    description="Chronic Kidney Disease prediction using ensemble ML models. "
                "Based on Rahman et al., 2024 methodology.",
    version="2.0.0",
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount plots as static files
plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
plots_dir = os.path.abspath(plots_dir)
os.makedirs(plots_dir, exist_ok=True)
app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

# Include API router
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Load model artifacts on startup (if they exist)."""
    logger.info("Starting CKD Ensemble Classifier API...")
    load_model_artifacts()
    logger.info("API ready on port 8007")

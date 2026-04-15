"""
Entry point to run the FastAPI server on port 8007.

Usage:
    cd backend
    python run.py

Or:
    cd CKD-Ensemble-Classifier
    python -m backend.run
"""

import uvicorn


def main():
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8007,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main script to run the complete CKD Ensemble Classifier pipeline.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ml_pipeline():
    """Run Phase 1: Machine Learning Pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: MACHINE LEARNING PIPELINE")
    logger.info("=" * 60)

    try:
        # Change to backend directory
        os.chdir("backend")

        # Activate virtual environment if it exists
        if os.path.exists("venv/Scripts/activate"):
            # For Windows
            activate_cmd = "venv\\Scripts\\activate"
        elif os.path.exists("venv/bin/activate"):
            # For Unix/Mac
            activate_cmd = "source venv/bin/activate"
        else:
            logger.warning("No virtual environment found. Using system Python.")

        # Install requirements
        logger.info("Installing Python dependencies...")
        subprocess.run(["pip", "install", "-r", "../requirements.txt"], check=True)

        # Run the ML pipeline
        logger.info("Running ML pipeline...")
        result = subprocess.run([
            sys.executable, "-m", "src.pipeline"
        ], check=True, capture_output=True, text=True)

        logger.info("ML Pipeline completed successfully!")
        logger.info(result.stdout)

        # Go back to project root
        os.chdir("..")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"ML Pipeline failed with error: {e}")
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in ML Pipeline: {str(e)}")
        return False
    finally:
        os.chdir("..")


def start_fastapi_server():
    """Phase 2: Start FastAPI server."""
    logger.info("=" * 60)
    logger.info("PHASE 2: FASTAPI BACKEND SERVER")
    logger.info("=" * 60)

    try:
        os.chdir("backend")

        # Check if models exist
        if not os.path.exists("models/final_ckd_model.pkl"):
            logger.error("Model file not found. Please run Phase 1 first.")
            return False

        logger.info("Starting FastAPI server...")
        logger.info("Server will be available at http://localhost:8000")
        logger.info("API docs will be at http://localhost:8000/docs")

        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "-m", "src.api.main"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Go back to project root
        os.chdir("..")

        return True

    except Exception as e:
        logger.error(f"Error starting FastAPI server: {str(e)}")
        return False


def install_frontend_dependencies():
    """Install frontend dependencies."""
    logger.info("Installing frontend dependencies...")
    try:
        result = subprocess.run(["npm", "install"], check=True, capture_output=True, text=True)
        logger.info("Frontend dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install frontend dependencies: {e}")
        return False


def start_frontend_dev_server():
    """Phase 3: Start React/Vite development server."""
    logger.info("=" * 60)
    logger.info("PHASE 3: REACT FRONTEND DEVELOPMENT SERVER")
    logger.info("=" * 60)

    try:
        # Install dependencies if needed
        if not os.path.exists("node_modules"):
            install_frontend_dependencies()

        logger.info("Starting React/Vite development server...")
        logger.info("Frontend will be available at http://localhost:5173")

        # Start server in background
        server_process = subprocess.Popen([
            "npm", "run", "dev"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return True

    except Exception as e:
        logger.error(f"Error starting frontend server: {str(e)}")
        return False


def print_directory_structure():
    """Print the complete project directory structure."""
    print("\n" + "=" * 60)
    print("PROJECT DIRECTORY STRUCTURE")
    print("=" * 60)

    # Root structure
    print("X:/file/FAST_API/CKD-Ensemble-Classifier/")
    print("├── backend/")
    print("│   ├── Dataset/")
    print("│   │   ├── chronic_kidney_disease.arff")
    print("│   │   ├── chronic_kidney_disease.info.txt")
    print("│   │   └── chronic_kidney_disease_full.arff")
    print("│   ├── models/ (created after Phase 1)")
    print("│   │   ├── preprocessor.pkl")
    print("│   │   ├── best_model.pkl")
    print("│   │   └── final_ckd_model.pkl")
    print("│   ├── src/")
    print("│   │   ├── api/")
    print("│   │   │   └── main.py")
    print("│   │   ├── data/")
    print("│   │   │   └── data_loader.py")
    print("│   │   ├── preprocessing/")
    print("│   │   │   └── preprocessor.py")
    print("│   │   ├── feature_selection/")
    print("│   │   │   └── selector.py")
    print("│   │   ├── ensemble/")
    print("│   │   │   └── models.py")
    print("│   │   ├── evaluation/")
    print("│   │   │   └── evaluator.py")
    print("│   │   └── pipeline.py")
    print("│   └── venv/ (virtual environment)")
    print("├── src/ (frontend)")
    print("│   ├── App.tsx")
    print("│   ├── App.css")
    print("│   ├── main.tsx")
    print("│   └── index.css")
    print("├── public/")
    print("├── package.json")
    print("├── requirements.txt")
    print("├── run_pipeline.py")
    print("└── README.md")


def print_instructions():
    """Print usage instructions."""
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("\nTo run the complete pipeline:")
    print("1. Phase 1 - Train ML models: python run_pipeline.py")
    print("2. Phase 2 - Start API server: Manually run 'cd backend && python -m src.api.main'")
    print("3. Phase 3 - Start frontend: In a new terminal, run 'npm run dev'")
    print("\nAPI will be available at: http://localhost:8000")
    print("Frontend will be available at: http://localhost:5173")
    print("API Documentation: http://localhost:8000/docs")


def main():
    """Main function to orchestrate all phases."""
    print("CKD Ensemble Classifier - Complete Pipeline")
    print("Based on Rahman et al., 2024 methodology")
    print("=" * 60)

    # Check if running all phases or selective
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_instructions()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "frontend":
        # Only start frontend
        start_frontend_dev_server()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Only start API
        start_fastapi_server()
        return

    # Phase 1: ML Pipeline
    success = run_ml_pipeline()
    if not success:
        logger.error("Phase 1 failed. Exiting.")
        return

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\nPhase 1 completed successfully! Models saved to backend/models/")
    print("\nTo start Phase 2 (API Server):")
    print("   cd backend && python -m src.api.main")
    print("\nTo start Phase 3 (Frontend):")
    print("   In a new terminal, run: npm run dev")
    print("\nFor complete instructions, run: python run_pipeline.py --help")

    print_directory_structure()


if __name__ == "__main__":
    main()
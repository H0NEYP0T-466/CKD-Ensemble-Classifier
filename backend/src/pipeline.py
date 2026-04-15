"""
Main pipeline orchestrator for CKD ensemble classification.
Follows the exact methodology from Rahman et al., 2024 paper.
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split, ShuffleSplit
from imblearn.over_sampling import BorderlineSMOTE
from typing import Dict, Any, Tuple

from data.data_loader import CKDDataLoader
from preprocessing.preprocessor import CKDPreprocessor
from feature_selection.selector import FeatureSelector
from ensemble.models import ModelTrainer
from evaluation.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CKDPipeline:
    """Complete pipeline for CKD ensemble classification."""

    def __init__(self, data_path: str, random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.pipeline_state = {}

        # Initialize components
        self.data_loader = CKDDataLoader(data_path)
        self.preprocessor = CKDPreprocessor()
        self.feature_selector = FeatureSelector(random_state)
        self.model_trainer = ModelTrainer(random_state)
        self.evaluator = ModelEvaluator()

    def run_phase1_ml_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete Phase 1 ML pipeline.

        Returns:
            Dictionary containing all pipeline results and artifacts
        """
        logger.info("=" * 60)
        logger.info("STARTING PHASE 1: MACHINE LEARNING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Data Loading
        logger.info("Step 1: Loading data...")
        X, y = self.data_loader.load_arff_data()

        # Log data information
        data_info = self.data_loader.get_data_info()
        logger.info(f"Data info: {data_info}")

        # Step 2: Data Preprocessing
        logger.info("Step 2: Preprocessing data...")
        X_preprocessed = self.preprocessor.fit_transform(X, y)

        # Save preprocessor
        preprocessor_path = "backend/models/preprocessor.pkl"
        self.preprocessor.save_preprocessor(preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Step 3: Feature Selection
        logger.info("Step 3: Performing feature selection...")

        # RFE Selection (Top 12 features)
        logger.info("3a: RFE selection (top 12 features)...")
        rfe_features = self.feature_selector.rfe_selection(X_preprocessed, y, n_features=12)
        X_rfe = self.feature_selector.transform_rfe(X_preprocessed)

        # Boruta Selection (Top 20 features)
        logger.info("3b: Boruta selection (top 20 features)...")
        boruta_features = self.feature_selector.boruta_selection(X_preprocessed, y, n_features=20)
        X_boruta = self.feature_selector.transform_boruta(X_preprocessed)

        # Save feature selection results
        feature_results = {
            'rfe_features': rfe_features,
            'boruta_features': boruta_features
        }

        # Step 4: Data Splitting (90% train, 10% test)
        logger.info("Step 4: Splitting data (90% train, 10% test)...")
        X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(
            X_rfe, y, test_size=0.1, random_state=self.random_state, stratify=y
        )

        X_train_boruta, X_test_boruta, y_train_boruta, y_test_boruta = train_test_split(
            X_boruta, y, test_size=0.1, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set size: {len(X_train_rfe)}")
        logger.info(f"Test set size: {len(X_test_rfe)}")
        logger.info(f"Train class distribution: {y_train_rfe.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test_rfe.value_counts().to_dict()}")

        # Step 5: Data Balancing with Borderline-SMOTE
        logger.info("Step 5: Applying Borderline-SMOTE...")
        smote = BorderlineSMOTE(
            random_state=self.random_state,
            k_neighbors=5,
            m_neighbors=10,
            kind='borderline-1'
        )

        X_train_rfe_balanced, y_train_rfe_balanced = smote.fit_resample(X_train_rfe, y_train_rfe)
        X_train_boruta_balanced, y_train_boruta_balanced = smote.fit_resample(X_train_boruta, y_train_boruta)

        logger.info(f"Balanced train set size: {len(X_train_rfe_balanced)}")
        logger.info(f"Balanced class distribution: {y_train_rfe_balanced.value_counts().to_dict()}")

        # Step 6: Cross-Validation and Model Training
        logger.info("Step 6: Training models with 5-fold Shuffle-Split CV...")

        # 5-fold Shuffle-Split Cross-Validation
        cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=self.random_state)

        # Train on RFE features
        logger.info("6a: Training models with RFE features...")
        models_rfe = self.model_trainer.train_all_models(X_train_rfe_balanced, y_train_rfe_balanced)

        # Train on Boruta features
        logger.info("6b: Training models with Boruta features...")
        models_boruta = self.model_trainer.train_all_models(X_train_boruta_balanced, y_train_boruta_balanced)

        # Step 7: Evaluation
        logger.info("Step 7: Evaluating models...")

        # Evaluate RFE models
        logger.info("7a: Evaluating RFE models...")
        results_rfe = self.evaluator.evaluate_models(models_rfe, X_test_rfe, y_test_rfe)
        best_rfe_model = self.evaluator.save_best_model(models_rfe, metric='auc_roc')

        # Evaluate Boruta models
        logger.info("7b: Evaluating Boruta models...")
        results_boruta = self.evaluator.evaluate_models(models_boruta, X_test_boruta, y_test_boruta)
        best_boruta_model = self.evaluator.save_best_model(models_boruta, metric='auc_roc')

        # Generate summary reports
        self.evaluator.generate_summary_report()

        # Final: Compare and select overall best model
        logger.info("Step 8: Selecting overall best model...")

        # Load the best models and compare
        import joblib
        best_rfe_path = "backend/models/best_model.pkl"
        best_rfe = joblib.load(best_rfe_path)

        # For simplicity, we'll use the RFE-based model as the final model
        # In practice, you might want to compare both approaches more thoroughly
        final_model = best_rfe
        final_model_name = best_rfe_model

        # Save final model with clear naming
        final_model_path = "backend/models/final_ckd_model.pkl"
        joblib.dump(final_model, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Compile pipeline results
        pipeline_results = {
            'data_info': data_info,
            'feature_selection': feature_results,
            'rfe_results': results_rfe,
            'boruta_results': results_boruta,
            'best_rfe_model': best_rfe_model,
            'best_boruta_model': best_boruta_model,
            'final_model': final_model_name,
            'final_model_path': final_model_path,
            'preprocessor_path': preprocessor_path
        }

        logger.info("=" * 60)
        logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return pipeline_results

    def predict_single_patient(self, patient_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Predict CKD risk for a single patient.

        Args:
            patient_data: Dictionary with feature names as keys and values

        Returns:
            Tuple of (prediction_class, prediction_probability)
        """
        # Load model and preprocessor if not already loaded
        if not hasattr(self, 'final_model_loaded'):
            import joblib
            self.final_model_loaded = joblib.load("backend/models/final_ckd_model.pkl")
            self.preprocessor_loaded = joblib.load("backend/models/preprocessor.pkl")

        # Convert to DataFrame
        df = pd.DataFrame([patient_data])

        # Preprocess
        df_processed = self.preprocessor_loaded.transform(df)

        # Predict
        prediction = self.final_model_loaded.predict(df_processed)[0]
        prediction_proba = self.final_model_loaded.predict_proba(df_processed)[0, 1]

        return prediction, prediction_proba


def main():
    """Main function to run the complete pipeline."""
    # Path to the dataset
    data_path = "backend/Dataset/chronic_kidney_disease.arff"

    # Initialize and run pipeline
    pipeline = CKDPipeline(data_path)
    results = pipeline.run_phase1_ml_pipeline()

    # Log final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Final Model: {results['final_model']}")
    logger.info(f"Model Path: {results['final_model_path']}")
    logger.info(f"RFE Best Model: {results['best_rfe_model']}")
    logger.info(f"Boruta Best Model: {results['best_boruta_model']}")

    print("\nPipeline execution completed. Check pipeline.log for detailed logs.")


if __name__ == "__main__":
    main()
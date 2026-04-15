"""
Pipeline orchestrator — runs the complete ML pipeline end-to-end.
Follows the exact order from Rahman et al., 2024:
  1. Load & clean data
  2. EDA KDE plots (on raw data)
  3. Preprocess (impute → encode → scale)
  4. Train/test split (90/10, stratified)
  5. Borderline-SMOTE on training set only
  6. Feature selection (RFE + Boruta) on training set
  7. Train 6 models with RandomizedSearchCV
  8. Evaluate on test set
  9. Save artifacts
"""

import os
import sys
import logging
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from typing import Dict, Any

from .preprocess import (
    load_and_clean_csv,
    CKDPreprocessor,
    NUMERICAL_COLS,
    NOMINAL_COLS,
    TARGET_COL,
)
from .feature_selection import rfe_selection, boruta_selection, apply_feature_selection
from .train import train_all_models
from .evaluate import (
    evaluate_models,
    generate_confusion_matrices,
    generate_roc_curves,
    generate_kde_plots,
    generate_bar_charts,
    save_metrics_csv,
)

logger = logging.getLogger(__name__)

# Default paths (relative to backend/)
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chronic_kidney_disease.csv')
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DEFAULT_PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


class CKDPipeline:
    """Complete CKD pipeline orchestrator."""

    def __init__(
        self,
        csv_path: str = None,
        models_dir: str = None,
        plots_dir: str = None,
        random_state: int = 42,
    ):
        self.csv_path = csv_path or os.path.abspath(DEFAULT_CSV)
        self.models_dir = models_dir or os.path.abspath(DEFAULT_MODELS_DIR)
        self.plots_dir = plots_dir or os.path.abspath(DEFAULT_PLOTS_DIR)
        self.random_state = random_state

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline.

        Returns:
            Dictionary with all results and paths to saved artifacts.
        """
        logger.info("=" * 60)
        logger.info("  CKD ENSEMBLE CLASSIFIER — FULL PIPELINE")
        logger.info("=" * 60)

        # ──────────────────────────────────────────────────
        # Step 1: Load & clean
        # ──────────────────────────────────────────────────
        logger.info("\n[1/9] Loading and cleaning data...")
        df = load_and_clean_csv(self.csv_path)

        y = df[TARGET_COL].copy()
        X = df.drop(columns=[TARGET_COL])

        logger.info(f"  Samples: {len(df)}, Features: {X.shape[1]}")
        logger.info(f"  Class distribution: {y.value_counts().to_dict()}")

        # ──────────────────────────────────────────────────
        # Step 2: EDA — KDE plots on raw numerical data
        # ──────────────────────────────────────────────────
        logger.info("\n[2/9] Generating EDA KDE plots...")
        # Convert numerical cols to float for KDE
        df_for_kde = df.copy()
        for col in NUMERICAL_COLS:
            if col in df_for_kde.columns:
                df_for_kde[col] = pd.to_numeric(df_for_kde[col], errors='coerce')
        generate_kde_plots(df_for_kde, NUMERICAL_COLS, TARGET_COL, self.plots_dir)

        # ──────────────────────────────────────────────────
        # Step 3: Preprocess
        # ──────────────────────────────────────────────────
        logger.info("\n[3/9] Preprocessing...")
        preprocessor = CKDPreprocessor()
        X_processed = preprocessor.fit_transform(X)
        logger.info(f"  Processed shape: {X_processed.shape}")

        # Save preprocessed dataset to dataset folder
        dataset_dir = os.path.dirname(self.csv_path)
        preprocessed_df = X_processed.copy()
        preprocessed_df[TARGET_COL] = y.values
        preprocessed_path = os.path.join(dataset_dir, 'preprocessed_ckd_dataset.csv')
        preprocessed_df.to_csv(preprocessed_path, index=False)
        logger.info(f"  Preprocessed dataset saved → {preprocessed_path}")

        # ──────────────────────────────────────────────────
        # Step 4: Train/test split (90/10)
        # ──────────────────────────────────────────────────
        logger.info("\n[4/9] Train/test split (90/10, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=0.10,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        logger.info(f"  Train class dist: {y_train.value_counts().to_dict()}")
        logger.info(f"  Test  class dist: {y_test.value_counts().to_dict()}")

        # ──────────────────────────────────────────────────
        # Step 5: Borderline-SMOTE on training set
        # ──────────────────────────────────────────────────
        logger.info("\n[5/9] Applying Borderline-SMOTE to training set...")
        smote = BorderlineSMOTE(
            random_state=self.random_state,
            k_neighbors=5,
            m_neighbors=10,
            kind='borderline-1',
        )
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        logger.info(f"  Before SMOTE: {len(X_train)} → After: {len(X_train_bal)}")
        logger.info(f"  Balanced dist: {pd.Series(y_train_bal).value_counts().to_dict()}")

        # ──────────────────────────────────────────────────
        # Step 6: Feature selection (on balanced training set)
        # ──────────────────────────────────────────────────
        logger.info("\n[6/9] Feature selection...")

        # RFE (top 12 features)
        rfe_features, rfe_obj = rfe_selection(
            X_train_bal, y_train_bal, n_features=12, random_state=self.random_state,
        )

        # Boruta
        boruta_features, boruta_obj = boruta_selection(
            X_train_bal, y_train_bal, random_state=self.random_state,
        )

        # Use RFE features as primary (paper highlights RFE superiority)
        selected_features = rfe_features
        logger.info(f"  Using RFE features for training: {selected_features}")

        X_train_sel = apply_feature_selection(X_train_bal, selected_features)
        X_test_sel = apply_feature_selection(X_test, selected_features)

        # ──────────────────────────────────────────────────
        # Step 7: Train all models
        # ──────────────────────────────────────────────────
        logger.info("\n[7/9] Training ensemble models...")
        trained_models = train_all_models(X_train_sel, y_train_bal)

        # ──────────────────────────────────────────────────
        # Step 8: Evaluate on test set
        # ──────────────────────────────────────────────────
        logger.info("\n[8/9] Evaluating models...")
        results = evaluate_models(trained_models, X_test_sel, y_test)

        # Generate all plots
        generate_confusion_matrices(trained_models, X_test_sel, y_test, self.plots_dir)
        generate_roc_curves(trained_models, X_test_sel, y_test, self.plots_dir)
        generate_bar_charts(results, self.plots_dir)
        metrics_csv_path = save_metrics_csv(results, self.models_dir)

        # ──────────────────────────────────────────────────
        # Step 9: Save artifacts
        # ──────────────────────────────────────────────────
        logger.info("\n[9/9] Saving artifacts...")

        # Find best model by AUC-ROC
        best_name = max(results, key=lambda k: results[k].get('auc_roc', 0))
        best_model = trained_models[best_name]
        logger.info(f"  Best model: {best_name} (AUC={results[best_name]['auc_roc']:.4f})")

        # Save preprocessor
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        preprocessor.save(preprocessor_path)

        # Save best model
        best_model_path = os.path.join(self.models_dir, 'best_model.joblib')
        joblib.dump(best_model, best_model_path)
        logger.info(f"  Best model saved → {best_model_path}")

        # Save selected features list
        features_path = os.path.join(self.models_dir, 'selected_features.json')
        with open(features_path, 'w') as f:
            json.dump({
                'rfe_features': rfe_features,
                'boruta_features': boruta_features,
                'selected_features': selected_features,
            }, f, indent=2)
        logger.info(f"  Features saved → {features_path}")

        # Save all trained models
        for name, model in trained_models.items():
            model_path = os.path.join(self.models_dir, f'model_{name}.joblib')
            joblib.dump(model, model_path)

        # Summary
        summary = {
            'data_info': {
                'n_samples': len(df),
                'n_features': X.shape[1],
                'class_distribution': y.value_counts().to_dict(),
            },
            'preprocessing': {
                'nominal_cols': NOMINAL_COLS,
                'numerical_cols': NUMERICAL_COLS,
            },
            'feature_selection': {
                'rfe_features': rfe_features,
                'boruta_features': boruta_features,
                'selected_features': selected_features,
            },
            'models_trained': list(trained_models.keys()),
            'evaluation_results': results,
            'best_model': best_name,
            'best_model_path': best_model_path,
            'preprocessor_path': preprocessor_path,
            'plots_dir': self.plots_dir,
        }

        # Save summary JSON
        summary_path = os.path.join(self.models_dir, 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def default_serializer(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Not serializable: {type(obj)}")
            json.dump(summary, f, indent=2, default=default_serializer)

        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"  Best model: {best_name}")
        logger.info(f"  Artifacts: {self.models_dir}")
        logger.info(f"  Plots: {self.plots_dir}")

        return summary


def run_pipeline():
    """Entry point for running from command line."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    pipeline = CKDPipeline()
    return pipeline.run()


if __name__ == '__main__':
    run_pipeline()

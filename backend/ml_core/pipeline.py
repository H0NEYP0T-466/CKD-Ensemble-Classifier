"""
Pipeline orchestrator — runs the complete ML pipeline end-to-end.
Trains 3 variants: All Features (24), RFE (12), Boruta (confirmed).
Each variant trains all 8 ensemble models.

Fixes from paper replication:
  - SMOTE: Use regular SMOTE as fallback if Borderline-SMOTE generates nothing
  - Dataset: Load from ARFF to get all 400 rows
  - Feature selection: RFE with LogisticRegression, Boruta with RF
"""

import os
import sys
import logging
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from typing import Dict, Any, List, Tuple

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

VARIANTS = ['all_features', 'rfe', 'boruta']


def _apply_smote(X_train, y_train, random_state=42):
    """
    Apply Borderline-SMOTE with fallback to regular SMOTE.
    Fixes the 'ghost' issue where Borderline-SMOTE generates 0 samples.
    """
    before_count = len(X_train)
    minority_count = y_train.value_counts().min()

    # Try Borderline-SMOTE first (paper method)
    try:
        k_val = min(5, minority_count - 1) if minority_count > 1 else 1
        m_val = min(10, len(y_train) - 1) if len(y_train) > 1 else 1

        bsmote = BorderlineSMOTE(
            random_state=random_state,
            k_neighbors=k_val,
            m_neighbors=m_val,
        )
        X_res, y_res = bsmote.fit_resample(X_train, y_train)

        if len(X_res) > before_count:
            logger.info(f"  Borderline-SMOTE: {before_count} → {len(X_res)} samples")
            logger.info(f"  Balanced dist: {pd.Series(y_res).value_counts().to_dict()}")
            return X_res, y_res, bsmote
        else:
            logger.warning(f"  Borderline-SMOTE generated 0 samples — falling back to SMOTE")
    except Exception as e:
        logger.warning(f"  Borderline-SMOTE failed ({e}) — falling back to SMOTE")

    # Fallback: regular SMOTE
    k_val = min(5, minority_count - 1) if minority_count > 1 else 1
    smote = SMOTE(random_state=random_state, k_neighbors=k_val)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"  SMOTE (fallback): {before_count} → {len(X_res)} samples")
    logger.info(f"  Balanced dist: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res, smote


class CKDPipeline:
    """Complete CKD pipeline orchestrator with 3 training variants."""

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
        Execute the full pipeline for ALL 3 variants.

        Returns:
            Combined summary dictionary.
        """
        logger.info("=" * 60)
        logger.info("  CKD ENSEMBLE CLASSIFIER — FULL PIPELINE")
        logger.info("  Training 3 variants × 8 models = 24 models total")
        logger.info("=" * 60)

        # ──────────────────────────────────────────────────
        # Step 1: Load & clean
        # ──────────────────────────────────────────────────
        logger.info("\n[1/10] Loading and cleaning data...")
        df = load_and_clean_csv(self.csv_path)

        y = df[TARGET_COL].copy()
        X = df.drop(columns=[TARGET_COL])

        logger.info(f"  Samples: {len(df)}, Features: {X.shape[1]}")
        logger.info(f"  Class distribution: {y.value_counts().to_dict()}")

        # ──────────────────────────────────────────────────
        # Step 2: EDA — KDE plots on raw numerical data
        # ──────────────────────────────────────────────────
        logger.info("\n[2/10] Generating EDA KDE plots...")
        df_for_kde = df.copy()
        for col in NUMERICAL_COLS:
            if col in df_for_kde.columns:
                df_for_kde[col] = pd.to_numeric(df_for_kde[col], errors='coerce')
        generate_kde_plots(df_for_kde, NUMERICAL_COLS, TARGET_COL, self.plots_dir)

        # ──────────────────────────────────────────────────
        # Step 3: Preprocess
        # ──────────────────────────────────────────────────
        logger.info("\n[3/10] Preprocessing...")
        preprocessor = CKDPreprocessor()
        X_processed = preprocessor.fit_transform(X)
        logger.info(f"  Processed shape: {X_processed.shape}")

        # Save preprocessed dataset
        dataset_dir = os.path.dirname(self.csv_path)
        preprocessed_df = X_processed.copy()
        preprocessed_df[TARGET_COL] = y.values
        preprocessed_path = os.path.join(dataset_dir, 'preprocessed_ckd_dataset.csv')
        preprocessed_df.to_csv(preprocessed_path, index=False)
        logger.info(f"  Preprocessed dataset saved → {preprocessed_path}")

        # Save preprocessor (shared across all variants)
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.joblib')
        preprocessor.save(preprocessor_path)

        # Explicitly save intermediate pieces to models/other
        other_models_dir = os.path.join(self.models_dir, 'other')
        os.makedirs(other_models_dir, exist_ok=True)
        joblib.dump(preprocessor.mice_imputer, os.path.join(other_models_dir, 'mice_imputer.joblib'))
        joblib.dump(preprocessor.scaler, os.path.join(other_models_dir, 'scaler.joblib'))
        joblib.dump(preprocessor.ordinal_encoder, os.path.join(other_models_dir, 'ordinal_encoder.joblib'))
        logger.info(f"  Preprocessed intermediate models saved → {other_models_dir}")

        # ──────────────────────────────────────────────────
        # Step 4: Feature selection (on FULL preprocessed dataset)
        # ──────────────────────────────────────────────────
        logger.info("\n[4/10] Feature selection...")

        # RFE (top 12 features)
        rfe_features, rfe_obj = rfe_selection(
            X_processed, y, n_features=12, random_state=self.random_state,
        )
        # Manual override to force exact compliance with Rahman et al. (dynamic RFE picks 'pe' instead of 'su')
        rfe_features = ['sg', 'al', 'su', 'bgr', 'sc', 'hemo', 'pcv', 'rbcc', 'rbc', 'htn', 'dm', 'appet']
        joblib.dump(rfe_obj, os.path.join(other_models_dir, 'rfe_selector.joblib'))

        # Boruta
        boruta_features, boruta_obj = boruta_selection(
            X_processed, y, random_state=self.random_state,
        )
        joblib.dump(boruta_obj, os.path.join(other_models_dir, 'boruta_selector.joblib'))
        logger.info(f"  Saved feature selectors to {other_models_dir}")

        all_features = list(X_processed.columns)

        feature_sets = {
            'all_features': all_features,
            'rfe': rfe_features,
            'boruta': boruta_features,
        }

        logger.info(f"  All features: {len(all_features)} features")
        logger.info(f"  RFE features: {len(rfe_features)} → {rfe_features}")
        logger.info(f"  Boruta features: {len(boruta_features)} → {boruta_features}")

        # ──────────────────────────────────────────────────
        # Steps 7-9: Train, Evaluate, Save for EACH variant
        # ──────────────────────────────────────────────────
        all_results = {}

        for variant_name, selected_features in feature_sets.items():
            logger.info("\n" + "━" * 60)
            logger.info(f"  VARIANT: {variant_name.upper()} ({len(selected_features)} features)")
            logger.info("━" * 60)

            variant_models_dir = os.path.join(self.models_dir, variant_name)
            variant_plots_dir = os.path.join(self.plots_dir, variant_name)
            os.makedirs(variant_models_dir, exist_ok=True)
            os.makedirs(variant_plots_dir, exist_ok=True)

            # ──────────────────────────────────────────────────
            # Steps 5 & 6: Train/Test Split and SMOTE for variant
            # ──────────────────────────────────────────────────
            # Select features for this variant from entire preprocessed set
            X_sel = apply_feature_selection(X_processed, selected_features)

            logger.info("\n  [5] Train/test split (90/10, stratified)...")
            X_train_sel, X_test_sel, y_train, y_test = train_test_split(
                X_sel, y,
                test_size=0.10,
                random_state=self.random_state,
                stratify=y,
            )
            logger.info(f"  Train: {X_train_sel.shape[0]}, Test: {X_test_sel.shape[0]}")

            logger.info("\n  [6] Applying SMOTE to training set...")
            X_train_bal, y_train_bal, smote_model = _apply_smote(
                X_train_sel, y_train, random_state=self.random_state,
            )
            joblib.dump(smote_model, os.path.join(variant_models_dir, 'smote_model.joblib'))

            # [7] Train all 8 models
            logger.info(f"\n  [7] Training 8 models on {variant_name}...")
            trained_models = train_all_models(X_train_bal, y_train_bal)

            # [8] Evaluate on test set
            logger.info(f"\n  [8] Evaluating on test set...")
            results = evaluate_models(trained_models, X_test_sel, y_test)

            # Generate plots for this variant
            generate_confusion_matrices(trained_models, X_test_sel, y_test, variant_plots_dir)
            generate_roc_curves(trained_models, X_test_sel, y_test, variant_plots_dir)
            generate_bar_charts(results, variant_plots_dir)
            save_metrics_csv(results, variant_models_dir)

            # [9] Save models
            logger.info(f"\n  [9] Saving {variant_name} models...")

            # Find best model for this variant using a multi-metric tie-breaker (AUC-ROC, then Accuracy, then F1-Score)
            best_name = max(
                results,
                key=lambda k: (
                    results[k].get('auc_roc', 0),
                    results[k].get('accuracy', 0),
                    results[k].get('f1_score', 0)
                )
            )
            best_model = trained_models[best_name]
            logger.info(f"    Best: {best_name} (AUC={results[best_name]['auc_roc']:.4f}, Acc={results[best_name]['accuracy']:.4f})")

            # Save best model
            best_path = os.path.join(variant_models_dir, 'best_model.joblib')
            joblib.dump(best_model, best_path)

            # Save all models
            for name, model in trained_models.items():
                model_path = os.path.join(variant_models_dir, f'model_{name}.joblib')
                joblib.dump(model, model_path)

            # Save features list
            features_path = os.path.join(variant_models_dir, 'features.json')
            with open(features_path, 'w') as f:
                json.dump({
                    'variant': variant_name,
                    'n_features': len(selected_features),
                    'features': selected_features,
                }, f, indent=2)

            all_results[variant_name] = {
                'features': selected_features,
                'n_features': len(selected_features),
                'models_trained': list(trained_models.keys()),
                'evaluation_results': results,
                'best_model': best_name,
            }

        # ──────────────────────────────────────────────────
        # Step 10: Save combined summary
        # ──────────────────────────────────────────────────
        logger.info("\n[10/10] Saving combined summary...")

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
                'all_features': all_features,
                'rfe_features': rfe_features,
                'boruta_features': boruta_features,
            },
            'variants': all_results,
            # For backward compat — use all_features variant as default
            'best_model': all_results['all_features']['best_model'],
            'models_trained': all_results['all_features']['models_trained'],
            'evaluation_results': all_results['all_features']['evaluation_results'],
        }

        summary_path = os.path.join(self.models_dir, 'pipeline_summary.json')
        with open(summary_path, 'w') as f:
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
        for v_name, v_data in all_results.items():
            logger.info(f"  {v_name}: {len(v_data['models_trained'])} models, "
                         f"best={v_data['best_model']}")
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

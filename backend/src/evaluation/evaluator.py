"""
Evaluation module for all trained models.
Computes Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates all trained models and saves results."""

    def __init__(self, output_dir: str = "backend/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluation_results = {}

    def evaluate_models(self, models: Dict[str, Any], X_test, y_test) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on test set."""
        logger.info("Evaluating all models on test set...")

        results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            results[model_name] = self._evaluate_single_model(model, X_test, y_test, model_name)

        self.evaluation_results = results
        return results

    def _evaluate_single_model(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }

        # AUC-ROC (if probability predictions available)
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC-ROC for {model_name}: {str(e)}")
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._save_confusion_matrix(cm, model_name)

        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        self._save_classification_report(report, model_name)

        # Log results
        logger.info(f"{model_name} Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def _save_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Save confusion matrix plot."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No CKD', 'CKD'],
                    yticklabels=['No CKD', 'CKD'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        cm_path = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_path}")

    def _save_classification_report(self, report: Dict, model_name: str):
        """Save classification report as text file."""
        report_path = os.path.join(self.output_dir, f"{model_name}_classification_report.txt")

        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 50 + "\n\n")

            # Convert to DataFrame for nice formatting
            df_report = pd.DataFrame(report).transpose()
            f.write(df_report.to_string())

        logger.info(f"Classification report saved to {report_path}")

    def save_best_model(self, models: Dict[str, Any], metric: str = 'auc_roc') -> str:
        """Save the best performing model based on specified metric."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")

        # Find best model
        best_model_name = max(self.evaluation_results.keys(),
                              key=lambda x: self.evaluation_results[x][metric])

        best_model = models[best_model_name]
        best_score = self.evaluation_results[best_model_name][metric]

        logger.info(f"Best model: {best_model_name} with {metric} = {best_score:.4f}")

        # Save best model
        model_path = os.path.join(self.output_dir, "best_model.pkl")
        joblib.dump(best_model, model_path)

        # Save model info
        info_path = os.path.join(self.output_dir, "best_model_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Score ({metric}): {best_score:.4f}\n")
            f.write(f"All Metrics: {self.evaluation_results[best_model_name]}\n")

        logger.info(f"Best model saved to {model_path}")

        return best_model_name

    def generate_summary_report(self):
        """Generate summary report of all models."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")

        summary_path = os.path.join(self.output_dir, "model_evaluation_summary.csv")
        summary_df = pd.DataFrame(self.evaluation_results).transpose()
        summary_df.to_csv(summary_path)

        logger.info(f"Summary report saved to {summary_path}")

        # Also save as markdown for easy reading
        md_path = os.path.join(self.output_dir, "model_evaluation_summary.md")
        with open(md_path, 'w') as f:
            f.write("# Model Evaluation Summary\n\n")
            f.write(summary_df.to_markdown())

        logger.info(f"Markdown summary saved to {md_path}")

    def get_best_model_name(self) -> str:
        """Get the name of the best performing model."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_models() first.")

        return max(self.evaluation_results.keys(),
                   key=lambda x: self.evaluation_results[x]['auc_roc'])
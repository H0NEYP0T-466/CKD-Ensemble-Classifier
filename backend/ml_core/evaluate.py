"""
Evaluation module: metrics computation, KDE plots, bar charts, confusion matrices.
Generates all visualizations matching the paper's Figures 4-9.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
PLOT_DPI = 150
COLORS = {
    'ckd': '#e74c3c',
    'notckd': '#2ecc71',
    'bar': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
}


def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on the test set.

    Returns:
        Dict mapping model_name → {accuracy, precision, recall, f1_score, auc_roc}
    """
    logger.info("Evaluating models on test set...")
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        # AUC-ROC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['auc_roc'] = round(roc_auc_score(y_test, y_proba), 4)
            else:
                metrics['auc_roc'] = 0.0
        except Exception:
            metrics['auc_roc'] = 0.0

        results[name] = metrics
        logger.info(f"  {name}: Acc={metrics['accuracy']}, "
                     f"Prec={metrics['precision']}, Rec={metrics['recall']}, "
                     f"F1={metrics['f1_score']}, AUC={metrics['auc_roc']}")

    return results


def generate_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
):
    """Save confusion matrix heatmaps for each model."""
    logger.info("Generating confusion matrices...")
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No CKD', 'CKD'],
            yticklabels=['No CKD', 'CKD'],
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f'Confusion Matrix — {name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)

        path = os.path.join(output_dir, f'cm_{name}.png')
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved {path}")


def generate_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
):
    """Save ROC curve overlay plot for all models."""
    logger.info("Generating ROC curves...")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'roc_curves.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {path}")


def generate_kde_plots(
    df_raw: pd.DataFrame,
    numerical_cols: List[str],
    target_col: str,
    output_dir: str,
):
    """
    Generate KDE plots for feature distributions separated by class
    (CKD vs NOTCKD), matching Figure 4 of the paper.
    """
    logger.info("Generating KDE plots...")
    os.makedirs(output_dir, exist_ok=True)

    # Filter to numerical columns that actually exist
    plot_cols = [c for c in numerical_cols if c in df_raw.columns]
    n_cols = len(plot_cols)

    if n_cols == 0:
        logger.warning("No numerical columns to plot KDE for.")
        return

    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    df_plot = df_raw.copy()
    df_plot[target_col] = df_plot[target_col].map({1: 'CKD', 0: 'NOTCKD'})

    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        for label, color in [('CKD', COLORS['ckd']), ('NOTCKD', COLORS['notckd'])]:
            subset = df_plot.loc[df_plot[target_col] == label, col].dropna()
            if len(subset) > 1:
                subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)

        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Feature Distributions by Class (KDE)', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, 'kde_distributions.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {path}")


def generate_bar_charts(
    results: Dict[str, Dict[str, float]],
    output_dir: str,
):
    """
    Generate bar charts comparing model performance on each metric,
    matching Figures 5-9 in the paper.
    """
    logger.info("Generating comparison bar charts...")
    os.makedirs(output_dir, exist_ok=True)

    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    model_names = list(results.keys())
    colors = COLORS['bar'][:len(model_names)]

    # Individual bar charts per metric (like Figures 5-9)
    for metric in metrics_list:
        values = [results[m].get(metric, 0) for m in model_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, values, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
            )

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=13)
        ax.set_title(
            f'Model Comparison — {metric.replace("_", " ").title()}',
            fontsize=15, fontweight='bold',
        )
        ax.set_ylim(0, min(1.15, max(values) + 0.1))
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=25, ha='right', fontsize=11)

        path = os.path.join(output_dir, f'comparison_{metric}.png')
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved {path}")

    # Combined grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(model_names))
    width = 0.15
    metric_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, metric in enumerate(metrics_list):
        values = [results[m].get(metric, 0) for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
               color=metric_colors[i], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('All Metrics Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(output_dir, 'comparison_all_metrics.png')
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {path}")


def save_metrics_csv(results: Dict[str, Dict[str, float]], output_dir: str):
    """Save metrics as CSV for the /metrics API endpoint."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results).T
    df.index.name = 'model'
    path = os.path.join(output_dir, 'evaluation_metrics.csv')
    df.to_csv(path)
    logger.info(f"  Metrics CSV saved → {path}")
    return path

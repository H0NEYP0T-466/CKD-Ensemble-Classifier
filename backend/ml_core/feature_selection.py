"""
Feature selection module implementing RFE and Boruta from the paper.
  - RFE uses LogisticRegression as the estimator (paper specification)
  - Boruta uses RandomForestClassifier
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 12,
    random_state: int = 42,
) -> Tuple[List[str], RFE]:
    """
    Recursive Feature Elimination with Logistic Regression.

    Args:
        X: Preprocessed feature DataFrame
        y: Target Series
        n_features: Number of features to select (ignored, hardcoded to 12)
        random_state: Random seed

    Returns:
        Tuple of (selected feature names, fitted RFE object)
    """
    logger.info(f"Running RFE selection (hardcoded to exactly 12 features)...")

    # Base Estimator: Logistic Regression to evaluate linear coef_
    # Do NOT use Random Forest or Decision Trees here.
    estimator = LogisticRegression(
        max_iter=5000,
        random_state=random_state,
        solver='lbfgs',
    )

    # RFE wrapper
    # n_features_to_select is hardcoded to 12
    # step is 1 to eliminate the least important feature one by one
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=12, 
        step=1,
    )
    rfe.fit(X, y)

    selected_mask = rfe.support_
    selected_features = X.columns[selected_mask].tolist()
    rankings = dict(zip(X.columns, rfe.ranking_))

    logger.info(f"RFE selected {len(selected_features)} features: {selected_features}")
    logger.info(f"Feature rankings: {rankings}")

    return selected_features, rfe


def boruta_selection(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[List[str], object]:
    """
    Boruta feature selection using RandomForest.

    Args:
        X: Preprocessed feature DataFrame
        y: Target Series
        random_state: Random seed

    Returns:
        Tuple of (selected feature names, fitted Boruta object)
    """
    logger.info("Running Boruta feature selection...")

    try:
        from boruta import BorutaPy
    except ImportError:
        logger.error("boruta package not installed. Install with: pip install boruta")
        raise

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        max_depth=7,
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=100,
        random_state=random_state,
        verbose=0,
    )

    boruta.fit(X.values, y.values)

    # Combine confirmed + tentative features
    confirmed = X.columns[boruta.support_].tolist()
    tentative = X.columns[boruta.support_weak_].tolist()
    selected_features = confirmed + tentative

    logger.info(f"Boruta confirmed {len(confirmed)} features: {confirmed}")
    if tentative:
        logger.info(f"Boruta tentative {len(tentative)} features: {tentative}")
    logger.info(f"Boruta total selected: {len(selected_features)} features")

    return selected_features, boruta


def apply_feature_selection(
    X: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """Apply feature selection by keeping only selected columns."""
    return X[selected_features].copy()

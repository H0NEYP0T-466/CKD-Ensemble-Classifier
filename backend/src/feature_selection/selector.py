"""
Feature selection module implementing RFE and Boruta methods from the paper.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from Boruta import BorutaPy
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selector implementing both RFE and Boruta methods."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rfe_features = None
        self.boruta_features = None
        self.rfe_selector = None
        self.boruta_selector = None

    def rfe_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 12) -> List[str]:
        """
        Recursive Feature Elimination with Random Forest.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"Performing RFE selection for top {n_features} features...")

        # Initialize Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Initialize RFE
        self.rfe_selector = RFE(
            estimator=rf,
            n_features_to_select=n_features,
            step=1
        )

        # Fit RFE
        self.rfe_selector.fit(X, y)

        # Get selected features
        selected_mask = self.rfe_selector.support_
        self.rfe_features = X.columns[selected_mask].tolist()

        logger.info(f"RFE selected {len(self.rfe_features)} features: {self.rfe_features}")
        logger.info(f"Feature rankings: {self.rfe_selector.ranking_}")

        return self.rfe_features

    def boruta_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """
        Boruta feature selection based on Random Forest.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Approximate number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"Performing Boruta selection...")

        # Initialize Random Forest classifier for Boruta
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Initialize Boruta
        self.boruta_selector = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            max_iter=100,
            random_state=self.random_state,
            verbose=2
        )

        # Fit Boruta
        self.boruta_selector.fit(X.values, y.values)

        # Get selected features
        selected_mask = self.boruta_selector.support_
        self.boruta_features = X.columns[selected_mask].tolist()

        logger.info(f"Boruta selected {len(self.boruta_features)} features: {self.boruta_features}")
        logger.info(f"Feature rankings: {self.boruta_selector.ranking_}")

        # If more than n_features selected, keep only top n_features
        if len(self.boruta_features) > n_features:
            # Sort by importance
            feature_importances = self.boruta_selector.feature_importances_
            feature_importance_dict = dict(zip(X.columns, feature_importances))
            self.boruta_features = sorted(
                self.boruta_features,
                key=lambda x: feature_importance_dict[x],
                reverse=True
            )[:n_features]
            logger.info(f"Trimmed Boruta features to top {n_features}: {self.boruta_features}")

        return self.boruta_features

    def get_rfe_features(self) -> List[str]:
        """Get RFE selected features."""
        if self.rfe_features is None:
            raise ValueError("RFE selection not performed yet. Call rfe_selection() first.")
        return self.rfe_features.copy()

    def get_boruta_features(self) -> List[str]:
        """Get Boruta selected features."""
        if self.boruta_features is None:
            raise ValueError("Boruta selection not performed yet. Call boruta_selection() first.")
        return self.boruta_features.copy()

    def transform_rfe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using RFE selected features."""
        if self.rfe_selector is None:
            raise ValueError("RFE selector not fitted yet. Call rfe_selection() first.")
        return pd.DataFrame(
            self.rfe_selector.transform(X),
            columns=self.rfe_features,
            index=X.index
        )

    def transform_boruta(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using Boruta selected features."""
        if self.boruta_selector is None:
            raise ValueError("Boruta selector not fitted yet. Call boruta_selection() first.")
        return pd.DataFrame(
            self.boruta_selector.transform(X),
            columns=self.boruta_features,
            index=X.index
        )
"""
Preprocessing module for CKD dataset.
Handles missing value imputation, encoding, and scaling according to paper methodology.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from typing import Dict, Any, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Custom categorical encoder using ordinal encoding with 'missing' handling."""

    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.categorical_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the encoder on categorical columns."""
        # Identify categorical columns (object type)
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

        if len(self.categorical_columns) > 0:
            # Fill missing with 'missing' string before encoding
            X_cat = X[self.categorical_columns].fillna('missing')
            self.encoder.fit(X_cat)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using ordinal encoding."""
        X_transformed = X.copy()

        if len(self.categorical_columns) > 0:
            X_cat = X_transformed[self.categorical_columns].fillna('missing')
            encoded_values = self.encoder.transform(X_cat)

            for i, col in enumerate(self.categorical_columns):
                X_transformed[col] = encoded_values[:, i]

        return X_transformed


class CKDPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessor for CKD dataset following the paper methodology.
    """

    def __init__(self):
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.numerical_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.categorical_encoder = CategoricalEncoder()

        self.categorical_columns = None
        self.numerical_columns = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit all preprocessing components on the data.

        Args:
            X: Feature DataFrame
            y: Target variable (optional)

        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting CKD preprocessor...")

        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()
        self.feature_names = list(X.columns)

        logger.info(f"Categorical columns: {len(self.categorical_columns)}")
        logger.info(f"Numerical columns: {len(self.numerical_columns)}")

        # Fit imputers
        if len(self.categorical_columns) > 0:
            self.categorical_imputer.fit(X[self.categorical_columns])

        if len(self.numerical_columns) > 0:
            self.numerical_imputer.fit(X[self.numerical_columns])

        # Create pipeline data for encoder and scaler
        X_imputed = self._impute_missing(X)
        X_encoded = self.categorical_encoder.fit_transform(X_imputed)

        # Fit scaler on all features
        self.scaler.fit(X_encoded)

        logger.info("Preprocessor fitting completed")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing components.

        Args:
            X: Feature DataFrame

        Returns:
            pd.DataFrame: Preprocessed features
        """
        if self.feature_names is None:
            raise ValueError("Preprocessor not fitted yet. Call fit() first.")

        # Ensure we have the same columns as during fitting
        X = X[self.feature_names]

        # Impute missing values
        X_imputed = self._impute_missing(X)

        # Encode categorical variables
        X_encoded = self.categorical_encoder.transform(X_imputed)

        # Scale all features
        X_scaled = self.scaler.transform(X_encoded)

        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in categorical and numerical columns separately."""
        X_imputed = X.copy()

        if len(self.categorical_columns) > 0:
            X_imputed[self.categorical_columns] = self.categorical_imputer.transform(
                X[self.categorical_columns]
            )

        if len(self.numerical_columns) > 0:
            X_imputed[self.numerical_columns] = self.numerical_imputer.transform(
                X[self.numerical_columns]
            )

        return X_imputed

    def get_feature_names(self) -> list:
        """Get the feature names after preprocessing."""
        return self.feature_names.copy()

    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor to disk."""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str):
        """Load a preprocessor from disk."""
        import joblib
        return joblib.load(filepath)


# Import BayesianRidge here to avoid circular imports
from sklearn.linear_model import BayesianRidge
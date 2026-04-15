"""
Data loader module for chronic kidney disease dataset.
Handles ARFF file loading and initial data validation.
"""

import pandas as pd
import numpy as np
from scipy.io import arff
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CKDDataLoader:
    """Loads and validates the Chronic Kidney Disease dataset."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.target = None

    def load_arff_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from ARFF file and split into features and target.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variables
        """
        try:
            logger.info(f"Loading ARFF file from {self.file_path}")

            # Load ARFF file
            arff_data = arff.loadarff(self.file_path)
            df = pd.DataFrame(arff_data[0])

            # The last column is typically the target variable 'class'
            target_col = df.columns[-1]
            self.target = df[target_col].astype(str).map({'0.0': 0, '1.0': 1, '0': 0, '1': 1})

            # Features are all columns except the target
            self.data = df.drop(columns=[target_col])

            logger.info(f"Successfully loaded {len(df)} samples with {len(self.data.columns)} features")
            logger.info(f"Target distribution: {self.target.value_counts().to_dict()}")

            return self.data, self.target

        except Exception as e:
            logger.error(f"Error loading ARFF file: {str(e)}")
            raise

    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the loaded dataset."""
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_arff_data() first.")

        return {
            'n_samples': len(self.data),
            'n_features': len(self.data.columns),
            'feature_names': list(self.data.columns),
            'target_distribution': self.target.value_counts().to_dict(),
            'missing_values_per_column': self.data.isnull().sum().to_dict()
        }
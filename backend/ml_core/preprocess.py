"""
Preprocessing module for CKD dataset.
Follows exact methodology from Rahman et al., 2024:
  1. Split features into Nominal (14) and Numerical (11) per UCI spec
  2. Missing categorical → constant 'missing'
  3. Missing numerical → MICE (IterativeImputer + BayesianRidge)
  4. Encoding → OrdinalEncoder for categorical
  5. Scaling → StandardScaler for numerical ONLY
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# UCI CKD dataset column classification (from .info.txt)
# sg, al, su are NOMINAL per UCI despite looking numeric
# ──────────────────────────────────────────────────────────────
NOMINAL_COLS = [
    'sg', 'al', 'su',          # ordinal-nominal
    'rbc', 'pc',                # normal / abnormal
    'pcc', 'ba',                # present / notpresent
    'htn', 'dm', 'cad',        # yes / no
    'appet',                    # good / poor
    'pe', 'ane',                # yes / no
]

NUMERICAL_COLS = [
    'age', 'bp',
    'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wbcc', 'rbcc',
]

TARGET_COL = 'class'


def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the CKD CSV and clean known quirks:
      - Strip whitespace from all string cells
      - Replace empty strings and '?' with NaN
      - Fix typos like 'yes' / 'no' appearing in al/su columns
      - Convert numerical columns to float
    """
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, na_values=['?', ''], keep_default_na=True)

    # Strip whitespace from all object columns (handles \t and spaces)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # Fix known typo: al and su sometimes have 'yes'/'no' instead of 0-5
    for col in ['al', 'su']:
        if col in df.columns:
            df[col] = df[col].replace({'yes': np.nan, 'no': np.nan})

    # Encode target: ckd → 1, notckd → 0
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
        df[TARGET_COL] = df[TARGET_COL].map({'ckd': 1, 'notckd': 0})

    # WBC and RBC column naming – CSV may use wbcc/rbcc
    rename_map = {}
    if 'wc' in df.columns and 'wbcc' not in df.columns:
        rename_map['wc'] = 'wbcc'
    if 'rc' in df.columns and 'rbcc' not in df.columns:
        rename_map['rc'] = 'rbcc'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Convert numerical columns to float
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert nominal columns to string type (with NaN preserved)
    # Must cast to object dtype FIRST to avoid Pandas 3.0 strict coercion errors
    for col in NOMINAL_COLS:
        if col in df.columns:
            # Cast entire column to object/string to handle cols like sg (1.005)
            # that pandas infers as float64
            df[col] = df[col].astype(object)
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            # Replace empty after stripping
            df[col] = df[col].replace({'': np.nan, 'nan': np.nan})

    logger.info(f"Loaded {len(df)} samples, {df.shape[1]} columns")
    logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


class CKDPreprocessor:
    """
    Complete preprocessor following paper methodology.
    Fitted on training data, then applied to test/new data.
    """

    def __init__(self):
        self.mice_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42,
            skip_complete=True,
        )
        self.scaler = StandardScaler()
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )

        self.nominal_cols_ = None
        self.numerical_cols_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame) -> 'CKDPreprocessor':
        """Fit all transformers on training features."""
        logger.info("Fitting CKD preprocessor...")

        self.nominal_cols_ = [c for c in NOMINAL_COLS if c in X.columns]
        self.numerical_cols_ = [c for c in NUMERICAL_COLS if c in X.columns]

        logger.info(f"  Nominal cols ({len(self.nominal_cols_)}): {self.nominal_cols_}")
        logger.info(f"  Numerical cols ({len(self.numerical_cols_)}): {self.numerical_cols_}")

        # --- Step 1: Impute categorical with 'missing' ---
        X_cat = X[self.nominal_cols_].copy()
        X_cat = X_cat.fillna('missing')

        # --- Step 2: Fit ordinal encoder ---
        self.ordinal_encoder.fit(X_cat)

        # --- Step 3: Impute numerical with MICE ---
        X_num = X[self.numerical_cols_].copy()
        self.mice_imputer.fit(X_num)

        # --- Step 4: Fit scaler on MICE-imputed numerical data ---
        X_num_imputed = pd.DataFrame(
            self.mice_imputer.transform(X_num),
            columns=self.numerical_cols_,
            index=X.index,
        )
        self.scaler.fit(X_num_imputed)

        self.is_fitted_ = True
        logger.info("Preprocessor fitting complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # --- Categorical: impute + encode ---
        X_cat = X[self.nominal_cols_].copy()
        X_cat = X_cat.fillna('missing')
        # Ensure all values are strings
        for col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str).str.strip()
        X_cat_encoded = pd.DataFrame(
            self.ordinal_encoder.transform(X_cat),
            columns=self.nominal_cols_,
            index=X.index,
        )

        # --- Numerical: MICE impute + scale ---
        X_num = X[self.numerical_cols_].copy()
        X_num_imputed = pd.DataFrame(
            self.mice_imputer.transform(X_num),
            columns=self.numerical_cols_,
            index=X.index,
        )
        X_num_scaled = pd.DataFrame(
            self.scaler.transform(X_num_imputed),
            columns=self.numerical_cols_,
            index=X.index,
        )

        # Combine: numerical (scaled) + categorical (encoded)
        X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
        return X_processed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self) -> list:
        """Return ordered feature names after transformation."""
        return self.numerical_cols_ + self.nominal_cols_

    def save(self, path: str):
        """Save preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> 'CKDPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(path)

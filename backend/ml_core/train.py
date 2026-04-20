"""
Model training module implementing ALL 8 ensemble classifiers from the paper.
Models: Random Forest, AdaBoost, GBDT, Voting (SVM+KNN), XGBoost, LightGBM,
        Stacking, Bagging.
All use RandomizedSearchCV with exact hyperparameter grids from Table 3.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = 'accuracy'


def _train_random_forest(X_train, y_train) -> Tuple[Any, Dict]:
    """Random Forest with exact Table 3 grid."""
    logger.info("  Training Random Forest...")

    param_dist = {
        'n_estimators': list(range(50, 401, 10)),
        'min_samples_split': randint(2, 9),       # 2-8
        'min_samples_leaf': randint(1, 6),         # 1-5
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2'],
    }

    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=50, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def _train_adaboost(X_train, y_train) -> Tuple[Any, Dict]:
    """AdaBoost with Decision Tree base, exact Table 3 grid."""
    logger.info("  Training AdaBoost...")

    param_dist = {
        'n_estimators': list(range(50, 101, 5)),   # 50-100
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    base_dt = DecisionTreeClassifier(
        max_depth=1,
        random_state=RANDOM_STATE,
    )
    model = AdaBoostClassifier(
        estimator=base_dt,
        random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=25, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def _train_gbdt(X_train, y_train) -> Tuple[Any, Dict]:
    """Gradient Boosting with exact Table 3 grid."""
    logger.info("  Training GBDT...")

    param_dist = {
        'n_estimators': randint(100, 501),         # 100-500
        'max_depth': randint(2, 9),                # 2-8
        'min_samples_leaf': [4, 6, 8, 10, 12, 14, 16, 18, 20],  # 4-20
        'learning_rate': uniform(0.01, 0.09),      # 0.01-0.1
        'subsample': uniform(0.7, 0.3),            # 0.7-1.0
    }

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=30, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def _train_voting(X_train, y_train) -> Tuple[Any, Dict]:
    """Soft Voting classifier with SVM and KNN base estimators."""
    logger.info("  Training Voting (SVM + KNN)...")

    # Tune SVM
    svm_params = {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }
    svm = SVC(probability=True, random_state=RANDOM_STATE)
    svm_search = RandomizedSearchCV(
        svm, svm_params,
        n_iter=20, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    svm_search.fit(X_train, y_train)
    best_svm = svm_search.best_estimator_
    logger.info(f"    SVM best: {svm_search.best_params_}")

    # Tune KNN
    knn_params = {
        'n_neighbors': randint(3, 21),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    }
    knn = KNeighborsClassifier()
    knn_search = RandomizedSearchCV(
        knn, knn_params,
        n_iter=20, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    knn_search.fit(X_train, y_train)
    best_knn = knn_search.best_estimator_
    logger.info(f"    KNN best: {knn_search.best_params_}")

    # Build voting classifier
    voting = VotingClassifier(
        estimators=[('svm', best_svm), ('knn', best_knn)],
        voting='soft',
    )
    voting.fit(X_train, y_train)

    combined_params = {
        'svm': svm_search.best_params_,
        'knn': knn_search.best_params_,
    }
    return voting, combined_params


def _train_xgboost(X_train, y_train) -> Tuple[Any, Dict]:
    """XGBoost classifier."""
    logger.info("  Training XGBoost...")

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")
        return None, {}

    param_dist = {
        'learning_rate': uniform(0.01, 0.09),
        'n_estimators': randint(100, 701),
        'max_depth': randint(2, 11),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }

    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0,
    )
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=30, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def _train_lightgbm(X_train, y_train) -> Tuple[Any, Dict]:
    """LightGBM classifier."""
    logger.info("  Training LightGBM...")

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed — skipping.")
        return None, {}

    param_dist = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'n_estimators': randint(100, 501),
        'max_depth': randint(2, 11),
        'num_leaves': randint(20, 100),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }

    model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=30, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def _train_stacking(X_train, y_train) -> Tuple[Any, Dict]:
    """
    Stacking (Stacked Generalization):
    Base estimators: RF, SVM, KNN
    Meta-learner: LogisticRegression
    """
    logger.info("  Training Stacking (RF + SVM + KNN → LogReg meta)...")

    base_rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
    )
    base_svm = SVC(
        kernel='rbf', probability=True, random_state=RANDOM_STATE,
    )
    base_knn = KNeighborsClassifier(
        n_neighbors=5, weights='distance',
    )

    stacking = StackingClassifier(
        estimators=[
            ('rf', base_rf),
            ('svm', base_svm),
            ('knn', base_knn),
        ],
        final_estimator=QuadraticDiscriminantAnalysis(reg_param=0.1),
        cv=CV_FOLDS,
        stack_method='predict_proba',
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    params = {
        'base_estimators': ['RF(100)', 'SVM(rbf)', 'KNN(5,distance)'],
        'meta_learner': 'QuadraticDiscriminantAnalysis',
        'cv': CV_FOLDS,
    }
    logger.info(f"    Stacking trained with 3 base estimators + QDA meta-learner")

    return stacking, params


def _train_bagging(X_train, y_train) -> Tuple[Any, Dict]:
    """
    Bagging (Bootstrap Aggregating):
    Base estimator: DecisionTree
    """
    logger.info("  Training Bagging (DecisionTree base)...")

    param_dist = {
        'n_estimators': list(range(10, 201, 10)),
        'max_samples': uniform(0.5, 0.5),          # 0.5-1.0
        'max_features': uniform(0.5, 0.5),          # 0.5-1.0
        'bootstrap': [True],
        'bootstrap_features': [True, False],
    }

    base_dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model = BaggingClassifier(
        estimator=base_dt,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=30, cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1, scoring=SCORING,
    )
    search.fit(X_train, y_train)

    logger.info(f"    Best params: {search.best_params_}")
    logger.info(f"    Best CV score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_


def train_all_models(X_train, y_train) -> Dict[str, Any]:
    """
    Train all 8 ensemble models.

    Returns:
        Dict mapping model name → fitted model. Models that failed import
        are excluded.
    """
    logger.info("=" * 60)
    logger.info("TRAINING ALL ENSEMBLE MODELS")
    logger.info("=" * 60)

    trainers = {
        'Random_Forest': _train_random_forest,
        'AdaBoost': _train_adaboost,
        'GBDT': _train_gbdt,
        'Voting_Soft': _train_voting,
        'XGBoost': _train_xgboost,
        'LightGBM': _train_lightgbm,
        'Stacking': _train_stacking,
        'Bagging': _train_bagging,
    }

    trained_models = {}
    best_params = {}

    for name, trainer_fn in trainers.items():
        try:
            model, params = trainer_fn(X_train, y_train)
            if model is not None:
                trained_models[name] = model
                best_params[name] = params
                logger.info(f"  ✓ {name} trained successfully")
            else:
                logger.warning(f"  ✗ {name} skipped (dependency missing)")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")

    logger.info(f"\nTotal models trained: {len(trained_models)}")
    return trained_models

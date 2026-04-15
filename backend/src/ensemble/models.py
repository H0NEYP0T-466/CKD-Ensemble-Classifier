"""
Ensemble models module implementing all 8 classifiers from the paper methodology.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and tunes all ensemble models specified in the paper."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models = {}
        self.tuning_results = {}

    def _random_forest_tuning(self, X_train, y_train) -> RandomForestClassifier:
        """Random Forest with RandomizedSearchCV."""
        param_dist = {
            'n_estimators': [50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400],
            'max_features': ['auto'],
            'bootstrap': [True, False],
            'min_samples_split': randint(2, 9),
            'min_samples_leaf': randint(1, 6),
            'max_depth': [None] + list(range(5, 21))
        }

        rf = RandomForestClassifier(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            rf, param_dist, n_iter=50, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        random_search.fit(X_train, y_train)

        logger.info(f"Random Forest best params: {random_search.best_params_}")
        logger.info(f"Random Forest best score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def _voting_tuning(self, X_train, y_train) -> VotingClassifier:
        """Soft Voting classifier with SVM and KNN."""
        # First tune SVM
        svm_param_dist = {
            'C': uniform(0.1, 10),
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5))
        }

        svm = SVC(probability=True, random_state=self.random_state)
        svm_random = RandomizedSearchCV(
            svm, svm_param_dist, n_iter=30, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        svm_random.fit(X_train, y_train)

        # Tune KNN
        knn_param_dist = {
            'n_neighbors': randint(3, 21),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }

        knn = KNeighborsClassifier()
        knn_random = RandomizedSearchCV(
            knn, knn_param_dist, n_iter=20, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        knn_random.fit(X_train, y_train)

        # Create voting classifier
        voting = VotingClassifier(
            estimators=[('svm', svm_random.best_estimator_),
                       ('knn', knn_random.best_estimator_)],
            voting='soft'
        )
        voting.fit(X_train, y_train)

        logger.info(f"Voting classifier created with SVM and KNN")

        return voting

    def _bagging_tuning(self, X_train, y_train) -> BaggingClassifier:
        """Bagging classifier with Decision Tree base estimator."""
        param_dist = {
            'n_estimators': [5, 10, 15, 20],
            'max_samples': uniform(0.5, 0.5),
            'max_features': uniform(0.5, 0.5),
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        }

        dt = DecisionTreeClassifier(random_state=self.random_state)
        bagging = BaggingClassifier(
            base_estimator=dt, random_state=self.random_state, n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            bagging, param_dist, n_iter=20, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        random_search.fit(X_train, y_train)

        logger.info(f"Bagging best params: {random_search.best_params_}")

        return random_search.best_estimator_

    def _adaboost_tuning(self, X_train, y_train) -> AdaBoostClassifier:
        """AdaBoost classifier with Decision Tree base estimator."""
        param_dist = {
            'n_estimators': [50, 60, 70, 80, 100],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
            'algorithm': ['SAMME', 'SAMME.R']
        }

        dt = DecisionTreeClassifier(random_state=self.random_state)
        adaboost = AdaBoostClassifier(
            base_estimator=dt, random_state=self.random_state
        )

        random_search = RandomizedSearchCV(
            adaboost, param_dist, n_iter=25, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        random_search.fit(X_train, y_train)

        logger.info(f"AdaBoost best params: {random_search.best_params_}")

        return random_search.best_estimator_

    def _gbdt_tuning(self, X_train, y_train) -> GradientBoostingClassifier:
        """Gradient Boosting Decision Trees tuning."""
        param_dist = {
            'n_estimators': randint(100, 501),
            'max_depth': randint(2, 9),
            'min_samples_leaf': [4, 8, 10, 16, 20],
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5)
        }

        gbdt = GradientBoostingClassifier(random_state=self.random_state)

        random_search = RandomizedSearchCV(
            gbdt, param_dist, n_iter=30, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        random_search.fit(X_train, y_train)

        logger.info(f"GBDT best params: {random_search.best_params_}")

        return random_search.best_estimator_

    def _xgboost_tuning(self, X_train, y_train) -> Any:
        """XGBoost classifier tuning."""
        try:
            import xgboost as xgb

            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': randint(100, 701),
                'max_depth': randint(2, 11),
                'min_child_weight': randint(1, 10),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5)
            }

            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )

            random_search = RandomizedSearchCV(
                xgb_model, param_dist, n_iter=30, cv=5, random_state=self.random_state,
                n_jobs=-1, scoring='accuracy'
            )
            random_search.fit(X_train, y_train)

            logger.info(f"XGBoost best params: {random_search.best_params_}")

            return random_search.best_estimator_

        except ImportError:
            logger.warning("XGBoost not available. Skipping...")
            return None

    def _lightgbm_tuning(self, X_train, y_train) -> Any:
        """LightGBM classifier tuning."""
        try:
            import lightgbm as lgb

            param_dist = {
                'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
                'n_estimators': randint(100, 501),
                'max_depth': randint(2, 11),
                'min_child_weight': [1, 3, 5, 7, 9],
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5)
            }

            lgb_model = lgb.LGBMClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )

            random_search = RandomizedSearchCV(
                lgb_model, param_dist, n_iter=30, cv=5, random_state=self.random_state,
                n_jobs=-1, scoring='accuracy'
            )
            random_search.fit(X_train, y_train)

            logger.info(f"LightGBM best params: {random_search.best_params_}")

            return random_search.best_estimator_

        except ImportError:
            logger.warning("LightGBM not available. Skipping...")
            return None

    def _stacking_tuning(self, X_train, y_train) -> StackingClassifier:
        """Stacking classifier with SVM and KNN as base, QDA as final estimator."""
        # Tune SVM
        svm_param_dist = {
            'C': uniform(0.1, 10),
            'kernel': ['linear', 'rbf', 'poly']
        }

        svm = SVC(probability=True, random_state=self.random_state)
        svm_random = RandomizedSearchCV(
            svm, svm_param_dist, n_iter=15, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        svm_random.fit(X_train, y_train)

        # Tune KNN
        knn_param_dist = {
            'n_neighbors': randint(3, 21),
            'weights': ['uniform', 'distance']
        }

        knn = KNeighborsClassifier()
        knn_random = RandomizedSearchCV(
            knn, knn_param_dist, n_iter=15, cv=5, random_state=self.random_state,
            n_jobs=-1, scoring='accuracy'
        )
        knn_random.fit(X_train, y_train)

        # Create stacking classifier
        estimators = [
            ('svm', svm_random.best_estimator_),
            ('knn', knn_random.best_estimator_)
        ]

        final_estimator = QuadraticDiscriminantAnalysis()

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
        stacking.fit(X_train, y_train)

        logger.info(f"Stacking classifier created with SVM, KNN, and QDA")

        return stacking

    def train_all_models(self, X_train, y_train) -> Dict[str, Any]:
        """Train all 8 models specified in the paper."""
        logger.info("Training all ensemble models...")

        models = {
            'Random_Forest': self._random_forest_tuning(X_train, y_train),
            'Voting_Soft': self._voting_tuning(X_train, y_train),
            'Bagging': self._bagging_tuning(X_train, y_train),
            'AdaBoost': self._adaboost_tuning(X_train, y_train),
            'GBDT': self._gbdt_tuning(X_train, y_train),
            'XGBoost': self._xgboost_tuning(X_train, y_train),
            'LightGBM': self._lightgbm_tuning(X_train, y_train),
            'Stacking': self._stacking_tuning(X_train, y_train)
        }

        # Remove None models (if any library not available)
        models = {k: v for k, v in models.items() if v is not None}

        self.trained_models = models
        return models

    def get_trained_models(self) -> Dict[str, Any]:
        return self.trained_models.copy()
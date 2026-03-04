"""
Baseline classifiers for BINN comparison.

Provides a unified BaselineWrapper around SVM-RBF, KNN, Random Forest,
and XGBoost with hyperparameter grids for inner-CV tuning via GridSearchCV.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)

# ── Hyperparameter grids ──────────────────────────────────────────────────────

SVM_RBF_PARAMS = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "class_weight": ["balanced", None],
}

KNN_PARAMS = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"],
}

RF_PARAMS = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 3, 5],
    "class_weight": ["balanced", "balanced_subsample", None],
}

XGBOOST_PARAMS = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
}

_PARAM_GRIDS = {
    "svm_rbf": SVM_RBF_PARAMS,
    "knn": KNN_PARAMS,
    "random_forest": RF_PARAMS,
    "xgboost": XGBOOST_PARAMS,
}

_MODEL_NAMES = list(_PARAM_GRIDS)


# ── BaselineWrapper ───────────────────────────────────────────────────────────

class BaselineWrapper:
    """
    Unified interface for baseline classifiers with inner-CV hyperparameter tuning.

    Parameters
    ----------
    model_name : str
        One of 'svm_rbf', 'knn', 'random_forest', 'xgboost'.
    random_state : int
        Random seed forwarded to the underlying estimator and GridSearchCV.
    """

    def __init__(self, model_name: str, random_state: int = config.RANDOM_SEED) -> None:
        if model_name not in _PARAM_GRIDS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {_MODEL_NAMES}"
            )
        self.model_name = model_name
        self.random_state = random_state
        self.best_params_: dict | None = None
        self._estimator: object | None = None  # fitted GridSearchCV or bare estimator
        self.compute_device = config.DEVICE
        self.compute_device_id = config.DEVICE_IDENTIFIER
        self.runtime_xgb_device: str | None = None

    # ── Base estimator (unfitted) ─────────────────────────────────────────────

    def _make_base_estimator(self, xgb_device: str | None = None):
        """Return a fresh, unfitted sklearn/XGBoost estimator."""
        if self.model_name == "svm_rbf":
            return SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
            )
        elif self.model_name == "knn":
            return KNeighborsClassifier()
        elif self.model_name == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError(
                    "xgboost is not installed. Run: uv add xgboost"
                )
            if xgb_device is None:
                xgb_device = config.XGBOOST_DEVICE
            return XGBClassifier(
                tree_method="hist",
                device=xgb_device,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

    def _build_grid(self, base, inner_cv: int) -> GridSearchCV:
        """Construct a GridSearchCV instance for the provided base estimator."""
        return GridSearchCV(
            estimator=base,
            param_grid=self.get_param_grid(),
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,       # re-fits on full X_train with best params
            n_jobs=-1,
            verbose=0,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def get_param_grid(self) -> dict:
        """Return the hyperparameter grid for this model."""
        return _PARAM_GRIDS[self.model_name]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        inner_cv: int = config.INNER_FOLDS,
    ) -> "BaselineWrapper":
        """
        Tune hyperparameters via GridSearchCV (inner CV) then refit on full
        training data with the best parameters.

        Parameters
        ----------
        X_train : (n_samples, n_features)
        y_train : (n_samples,) binary int array
        inner_cv : number of inner CV folds (default config.INNER_FOLDS)
        """
        xgb_device = config.XGBOOST_DEVICE if self.model_name == "xgboost" else None
        base = self._make_base_estimator(xgb_device=xgb_device)
        if self.model_name == "xgboost":
            self.runtime_xgb_device = xgb_device
            log.info(
                f"{self.model_name}: runtime device={xgb_device} "
                f"(primary={self.compute_device_id})"
            )
        else:
            log.info(
                f"{self.model_name}: sklearn CPU backend "
                f"(primary={self.compute_device_id})"
            )
        grid = self._build_grid(base, inner_cv)
        try:
            grid.fit(X_train, y_train)
        except Exception as exc:
            if self.model_name == "xgboost" and xgb_device == "cuda":
                log.warning(
                    "xgboost CUDA training failed (%s). Retrying on CPU.",
                    exc,
                )
                xgb_device = "cpu"
                self.runtime_xgb_device = xgb_device
                base = self._make_base_estimator(xgb_device=xgb_device)
                grid = self._build_grid(base, inner_cv)
                grid.fit(X_train, y_train)
            else:
                raise
        self._estimator = grid
        self.best_params_ = grid.best_params_
        log.info(
            f"{self.model_name}: best params={self.best_params_}  "
            f"inner_CV_AUC={grid.best_score_:.4f}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary class predictions (0 / 1)."""
        if self._estimator is None:
            raise RuntimeError("Call fit() before predict().")
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, shape (n_samples,)."""
        if self._estimator is None:
            raise RuntimeError("Call fit() before predict_proba().")
        proba = self._estimator.predict_proba(X)
        # predict_proba returns (n_samples, n_classes); take positive-class col
        return proba[:, 1].astype(np.float32)

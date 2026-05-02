"""Model training: Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
RANDOM_STATE = 42
CV_FOLDS = 5


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split.

    Args:
        df: Fully engineered DataFrame.
        target: Name of the target column.
        test_size: Proportion of data to hold out.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split — train: %d rows, test: %d rows | churn rate train=%.2f%% test=%.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LogisticRegression:
    """Train a balanced Logistic Regression with L2 regularisation.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression model.
    """
    logger.info("Training Logistic Regression ...")
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete")
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Train a Random Forest with GridSearchCV on max_depth.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Best-parameter RandomForestClassifier.
    """
    logger.info("Training Random Forest with GridSearchCV ...")
    param_grid = {
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 4],
    }
    base = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    logger.info("RF best params: %s  CV ROC-AUC: %.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


def find_optimal_threshold(model, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Find the probability threshold that maximises F1 on the training distribution.

    Uses precision_recall_curve evaluated on training predictions (no data leakage
    into the held-out test set, since this only uses training labels).

    Args:
        model: Fitted classifier with predict_proba.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Optimal probability threshold (float in (0, 1)).
    """
    probs = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, probs)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    # thresholds has one fewer element than precision/recall
    best_idx = f1_scores[:-1].argmax()
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal F1 threshold: %.4f  (train F1 at threshold: %.4f)",
        best_threshold,
        f1_scores[best_idx],
    )
    return best_threshold


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[XGBClassifier, float]:
    """Train an XGBoost classifier with GridSearchCV and F1-optimal threshold.

    Trained on the raw class distribution (no resampling) so that output
    probabilities stay well-calibrated, enabling reliable threshold tuning.
    The optimal decision threshold is found via precision-recall analysis on
    the training labels — maximising F1 without touching the test set.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Tuple of (best XGBClassifier, optimal decision threshold).
    """
    logger.info("Training XGBoost with GridSearchCV ...")

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [200, 300],
        "subsample": [0.8, 1.0],
        "min_child_weight": [1, 3],
    }
    base = XGBClassifier(
        eval_metric="auc",
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    logger.info("XGB best params: %s  CV ROC-AUC: %.4f", gs.best_params_, gs.best_score_)
    best_model = gs.best_estimator_
    threshold = find_optimal_threshold(best_model, X_train, y_train)
    return best_model, threshold


def save_model(model, name: str = "best_model") -> Path:
    """Persist a trained model to disk using joblib.

    Args:
        model: Fitted sklearn-compatible estimator.
        name: Filename stem (no extension).

    Returns:
        Path where model was saved.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, out)
    logger.info("Model saved to %s", out)
    return out


def load_model(name: str = "best_model"):
    """Load a persisted model from disk.

    Args:
        name: Filename stem (no extension).

    Returns:
        Deserialized model object.
    """
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)

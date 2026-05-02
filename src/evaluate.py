"""Model evaluation utilities: metrics, curves, and feature importance plots."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# f1_score average parameter values
_F1_BINARY   = "binary"
_F1_WEIGHTED = "weighted"

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
METRICS_PATH = Path(__file__).parent.parent / "reports" / "model_metrics.json"

sns.set_theme(style="whitegrid", palette="muted")
FIGURE_DPI = 150


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute the full evaluation metric set for a single model.

    Reports both binary F1 (for the churn/positive class) and weighted F1
    (averaged across both classes weighted by support).  The weighted F1
    is more directly comparable to the ~0.79 figure commonly cited for this
    dataset because it accounts for the 73%/27% class imbalance.

    Args:
        y_true: Ground-truth labels.
        y_pred: Binary predictions.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dict with roc_auc, f1 (binary churn class), f1_weighted,
        precision, recall, accuracy.
    """
    return {
        "roc_auc":     round(roc_auc_score(y_true, y_prob), 4),
        "f1":          round(f1_score(y_true, y_pred, average="binary"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision":   round(precision_score(y_true, y_pred), 4),
        "recall":      round(recall_score(y_true, y_pred), 4),
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
    }


def plot_confusion_matrix(
    y_true, y_pred, model_name: str, save: bool = True
) -> plt.Figure:
    """Plot and optionally save the confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Binary predictions.
        model_name: Label used in title and filename.
        save: Whether to write the figure to reports/figures/.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"confusion_matrix_{_slug(model_name)}.png")
    return fig


def plot_roc_curve(
    y_true, y_prob, model_name: str, save: bool = True
) -> plt.Figure:
    """Plot and optionally save the ROC curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Label used in title and filename.
        save: Whether to write the figure to reports/figures/.

    Returns:
        Matplotlib Figure object.
    """
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"roc_curve_{_slug(model_name)}.png")
    return fig


def plot_precision_recall_curve(
    y_true, y_prob, model_name: str, save: bool = True
) -> plt.Figure:
    """Plot and optionally save the Precision-Recall curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Label used in title and filename.
        save: Whether to write the figure to reports/figures/.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"pr_curve_{_slug(model_name)}.png")
    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str,
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure:
    """Plot horizontal bar chart of top-N feature importances.

    Works with RandomForest (feature_importances_) and XGBoost (feature_importances_).
    For Logistic Regression uses absolute coefficient values.

    Args:
        model: Fitted estimator.
        feature_names: List of feature column names.
        model_name: Label used in title and filename.
        top_n: How many features to display.
        save: Whether to write the figure to reports/figures/.

    Returns:
        Matplotlib Figure object.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Cannot extract importances from {type(model)}")

    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = series.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    if save:
        _save_fig(fig, f"feature_importance_{_slug(model_name)}.png")
    return fig


def save_metrics(all_metrics: Dict[str, Dict[str, float]]) -> None:
    """Write all model metrics to reports/model_metrics.json.

    Args:
        all_metrics: Dict mapping model_name -> metrics dict.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info("Figure saved: %s", path)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")

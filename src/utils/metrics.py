"""Evaluation metrics: accuracy, precision, recall, F1, AUC, sensitivity, specificity."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


# ---------------------------------------------------------------------------
# AUC (one-vs-rest)
# ---------------------------------------------------------------------------
def compute_auc(y_true, y_probs, average: str = "weighted") -> float:
    """Compute multi-class AUC using one-vs-rest strategy.

    Args:
        y_true: integer labels, shape ``(N,)``.
        y_probs: softmax probabilities, shape ``(N, C)``.
    """
    try:
        return roc_auc_score(y_true, y_probs, multi_class="ovr", average=average)
    except ValueError:
        # Can happen if a class is missing from the batch
        return 0.0


# ---------------------------------------------------------------------------
# Per-class sensitivity & specificity
# ---------------------------------------------------------------------------
def sensitivity_specificity(y_true, y_pred, num_classes: int = None) -> dict:
    """Return per-class sensitivity (recall) and specificity.

    Returns::

        {
            "sensitivity": [s0, s1, ...],
            "specificity": [sp0, sp1, ...],
        }
    """
    cm = confusion_matrix(y_true, y_pred)
    num_classes = num_classes or cm.shape[0]

    sensitivity = []
    specificity = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity.append(sens)
        specificity.append(spec)

    return {"sensitivity": sensitivity, "specificity": specificity}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def get_classification_report(y_true, y_pred, class_names=None) -> str:
    """Return a formatted sklearn classification report."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)

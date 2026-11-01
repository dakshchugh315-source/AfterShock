"""Evaluation metrics for 4-class damage classification."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import DAMAGE_CLASSES


def damage_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(DAMAGE_CLASSES)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    # xView2-style damage score: harmonic mean of per-class F1 (one bad class drags it down)
    harmonic = 0.0 if np.any(f1 == 0) else len(f1) / np.sum(1.0 / f1)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1.mean()),
        "harmonic_f1": float(harmonic),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(DAMAGE_CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def selective_accuracy(y_true, y_pred, uncertainty, coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5)) -> list:
    """Accuracy when the model answers only its most certain fraction and sends the rest to humans."""
    order = np.argsort(np.asarray(uncertainty))
    y_true, y_pred = np.asarray(y_true)[order], np.asarray(y_pred)[order]
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        out.append({"coverage": c, "accuracy": float((y_true[:k] == y_pred[:k]).mean()),
                    "macro_f1": damage_metrics(y_true[:k], y_pred[:k])["macro_f1"], "to_human": len(order) - k})
    return out

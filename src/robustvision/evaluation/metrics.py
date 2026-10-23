from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, labels=range(10)) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "samples": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
        ),
    }


def accuracy_drop(clean: float, corrupted: float) -> float:
    return clean - corrupted


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    probs = np.asarray(probabilities)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == np.asarray(labels)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)

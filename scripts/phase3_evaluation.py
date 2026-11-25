"""
phase3_evaluation.py
====================
Separate evaluation of the two Phase 3 heads.

Classification: accuracy, macro P/R/F1, per-class P/R/F1, confusion matrix,
                class distribution (true and predicted).
Regression:     MAE, RMSE, R^2, directional agreement (all rows and, separately,
                rows whose true return is clearly non-zero).

Model selection must NOT rely on accuracy alone: use macro-F1 together with
validation loss and the regression metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from phase3_config import CLASS_NAMES, N_CLASSES


def predict_bundle(model, X: np.ndarray, batch_size: int = 256) -> Dict[str, np.ndarray]:
    """Run multi-task inference and return a flat, easy-to-evaluate bundle."""
    outputs = model.predict(X, batch_size=batch_size, verbose=0)
    cls_proba = np.asarray(outputs["classification"], dtype=np.float64)
    reg_pred = np.asarray(outputs["regression"], dtype=np.float64).reshape(-1)
    return {
        "class_probabilities": cls_proba,
        "class_predictions": cls_proba.argmax(axis=1).astype(np.int64),
        "regression_predictions": reg_pred,
    }


def evaluate_classification(
    y_true: np.ndarray,
    class_probabilities: np.ndarray,
    class_names=CLASS_NAMES,
) -> dict:
    """Evaluate the 3-class softmax head (labels 0=DOWN, 1=FLAT, 2=UP)."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = np.asarray(class_probabilities, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != N_CLASSES:
        raise ValueError(f"Expected probabilities of shape (n,{N_CLASSES}), got {proba.shape}")
    if len(y_true) != len(proba):
        raise ValueError("y_true and class_probabilities have different lengths")

    y_pred = proba.argmax(axis=1)
    labels = list(range(N_CLASSES))
    names = list(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(N_CLASSES)
        },
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": names,
        "class_distribution": {
            "true": {names[i]: int((y_true == i).sum()) for i in range(N_CLASSES)},
            "predicted": {names[i]: int((y_pred == i).sum()) for i in range(N_CLASSES)},
        },
        "n_samples": int(len(y_true)),
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nonzero_threshold_pct: float = 0.10,
) -> dict:
    """Evaluate the linear return head. Returns are in percentage points."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty regression arrays")

    r2 = float(r2_score(y_true, y_pred)) if np.ptp(y_true) > 0 else None
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    directional = float(np.mean(sign_true == sign_pred))
    nonzero = np.abs(y_true) >= nonzero_threshold_pct
    directional_nonzero = (
        float(np.mean(sign_true[nonzero] == sign_pred[nonzero]))
        if nonzero.any() else None
    )

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "directional_agreement": directional,
        "directional_agreement_nonzero": directional_nonzero,
        "nonzero_threshold_pct": float(nonzero_threshold_pct),
        "n_samples": int(len(y_true)),
        "mean_true_return_pct": float(y_true.mean()),
        "mean_predicted_return_pct": float(y_pred.mean()),
    }


def evaluate_model(model, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray) -> dict:
    """Evaluate both heads on one (already held-out) partition."""
    bundle = predict_bundle(model, X)
    return {
        "classification": evaluate_classification(
            y_cls, bundle["class_probabilities"]
        ),
        "regression": evaluate_regression(
            y_reg, bundle["regression_predictions"]
        ),
    }

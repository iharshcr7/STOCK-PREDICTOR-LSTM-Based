"""
scaler_compat.py
================
Phase 5.2.1 — validate the ORIGINAL fitted MinMaxScaler without modifying it.

The artifact was pickled with scikit-learn 1.7.2. Newer scikit-learn versions
emit an ``InconsistentVersionWarning`` but ``transform()`` remains compatible
for MinMaxScaler. This module verifies that behaviour numerically instead of
silently ignoring the warning, and never re-pickles or refits the scaler.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALER_FILE = PROJECT_ROOT / "models" / "lstm_scaler.pkl"
EXPECTED_PICKLE_VERSION = "1.7.2"

# Deterministic probe points derived from the scaler's own fitted statistics,
# so the test works without any market data.
def probe_matrix(scaler, n: int = 6):
    low = np.asarray(scaler.data_min_, dtype=np.float64)
    high = np.asarray(scaler.data_max_, dtype=np.float64)
    span = np.where(high == low, 1.0, high - low)
    fractions = np.linspace(0.0, 1.0, n)
    return np.array([low + f * span for f in fractions], dtype=np.float64)


def check_scaler() -> dict:
    """Load and numerically validate the original scaler."""
    result = {
        "scaler_path": str(SCALER_FILE),
        "exists": SCALER_FILE.exists(),
        "runtime_sklearn_version": None,
        "warning": None,
        "scaler_type": None,
        "n_features": None,
        "feature_range": None,
        "transform_consistent": False,
        "max_abs_error": None,
        "error": None,
    }
    if not result["exists"]:
        result["error"] = "Original scaler artifact is missing"
        return result

    import sklearn
    result["runtime_sklearn_version"] = sklearn.__version__

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            scaler = joblib.load(SCALER_FILE)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"Could not load scaler: {exc}"
            return result

    version_warnings = [
        str(w.message) for w in caught
        if "version" in str(w.message).lower()
    ]
    result["warning"] = version_warnings[0] if version_warnings else None
    result["scaler_type"] = type(scaler).__name__
    result["n_features"] = int(getattr(scaler, "n_features_in_", 0))
    result["feature_range"] = tuple(getattr(scaler, "feature_range", ()))

    try:
        probe = probe_matrix(scaler)
        transformed = np.asarray(scaler.transform(probe), dtype=np.float64)
        low = np.asarray(scaler.data_min_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        offset = np.asarray(scaler.min_, dtype=np.float64)
        # sklearn MinMaxScaler: scale_ = range / feature_range_width;
        # transform(X) = X * scale_ + min_ (NOT (X - min) / scale).
        expected = probe * scale + offset
        min_value, max_value = result["feature_range"]
        expected = np.clip(expected, min_value, max_value)
        error = float(np.max(np.abs(transformed - expected)))
        result["max_abs_error"] = error
        result["transform_consistent"] = bool(
            error < 1e-5
            and np.isfinite(transformed).all()
            and result["n_features"] == 21
            and result["scaler_type"] == "MinMaxScaler"
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Transform validation failed: {exc}"
    return result


if __name__ == "__main__":
    print(json.dumps(check_scaler(), indent=2, default=str))

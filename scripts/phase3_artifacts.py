"""
phase3_artifacts.py
===================
Versioned Phase 3 artifact persistence.

Every filename is prefixed ``phase3_<version>_<variant>``. A hard guard rejects
any path that matches a legacy artifact name, so the existing
``lstm_model.keras`` / ``lstm_scaler.pkl`` / ``lstm_metadata.json`` /
``lstm_history.json`` can never be overwritten by this module.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib

from phase3_config import (
    ARCHITECTURE,
    CLASS_LABELS,
    CLASS_NAMES,
    FEATURED_DATA_FILE,
    INPUT_SHAPE,
    LEGACY_ARTIFACTS,
    LOSS_CONFIG,
    MODEL_FAMILY,
    N_FEATURES,
    PHASE3_VERSION,
    SEEDS,
    SEQUENCE_LENGTH,
    TRAINING_CONFIG,
    UNKNOWN_TARGET,
    artifact_paths,
    assert_no_legacy_target,
)
from phase3_model import model_signature


def _write_json(path: Path, payload: dict) -> None:
    if path.name in LEGACY_ARTIFACTS:
        raise RuntimeError(f"Refusing to write legacy artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_artifacts(
    model,
    variant: str,
    scaler,
    feature_columns: Iterable[str],
    history,
    validation_metrics: dict,
    training_metadata: dict,
    sequence_metadata=None,
) -> dict[str, str]:
    """Persist all required Phase 3 artifacts under versioned names."""
    paths = artifact_paths(variant)
    assert_no_legacy_target(paths)
    MODELS_DIR = paths["model"].parent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Keras model — versioned name only.
    model.save(str(paths["model"]))

    # Scaler (train-only fit).
    joblib.dump(scaler, paths["scaler"], compress=3)

    _write_json(paths["feature_schema"], {
        "feature_columns": list(feature_columns),
        "n_features": N_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "input_shape": list(INPUT_SHAPE),
        "order_source": "model_config.FINAL_FEATURE_COLUMNS (Phase 2 canonical)",
        "future_columns_excluded": [
            "Target", "Target_Date", "Next_Return_Pct", "Future_Return_Pct",
        ],
    })

    _write_json(paths["class_mapping"], {
        "mapping": {str(k): v for k, v in CLASS_LABELS.items()},
        "class_names": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "unknown_target": UNKNOWN_TARGET,
        "label_mode": "three_class",
        "classification_output": "Dense(3, softmax) probability per class",
        "regression_output": "Dense(1, linear) next-session percentage return",
        "target_definition": {
            "row_t": "return from t to t+1 = (Close[t+1]/Close[t]-1)*100",
            "DOWN": "return < -1.0%",
            "FLAT": "-1.0% <= return <= +1.0%",
            "UP": "return > +1.0%",
        },
    })

    _write_json(paths["model_config"], {
        "version": PHASE3_VERSION,
        "variant": variant,
        "model_family": MODEL_FAMILY,
        "architecture": ARCHITECTURE,
        "loss": LOSS_CONFIG,
        "training_controls": TRAINING_CONFIG,
        "seeds": SEEDS,
        "input_shape": list(INPUT_SHAPE),
        "output_shapes": {
            "classification": [None, len(CLASS_NAMES)],
            "regression": [None, 1],
        },
    })

    _write_json(paths["training_history"], {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    })

    _write_json(paths["validation_metrics"], {
        "partition": "validation",
        "note": (
            "Test partition was NOT used for model selection or tuning and "
            "must only be evaluated once after explicit approval."
        ),
        "metrics": validation_metrics,
    })

    signature = model_signature(model)
    _write_json(paths["training_metadata"], {
        "version": PHASE3_VERSION,
        "variant": variant,
        "trained_at": datetime.now().isoformat(),
        "source_data": str(FEATURED_DATA_FILE),
        "model_signature": signature,
        **training_metadata,
        "artifact_paths": {k: str(v) for k, v in paths.items()},
        "legacy_artifacts_preserved": list(LEGACY_ARTIFACTS),
    })

    if sequence_metadata is not None:
        sequence_metadata.to_csv(paths["sequence_metadata"], index=False)

    return {k: str(v) for k, v in paths.items()}


def load_artifact_bundle(variant: str = "baseline") -> dict:
    """Load a versioned Phase 3 model plus its schema/config companions."""
    paths = artifact_paths(variant)
    assert_no_legacy_target(paths)
    from tensorflow import keras

    model = keras.models.load_model(str(paths["model"]))
    scaler = joblib.load(paths["scaler"])

    def _json(key):
        with open(paths[key], "r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "model": model,
        "scaler": scaler,
        "feature_schema": _json("feature_schema"),
        "class_mapping": _json("class_mapping"),
        "model_config": _json("model_config"),
        "training_metadata": _json("training_metadata"),
        "validation_metrics": _json("validation_metrics"),
    }

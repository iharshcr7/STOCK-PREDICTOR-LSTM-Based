"""
phase3_config.py
================
Configuration for the Phase 3 multi-stock, multi-task LSTM.

This module deliberately does NOT change the Phase 2 data contract. It only
adds a versioned artifact namespace so that no legacy checkpoint is touched:

    models/lstm_model.keras          (legacy — never written by Phase 3)
    models/lstm_scaler.pkl           (legacy — never written by Phase 3)
    models/lstm_metadata.json        (legacy — never written by Phase 3)
    models/lstm_history.json         (legacy — never written by Phase 3)

All Phase 3 outputs are prefixed with ``phase3_<version>_<variant>``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FEATURED_DATA_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"

# ---------------------------------------------------------------------------
# Version / identity
# ---------------------------------------------------------------------------
PHASE3_VERSION = "v3.0.0"
MODEL_FAMILY = "multitask_lstm"

# Column names (identical to the Phase 2 canonical contract).
SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"
CLASS_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
N_CLASSES = 3
UNKNOWN_TARGET = -1

# Legacy artifact names — Phase 3 must never write to these paths.
LEGACY_ARTIFACTS = (
    "lstm_model.keras",
    "lstm_scaler.pkl",
    "lstm_metadata.json",
    "lstm_history.json",
)

# ---------------------------------------------------------------------------
# Input contract (must equal the Phase 2 pipeline)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 60
N_FEATURES = 21
INPUT_SHAPE = (SEQUENCE_LENGTH, N_FEATURES)

# ---------------------------------------------------------------------------
# Architecture — intentionally lightweight for CPU training
# ---------------------------------------------------------------------------
ARCHITECTURE = {
    "lstm_1_units": 64,
    "lstm_2_units": 32,          # baseline only; attention replaces this
    "shared_dense_units": 32,
    "dropout_rate": 0.2,
    "l2_reg": 1e-4,
    "attention_score_units": 8,  # attention variant only (additive scoring)
    "huber_delta": 1.0,
}

# ---------------------------------------------------------------------------
# Losses / optimizer
# ---------------------------------------------------------------------------
# Sparse categorical cross-entropy is the primary objective. Huber regression
# is deliberately down-weighted: percentage returns are unbounded and a few
# extreme sessions would otherwise dominate the shared encoder. 1.0 : 0.25
# keeps the 3-class softmax head dominant while still learning return size.
LOSS_CONFIG = {
    "classification": "sparse_categorical_crossentropy",
    "regression": "huber",
    "loss_weights": {"classification": 1.0, "regression": 0.25},
    "optimizer": "adam",
    "learning_rate": 1e-3,
}

# ---------------------------------------------------------------------------
# Training controls (proposed defaults — require explicit approval to run)
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "batch_size": 64,
    "max_epochs": 20,
    "early_stopping_patience": 6,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "monitor": "val_loss",
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "label_mode": "three_class",
    "smoke_batch_size": 16,
    "smoke_epochs": 2,
}

SEEDS = {
    "python": 42,
    "numpy": 42,
    "tensorflow": 42,
}


# ---------------------------------------------------------------------------
# Versioned artifact paths
# ---------------------------------------------------------------------------
def artifact_paths(variant: str = "baseline") -> dict[str, Path]:
    """Return all Phase 3 output paths for one model variant.

    ``variant`` must be 'baseline' or 'attention'. Every path is unique to
    Phase 3, so the four legacy LSTM artifacts can never be overwritten.
    """
    if variant not in ("baseline", "attention"):
        raise ValueError("variant must be 'baseline' or 'attention'")
    prefix = f"phase3_{PHASE3_VERSION}_{variant}"
    return {
        "model": MODELS_DIR / f"{prefix}.keras",
        "scaler": MODELS_DIR / f"{prefix}_scaler.pkl",
        "feature_schema": MODELS_DIR / f"{prefix}_feature_schema.json",
        "class_mapping": MODELS_DIR / f"{prefix}_class_mapping.json",
        "model_config": MODELS_DIR / f"{prefix}_config.json",
        "training_history": MODELS_DIR / f"{prefix}_history.json",
        "validation_metrics": MODELS_DIR / f"{prefix}_validation_metrics.json",
        "training_metadata": MODELS_DIR / f"{prefix}_training_metadata.json",
        "sequence_metadata": MODELS_DIR / f"{prefix}_sequence_metadata.csv",
    }


def assert_no_legacy_target(paths: dict[str, Path]) -> None:
    """Hard guard: refuse to write anywhere near a legacy artifact."""
    for key, path in paths.items():
        if path.name in LEGACY_ARTIFACTS:
            raise RuntimeError(
                f"Refusing to overwrite legacy artifact {path.name} (key={key})"
            )

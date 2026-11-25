"""
train_lstm_model.py
===================
LSTM model training for stock movement prediction using time-series sequences.

Unlike Random Forest (which uses a single-row snapshot of 21 features),
LSTM learns from a SEQUENCE of the last SEQUENCE_LENGTH days to capture
temporal patterns, trends, and momentum over time.

Architecture:
    Input: (batch, 60, 21) — 60 days × 21 features per day
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, sigmoid)
    Output: probability of UP (>0.5 = UP, <=0.5 = DOWN) — binary checkpoint

Data contract (Phase 2): all symbols are processed independently, targets are
3-class (0=DOWN, 1=FLAT, 2=UP) for the session after the final input date,
sequences are per-symbol/per-block, the split is per-symbol chronological
70/15/15 by target date, and the scaler is fitted on training rows only.
The current binary head cannot emit FLAT; LABEL_MODE="legacy_binary" keeps it
usable until Phase 3 retrains with a 3-class head.

Usage:
    python scripts/train_lstm_model.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from lstm_data_pipeline import (
    FEATURE_COLUMNS as CANONICAL_FEATURES,
    SEQUENCE_LENGTH as CANONICAL_SEQUENCE_LENGTH,
    build_labeled_sequences,
    chronological_partition,
    fit_scaler_on_train,
    split_frames_by_symbol,
    transform_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras"
LSTM_SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"
LSTM_HISTORY_FILE = MODELS_DIR / "lstm_history.json"   # training curves for dashboard

# Single canonical feature schema (names AND order) shared with live inference.
FEATURE_COLUMNS = list(CANONICAL_FEATURES)

TARGET_COLUMN = "Target"
SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"

# No hard-coded symbol list: every symbol in the dataset is processed
# independently through symbol_mapper/dataset discovery upstream.
SEQUENCE_LENGTH = CANONICAL_SEQUENCE_LENGTH  # 60 past observations
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# The stored checkpoint has a binary sigmoid head. "legacy_binary" keeps that
# head usable (FLAT rows remain in the input series but are not emitted as
# labelled samples). "three_class" emits 0=DOWN/1=FLAT/2=UP and requires the
# Phase 3 softmax head + retraining. Do not switch while using the old model.
LABEL_MODE = "legacy_binary"

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        logger.error("Run scripts/feature_engineering.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded dataset: %d rows × %d columns", len(df), len(df.columns))
    return df


def filter_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns: %s", sorted(missing))
        sys.exit(1)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    # Dynamic, all-symbol processing. Sorting is per symbol in the next step.
    df = df.sort_values([SYMBOL_COLUMN, DATE_COLUMN]).reset_index(drop=True)

    # Remove NaN / Inf in model inputs only. Target == -1 (UNKNOWN, final row
    # of a symbol) is intentionally retained as input history.
    invalid = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    df = df[~invalid].reset_index(drop=True)

    logger.info(
        "Validated dataset: %d rows | symbols=%d | %s → %s",
        len(df),
        df[SYMBOL_COLUMN].nunique(),
        df[DATE_COLUMN].min().date(),
        df[DATE_COLUMN].max().date(),
    )
    return df


def build_sequences(df: pd.DataFrame) -> tuple:
    """
    Build (X, y, metadata) using the canonical per-symbol/per-block builder.

    A sequence ending at observation t uses rows [t-59 ... t] of ONE symbol
    and its label is Target[t] — the return from t to t+1. No cross-symbol
    windows, no cross-gap windows, no off-by-one target.
    """
    frames = split_frames_by_symbol(df)
    X, y, meta = build_labeled_sequences(frames, label_mode=LABEL_MODE)
    logger.info("Total sequences: X=%s  y=%s  symbols=%d", X.shape, y.shape,
                meta[SYMBOL_COLUMN].nunique() if not meta.empty else 0)
    return X, y, meta


def chronological_split(meta: pd.DataFrame):
    """
    Leakage-safe 70/15/15 split per symbol, by each sequence's target date.
    Returns (train_idx, val_idx, test_idx, boundary_report).
    """
    tr, va, te, report = chronological_partition(meta, TRAIN_FRAC, VAL_FRAC)
    logger.info("Split: train=%d  val=%d  test=%d", len(tr), len(va), len(te))
    if not report.empty and "status" in report.columns:
        excluded = report[report["status"] != "included"]
        if not excluded.empty:
            logger.warning("Excluded %d short-history symbols from partitioning", len(excluded))
    return tr, va, te, report


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------

def build_lstm_model(n_features: int, seq_len: int):
    """Build and compile the LSTM model."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow==2.15.0")
        sys.exit(1)

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),

        # First LSTM layer — returns sequences for stacking
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),

        # Second LSTM layer — returns final hidden state
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),

        # Dense head
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # Binary: UP probability
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val):
    """Train with early stopping and learning rate reduction."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        logger.error("TensorFlow not installed.")
        sys.exit(1)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(LSTM_MODEL_FILE),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Training LSTM: epochs=%d  batch=%d  seq_len=%d", EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
    )

    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["DOWN", "UP"]
        ),
    }

    confidence = np.maximum(y_proba, 1.0 - y_proba)
    metrics["prediction_confidence"] = {
        "mean":   float(np.mean(confidence)),
        "median": float(np.median(confidence)),
        "p90":    float(np.percentile(confidence, 90)),
    }

    logger.info("=" * 60)
    logger.info("  LSTM EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info("  Accuracy:  %.4f (%.2f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1-Score:  %.4f", metrics["f1_score"])
    logger.info("  ROC AUC:   %.4f", metrics["roc_auc"])
    logger.info("  Classification Report:\n%s", metrics["classification_report"])
    logger.info("=" * 60)

    return metrics


def save_artifacts(model, scaler, metrics: dict, history, split_report=None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model already saved by ModelCheckpoint callback — just log size
    if LSTM_MODEL_FILE.exists():
        size_mb = LSTM_MODEL_FILE.stat().st_size / (1024 * 1024)
        logger.info("LSTM model saved → %s (%.2f MB)", LSTM_MODEL_FILE, size_mb)

    # Save scaler
    joblib.dump(scaler, LSTM_SCALER_FILE, compress=3)
    logger.info("Scaler saved → %s", LSTM_SCALER_FILE)

    # Save training history (for dashboard graphs)
    history_data = {
        "accuracy":     [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in history.history.get("loss", [])],
        "val_loss":     [float(v) for v in history.history.get("val_loss", [])],
    }
    with open(LSTM_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    logger.info("Training history saved → %s", LSTM_HISTORY_FILE)

    # Save metadata
    val_acc = history.history.get("val_accuracy", [])
    val_loss = history.history.get("val_loss", [])

    metadata = {
        "model_type": "LSTM",
        "model_file": str(LSTM_MODEL_FILE),
        "scaler_file": str(LSTM_SCALER_FILE),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "symbols": "dynamic (all eligible symbols from the dataset; no fixed list)",
        "sequence_length": SEQUENCE_LENGTH,
        "target_convention": {
            "row_t_target": "return from t to next trading session t+1",
            "classes": {"0": "DOWN", "1": "FLAT", "2": "UP"},
            "threshold_pct": 1.0,
            "unknown_label": -1,
            "flat_rows_retained_in_input_series": True,
        },
        "label_mode_used": LABEL_MODE,
        "architecture": {
            "layers": [
                "LSTM(128, return_sequences=True)",
                "Dropout(0.2)",
                "LSTM(64)",
                "Dropout(0.2)",
                "Dense(32, relu)",
                "Dense(1, sigmoid)",
            ],
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "binary_crossentropy",
        },
        "training": {
            "epochs_configured": EPOCHS,
            "epochs_trained": len(val_loss),
            "batch_size": BATCH_SIZE,
            "split_strategy": "per_symbol_chronological_70_15_15_by_target_date",
            "scaler_fit": "training partition only; applied unchanged to val/test",
            "partition_report": (
                split_report.to_dict(orient="records")
                if split_report is not None and not split_report.empty else []
            ),
            "early_stopping": True,
        },
        "metrics": {
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1_score":  metrics["f1_score"],
            "roc_auc":   metrics["roc_auc"],
            "prediction_confidence": metrics["prediction_confidence"],
            "best_val_accuracy": float(max(val_acc)) if val_acc else None,
            "best_val_loss":     float(min(val_loss)) if val_loss else None,
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "prediction_config": {
            "labels": {"0": "DOWN", "1": "UP"},
            "threshold": 0.5,
            "confidence_source": "sigmoid_output_probability",
        },
        "deployment_notes": {
            "streamlit_compatible": True,
            "live_mode_only": True,
            "reason": (
                "LSTM requires a sequence of SEQUENCE_LENGTH past days. "
                "Manual mode provides only a single snapshot, so LSTM is "
                "only available in Live Data mode."
            ),
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(LSTM_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", LSTM_METADATA_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("LSTM TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load & filter data
    df = load_dataset()
    df = filter_and_validate(df)

    # 2. Build per-symbol sequences BEFORE scaling (no scaler leakage).
    logger.info("Building sequences (seq_len=%d, label_mode=%s)...",
                SEQUENCE_LENGTH, LABEL_MODE)
    X, y, meta = build_sequences(df)

    # 3. Leakage-safe per-symbol chronological split by target date.
    tr_idx, va_idx, te_idx, split_report = chronological_split(meta)

    # 4. Fit preprocessing on TRAIN sequences only; apply unchanged elsewhere.
    scaler = fit_scaler_on_train(X[tr_idx])
    logger.info("MinMaxScaler fitted on %d training rows only", len(tr_idx) * SEQUENCE_LENGTH)

    X_train = transform_sequences(scaler, X[tr_idx])
    X_val = transform_sequences(scaler, X[va_idx])
    X_test = transform_sequences(scaler, X[te_idx])
    y_train, y_val, y_test = y[tr_idx], y[va_idx], y[te_idx]
    logger.info("Train=%d  Validation=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "One or more partitions are empty. Add historical data or relax "
            "the short-history exclusion policy; never train on an empty split."
        )

    up_train = int(y_train.sum())
    down_train = int(len(y_train) - up_train)
    logger.info("Train target: UP=%d  DOWN/non-up=%d", up_train, down_train)

    # 5. Build model (architecture unchanged).
    n_features = len(FEATURE_COLUMNS)
    model = build_lstm_model(n_features, SEQUENCE_LENGTH)

    # 6. Train (early stopping on validation, never on test).
    history = train(model, X_train, y_train, X_val, y_val)

    # 7. Final evaluation on the untouched test partition.
    metrics = evaluate(model, X_test, y_test)

    # 8. Save artifacts (existing paths; no old checkpoints overwritten here).
    save_artifacts(model, scaler, metrics, history, split_report=split_report)

    logger.info("=" * 60)
    logger.info("LSTM TRAINING COMPLETE")
    logger.info("  Model:    %s", LSTM_MODEL_FILE)
    logger.info("  Scaler:   %s", LSTM_SCALER_FILE)
    logger.info("  Metadata: %s", LSTM_METADATA_FILE)
    logger.info("=" * 60)
    logger.info("Next step: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("LSTM training failed: %s", exc)
        sys.exit(1)

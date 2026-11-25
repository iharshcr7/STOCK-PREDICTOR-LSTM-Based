"""
phase3_data.py
==============
Memory-conscious dataset assembly for Phase 3.

The featured CSV is streamed in chunks and only ONE symbol's frame is prepared
at a time; raw histories for all symbols are never resident simultaneously.
All feature/target/sequence definitions still come from the Phase 2 canonical
module — this file only orchestrates them and adds the regression target
interface used by the multi-task head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from lstm_data_pipeline import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    build_labeled_sequences,
    chronological_partition,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from phase3_config import (
    CLASS_LABELS,
    DATE_COLUMN,
    N_FEATURES,
    SEQUENCE_LENGTH as CONFIG_SEQUENCE_LENGTH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    TRAINING_CONFIG,
    UNKNOWN_TARGET,
)

_REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | {
    SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN,
    "Target_Date", "Next_Return_Pct",
}


# ---------------------------------------------------------------------------
# Streaming / per-symbol preparation
# ---------------------------------------------------------------------------
def validate_featured_dataset(path: Path) -> dict:
    """Cheap header/level validation without loading the whole file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Featured dataset not found: {path}")
    header = pd.read_csv(path, nrows=0)
    missing = _REQUIRED_COLUMNS - set(header.columns)
    if missing:
        raise ValueError(
            f"Featured dataset missing columns: {sorted(missing)}. "
            "Regenerate it with scripts/feature_engineering.py (Phase 2)."
        )
    if len(FEATURE_COLUMNS) != N_FEATURES or CONFIG_SEQUENCE_LENGTH != SEQUENCE_LENGTH:
        raise ValueError("Phase 3 config disagrees with the Phase 2 data contract")
    return {"columns": list(header.columns), "n_features": len(FEATURE_COLUMNS)}


def iter_symbol_frames(path: Path, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    """Yield one prepared symbol frame at a time, in symbol-sorted file order.

    The featured CSV is written symbol-sorted by feature_engineering.py. A
    non-contiguous symbol reappearance is treated as a hard error so no symbol
    can be split into two independent frames (which would duplicate sequences).
    """
    emitted: set[str] = set()
    pending: pd.DataFrame | None = None
    pending_symbol: str | None = None

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk[DATE_COLUMN] = pd.to_datetime(chunk[DATE_COLUMN], errors="coerce")
        for symbol, group in chunk.groupby(SYMBOL_COLUMN, sort=False):
            symbol = str(symbol)
            if pending_symbol is not None and symbol != pending_symbol:
                if pending_symbol in emitted:
                    raise ValueError(
                        f"Symbol {pending_symbol} is non-contiguous in the featured "
                        "CSV; regenerate the file with feature_engineering.py"
                    )
                emitted.add(pending_symbol)
                yield prepare_symbol_frame(pending, symbol=pending_symbol)
                pending = None
            pending = group if pending is None else pd.concat(
                [pending, group], ignore_index=True
            )
            pending_symbol = symbol

    if pending is not None:
        if pending_symbol in emitted:
            raise ValueError(f"Symbol {pending_symbol} is non-contiguous in the CSV")
        emitted.add(pending_symbol)
        yield prepare_symbol_frame(pending, symbol=pending_symbol)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def build_phase3_dataset(
    path: Path,
    chunksize: int = 50_000,
    train_frac: float | None = None,
    val_frac: float | None = None,
):
    """Build scaled, partitioned (X, y_cls, y_reg, metadata).

    Uses only Phase 2 helpers: per-symbol sequence construction, per-symbol
    chronological partition, and a scaler fitted on TRAIN sequences only.
    """
    validate_featured_dataset(path)
    train_frac = TRAINING_CONFIG["train_frac"] if train_frac is None else train_frac
    val_frac = TRAINING_CONFIG["val_frac"] if val_frac is None else val_frac

    X_parts, y_parts, yr_parts, meta_parts = [], [], [], []
    excluded: list[dict] = []
    offset = 0
    n_symbols = 0

    for frame in iter_symbol_frames(path, chunksize=chunksize):
        n_symbols += 1
        Xi, yi, metai, yri = build_labeled_sequences(
            [frame], label_mode="three_class", return_regression=True
        )
        if len(Xi) == 0:
            excluded.append({
                SYMBOL_COLUMN: str(frame[SYMBOL_COLUMN].iloc[0]),
                "status": "excluded_no_valid_sequences",
                "reason": "fewer than 60 finite feature rows or all labels UNKNOWN",
            })
            continue
        metai.index = range(offset, offset + len(metai))
        offset += len(metai)
        X_parts.append(Xi)
        y_parts.append(yi)
        yr_parts.append(yri)
        meta_parts.append(metai)

    if not X_parts:
        raise ValueError(
            "No symbol produced a valid 3-class sequence. Regenerate the "
            "featured CSV from the Phase 2 pipeline first."
        )

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_cls = np.concatenate(y_parts).astype(np.int64)
    y_reg = np.concatenate(yr_parts).astype(np.float32)
    meta = pd.concat(meta_parts, axis=0)
    del X_parts, y_parts, yr_parts, meta_parts

    tr, va, te, report = chronological_partition(meta, train_frac, val_frac)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(
            "One or more partitions are empty; never train on an empty split."
        )

    scaler = fit_scaler_on_train(X[tr])
    X_train = transform_sequences(scaler, X[tr])
    X_val = transform_sequences(scaler, X[va])
    X_test = transform_sequences(scaler, X[te])
    del X  # free the unscaled copy before training

    class_counts = {CLASS_LABELS[i]: int((y_cls == i).sum()) for i in CLASS_LABELS}
    if class_counts["FLAT"] == 0:
        raise ValueError(
            "No FLAT labels found. The featured CSV is likely a pre-Phase-2 "
            "binary artifact. Regenerate data with scripts/feature_engineering.py."
        )
    if excluded:
        for item in excluded:
            print(f"[phase3-data] WARNING excluded: {item}")

    meta = meta.copy()
    meta["partition"] = "unassigned"
    meta.loc[tr, "partition"] = "train"
    meta.loc[va, "partition"] = "val"
    meta.loc[te, "partition"] = "test"

    return {
        "X_train": X_train, "y_train_cls": y_cls[tr], "y_train_reg": y_reg[tr],
        "X_val": X_val, "y_val_cls": y_cls[va], "y_val_reg": y_reg[va],
        "X_test": X_test, "y_test_cls": y_cls[te], "y_test_reg": y_reg[te],
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "metadata": meta, "scaler": scaler,
        "partition_report": report, "excluded_symbols": excluded,
        "n_symbols_seen": n_symbols,
        "class_counts": class_counts,
    }


def estimate_memory(n_sequences: int) -> dict:
    """Analytic CPU memory estimate for a given number of sequences."""
    bytes_per_sequence = SEQUENCE_LENGTH * N_FEATURES * np.dtype(np.float32).itemsize
    sequences_mb = n_sequences * bytes_per_sequence / (1024 ** 2)
    labels_mb = (
        n_sequences * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
        / (1024 ** 2)
    )
    return {
        "n_sequences": int(n_sequences),
        "bytes_per_sequence": int(bytes_per_sequence),
        "all_sequences_mb": round(sequences_mb, 1),
        "labels_mb": round(labels_mb, 1),
        "scaled_partitions_mb": round(sequences_mb, 1),
        "transient_peak_sequences_mb": round(sequences_mb * 2, 1),
        "note": (
            "Raw copy is deleted after scaling. Add roughly 0.5-1.0 GB for "
            "TensorFlow/Keras and Python runtime when budgeting against 8 GB RAM."
        ),
    }

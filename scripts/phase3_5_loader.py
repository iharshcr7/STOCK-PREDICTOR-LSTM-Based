"""
phase3_5_loader.py
==================
Bounded-memory, TensorFlow-compatible loader for the Phase 3 multi-task LSTM.

Design (no duplicated windows, no shards of the 10.62 GB CSV)
-------------------------------------------------------------
Pass 1 — ``calibrate_scaler``:
    Stream one symbol at a time. Derive the exact Phase 3 sequence index and
    per-symbol chronological partitions, mark the raw feature rows covered by
    TRAIN sequences, and accumulate per-feature min/max. A MinMaxScaler is then
    fitted from those two boundary rows. MinMaxScaler depends only on min/max,
    so this is exactly equivalent to ``fit_scaler_on_train`` on every training
    window (verified by tests) without ever materialising a training window.

Pass 2 — ``make_tf_dataset`` / ``iter_sequences``:
    Stream the CSV again, scale each symbol's retained feature rows once, and
    yield overlapping 60-step windows on demand. Only one symbol's rows and one
    window buffer are resident at a time. No expanded-window storage is written.

Guarantees
----------
* Same samples/labels/partitions as the validated Phase 3 pipeline
  (``lstm_data_pipeline.build_labeled_sequences`` + ``chronological_partition``).
* No cross-symbol windows; no windows across >7-calendar-day gaps.
* UNKNOWN (-1) rows stay in history but are never supervised labels.
* Train shuffling happens only on train sequences, in a bounded tf.data buffer.
* Validation/test remain chronological and unshuffled; test is never touched
  by this module's benchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from lstm_data_pipeline import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    UNKNOWN_TARGET,
    add_next_session_target,
    chronological_partition,
    compute_technical_features,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from phase3_config import N_FEATURES, TRAINING_CONFIG

REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | {
    SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN, "Target_Date", "Next_Return_Pct",
}


# ---------------------------------------------------------------------------
# Sequence index (metadata only — no expanded windows)
# ---------------------------------------------------------------------------
@dataclass
class SequenceIndex:
    """Per-symbol arrays, one entry per valid 60-step sequence."""

    ends: np.ndarray                 # frame row position of the window's last row
    labels: np.ndarray               # int64 class label (0/1/2)
    next_returns: np.ndarray         # float64 next-session percentage return
    target_dates: np.ndarray         # datetime64, exact predicted session date
    input_start_dates: np.ndarray    # datetime64
    input_end_dates: np.ndarray      # datetime64
    symbols: np.ndarray              # object array (one symbol per sequence)

    def __len__(self) -> int:
        return int(len(self.ends))


def build_sequence_index(frame: pd.DataFrame) -> SequenceIndex:
    """Vectorised metadata equivalent of ``build_labeled_sequences(three_class)``.

    A sequence ending at row ``e`` uses rows ``[e-59 .. e]`` of ONE symbol and
    ONE gap block; its label is ``Target[e]`` for session e -> e+1. Rows with
    non-finite features and UNKNOWN labels are never emitted.
    """
    df = frame.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    if "Next_Return_Pct" not in df.columns:
        df = add_next_session_target(compute_technical_features(df))

    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    targets = df[TARGET_COLUMN].to_numpy()
    next_returns = df["Next_Return_Pct"].to_numpy(dtype=np.float64)
    target_dates = pd.to_datetime(df["Target_Date"]).to_numpy()
    dates = pd.to_datetime(df[DATE_COLUMN]).to_numpy()
    blocks = df["Sequence_Block"].to_numpy()
    cum_finite = np.concatenate([[0], np.cumsum(finite.astype(np.int32))])

    ends, labels, rets, tdates, starts, symbols = [], [], [], [], [], []
    symbol = str(df[SYMBOL_COLUMN].iloc[0])

    for block_id in np.unique(blocks):
        pos = np.flatnonzero(blocks == block_id)
        pos = pos[finite[pos]]
        if len(pos) < SEQUENCE_LENGTH + 1:
            continue
        raw_ends = pos[SEQUENCE_LENGTH - 1 : len(pos) - 1]
        window_ok = (
            cum_finite[raw_ends + 1]
            - cum_finite[raw_ends + 1 - SEQUENCE_LENGTH]
        ) == SEQUENCE_LENGTH
        keep = (targets[raw_ends] != UNKNOWN_TARGET) & window_ok & np.isfinite(next_returns[raw_ends])
        use = raw_ends[keep]
        if len(use) == 0:
            continue
        ends.append(use)
        labels.append(targets[use].astype(np.int64))
        rets.append(next_returns[use])
        tdates.append(target_dates[use])
        starts.append(dates[use - SEQUENCE_LENGTH + 1])
        symbols.append(np.full(len(use), symbol))

    if not ends:
        return SequenceIndex(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            np.array([], dtype=np.float64), np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]"),
            np.array([], dtype=object),
        )

    return SequenceIndex(
        np.concatenate(ends), np.concatenate(labels), np.concatenate(rets),
        np.concatenate(tdates), np.concatenate(starts),
        pd.to_datetime(dates[np.concatenate(ends)]).to_numpy(),
        np.concatenate(symbols),
    )


def partition_positions(
    frame: pd.DataFrame,
    index: SequenceIndex,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Canonical per-symbol chronological partition; returns frame row positions."""
    train_frac = TRAINING_CONFIG["train_frac"] if train_frac is None else train_frac
    val_frac = TRAINING_CONFIG["val_frac"] if val_frac is None else val_frac
    meta = pd.DataFrame({
        SYMBOL_COLUMN: index.symbols,
        "Sequence_Block": 0,
        "Input_Start_Date": pd.to_datetime(index.input_start_dates),
        "Input_End_Date": pd.to_datetime(index.input_end_dates),
        "Target_Date": pd.to_datetime(index.target_dates),
        TARGET_COLUMN: index.labels,
    }, index=index.ends)
    tr, va, te, report = chronological_partition(meta, train_frac, val_frac)
    return {"train": tr, "val": va, "test": te}, report


# ---------------------------------------------------------------------------
# Streaming with an optional symbol whitelist
# ---------------------------------------------------------------------------
def stream_symbol_frames(
    path: Path,
    symbols: set[str] | None = None,
    chunksize: int = 50_000,
) -> Iterator[pd.DataFrame]:
    """Yield one prepared symbol frame, filtering rows before frame preparation."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    if not REQUIRED_COLUMNS.issubset(header.columns):
        raise ValueError(
            f"CSV missing required columns: {sorted(REQUIRED_COLUMNS - set(header.columns))}"
        )

    emitted: set[str] = set()
    pending: pd.DataFrame | None = None
    pending_symbol: str | None = None

    with pd.read_csv(path, chunksize=chunksize) as reader:
        for chunk in reader:
            if symbols is not None:
                chunk = chunk[chunk[SYMBOL_COLUMN].isin(symbols)]
                if chunk.empty:
                    continue
            chunk[DATE_COLUMN] = pd.to_datetime(chunk[DATE_COLUMN], errors="coerce")
            for symbol, group in chunk.groupby(SYMBOL_COLUMN, sort=False):
                symbol = str(symbol)
                if pending_symbol is not None and symbol != pending_symbol:
                    if pending_symbol in emitted:
                        raise ValueError(f"Non-contiguous symbol in CSV: {pending_symbol}")
                    emitted.add(pending_symbol)
                    yield prepare_symbol_frame(pending, symbol=pending_symbol)
                    pending = None
                pending = (
                    group if pending is None
                    else pd.concat([pending, group], ignore_index=True)
                )
                pending_symbol = symbol

    if pending is not None:
        if pending_symbol in emitted:
            raise ValueError(f"Non-contiguous symbol in CSV: {pending_symbol}")
        emitted.add(pending_symbol)
        yield prepare_symbol_frame(pending, symbol=pending_symbol)


def create_symbol_subset(
    source: Path, destination: Path, symbols: set[str], chunksize: int = 50_000
) -> dict:
    """Write a small filtered CSV (benchmark fixture) with bounded memory."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        first = True
        for chunk in pd.read_csv(source, chunksize=chunksize):
            part = chunk[chunk[SYMBOL_COLUMN].isin(symbols)]
            if part.empty:
                continue
            part.to_csv(handle, header=first, index=False)
            first = False
            rows += len(part)
    return {
        "source": str(source), "destination": str(destination),
        "symbols": sorted(symbols), "rows": rows,
        "size_mb": round(destination.stat().st_size / (1024 ** 2), 2),
    }


# ---------------------------------------------------------------------------
# Pass 1 — train-only scaler calibration
# ---------------------------------------------------------------------------
def calibrate_scaler(
    path: Path,
    symbols: set[str] | None = None,
    chunksize: int = 50_000,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> tuple[object, dict]:
    """Accumulate min/max over rows covered by TRAIN sequences only.

    ``MinMaxScaler`` is a function of min/max alone, so fitting it on the two
    boundary rows yields exactly the transform of a scaler fitted on all train
    windows — without ever creating those windows.
    """
    mins = np.full(N_FEATURES, np.inf, dtype=np.float64)
    maxs = np.full(N_FEATURES, -np.inf, dtype=np.float64)
    counts: Counter = Counter()
    excluded: list[dict] = []
    n_frames = 0

    for frame in stream_symbol_frames(path, symbols=symbols, chunksize=chunksize):
        n_frames += 1
        symbol = str(frame[SYMBOL_COLUMN].iloc[0])
        index = build_sequence_index(frame)
        parts, report = partition_positions(frame, index, train_frac, val_frac)
        counts.update({name: len(idx) for name, idx in parts.items()})
        for rec in report.to_dict(orient="records"):
            if rec.get("status") == "excluded_short_history":
                excluded.append({"symbol": symbol, "reason": "short_history"})
        if len(parts["train"]) == 0:
            continue

        ends = np.asarray(parts["train"], dtype=np.int64)
        diff = np.zeros(len(frame) + 1, dtype=np.int32)
        diff[ends - SEQUENCE_LENGTH + 1] += 1
        diff[ends + 1] -= 1
        covered = np.cumsum(diff)[:-1] > 0
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[covered]
        if len(values) == 0:
            continue
        mins = np.minimum(mins, values.min(axis=0).astype(np.float64))
        maxs = np.maximum(maxs, values.max(axis=0).astype(np.float64))

    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        raise ValueError("No finite TRAIN rows found; cannot calibrate scaler")
    scaler = fit_scaler_on_train(np.vstack([mins, maxs]))
    stats = {
        "symbols": n_frames,
        "sequence_counts": dict(counts),
        "excluded": excluded,
        "train_feature_min": mins.tolist(),
        "train_feature_max": maxs.tolist(),
        "n_train_rows_covered": int(counts.get("train", 0)),
    }
    return scaler, stats


# ---------------------------------------------------------------------------
# Pass 2 — on-demand sequence generation
# ---------------------------------------------------------------------------
def iter_sequences(
    path: Path,
    partition: str,
    scaler,
    symbols: set[str] | None = None,
    chunksize: int = 50_000,
    batch_size_hint: int | None = None,
    with_metadata: bool = False,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> Iterator[tuple]:
    """Yield ``(window, {classification, regression})`` for one partition.

    ``with_metadata=True`` adds a third dict with Symbol / dates / Target so
    evaluation can retain identity without a second scan.
    """
    if partition not in ("train", "val", "test"):
        raise ValueError("partition must be 'train', 'val' or 'test'")
    for frame in stream_symbol_frames(path, symbols=symbols, chunksize=chunksize):
        index = build_sequence_index(frame)
        if len(index) == 0:
            continue
        parts, _ = partition_positions(frame, index, train_frac, val_frac)
        ends = np.asarray(parts[partition], dtype=np.int64)
        if len(ends) == 0:
            continue
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        # Scale the retained feature rows ONCE per symbol, then slice windows.
        scaled = values if scaler is None else transform_sequences(scaler, values)
        label_at = {int(e): i for i, e in enumerate(index.ends)}
        symbol = str(frame[SYMBOL_COLUMN].iloc[0])
        for end in ends:
            e = int(end)
            i = label_at[e]
            window = scaled[e - SEQUENCE_LENGTH + 1 : e + 1].copy()
            targets = {
                "classification": np.int64(index.labels[i]),
                "regression": np.float32(index.next_returns[i]),
            }
            if with_metadata:
                meta = {
                    "Symbol": symbol,
                    "Input_Start_Date": str(pd.Timestamp(index.input_start_dates[i]).date()),
                    "Input_End_Date": str(pd.Timestamp(index.input_end_dates[i]).date()),
                    "Target_Date": str(pd.Timestamp(index.target_dates[i]).date()),
                    "Target": np.int64(index.labels[i]),
                }
                yield window, targets, meta
            else:
                yield window, targets


def make_tf_dataset(
    path: Path,
    partition: str,
    scaler,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    reshuffle_each_iteration: bool = False,
    shuffle_buffer: int = 10_000,
    chunksize: int = 50_000,
    symbols: set[str] | None = None,
) -> object:
    """Build a ``tf.data.Dataset`` yielding model-ready ``(X, y_dict)`` batches.

    Shuffling, when enabled, is applied to the TRAIN stream only (the caller
    passes the partition) and is bounded by ``shuffle_buffer`` sequences.
    """
    import tensorflow as tf

    generator = lambda: iter_sequences(
        path, partition, scaler, symbols=symbols, chunksize=chunksize
    )
    signature = (
        tf.TensorSpec(shape=(SEQUENCE_LENGTH, N_FEATURES), dtype=tf.float32),
        {
            "classification": tf.TensorSpec(shape=(), dtype=tf.int64),
            "regression": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=signature)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer, seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
    ds = ds.batch(batch_size)
    return ds.prefetch(2)

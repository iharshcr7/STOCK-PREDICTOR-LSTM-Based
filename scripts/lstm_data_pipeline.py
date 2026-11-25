"""
lstm_data_pipeline.py
=====================
Single canonical definition of the LSTM data contract:

  * the 21 model input features and their exact calculation,
  * the 3-class (DOWN / FLAT / UP) next-session target,
  * per-symbol, per-block 60-observation sequence construction,
  * leakage-safe per-symbol chronological 70/15/15 partitioning,
  * train-only scaler fitting and shared scaling/clipping.

This module is a shared helper inside the existing `scripts/` package. It does
not add a new backend/frontend and does not change the LSTM architecture.
Historical feature engineering, live feature computation and the training
entry point all import from here so that their definitions cannot drift.

Target convention (canonical)
-----------------------------
For a row dated t, `Target` describes the return of the NEXT trading session,
from t to t+1:

    Next_Return_Pct = (Close[t+1] / Close[t] - 1) * 100
    Target = 0 (DOWN) if Next_Return_Pct < -1.0
    Target = 1 (FLAT) if -1.0 <= Next_Return_Pct <= +1.0
    Target = 2 (UP)   if Next_Return_Pct > +1.0

The final row of each symbol has no t+1 and is marked UNKNOWN (-1). It stays
in the input time series but is never used as a label. `Target_Date` stores
the exact date being predicted (t+1). `Next_Return_Pct`, `Target` and
`Target_Date` are NEVER model inputs (only FEATURE_COLUMNS are).

Sequence alignment
------------------
A sequence ending at observation t contains rows [t-59 ... t] (60 rows) of a
single symbol and a single continuous block. Its label is `Target[t]`, i.e.
the session from t to t+1. This is one step later than a sequence built from
rows [t-60 ... t-1]; the previous implementation used that off-by-one label.

Missing data / gap policy
-------------------------
Rows with invalid dates or non-positive OHLCV are dropped. Duplicate dates
within a symbol keep the last occurrence. If the gap between consecutive
valid trading dates exceeds MAX_CALENDAR_GAP_DAYS (7 calendar days, allowing
weekends/holidays), a new Sequence_Block starts. Sequences never cross a
block boundary, so they cannot silently bridge missing observations.

Short-history policy
--------------------
A (symbol, block) needs at least SEQUENCE_LENGTH + 3 labelled rows to produce
at least one sequence in each of train/validation/test. Symbols/blocks with
less history are excluded from partitioning (logged, never silently split).

The first 60 observations of a block form its first input window. They are not
dropped; their sequence inherits the partition of its target date.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from model_config import (
    DATE_COLUMN,
    FINAL_FEATURE_COLUMNS as FEATURE_COLUMNS,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60
TARGET_THRESHOLD_PCT = 1.0
TARGET_CLASSES: Dict[int, str] = {0: "DOWN", 1: "FLAT", 2: "UP"}
UNKNOWN_TARGET = -1
MAX_CALENDAR_GAP_DAYS = 7
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MIN_ROWS_FOR_PARTITION = SEQUENCE_LENGTH + 3

# Columns that must never be fed to the model.
NON_FEATURE_COLUMNS = (SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN,
                       "Target_Date", "Next_Return_Pct", "Sequence_Block")


# ---------------------------------------------------------------------------
# Per-symbol preparation
# ---------------------------------------------------------------------------

def prepare_symbol_frame(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Validate, de-duplicate, sort and block-tag one symbol's OHLCV history.
    Preserves the Symbol column. Never mixes another symbol's rows in.
    """
    df = df.copy()

    if SYMBOL_COLUMN not in df.columns or df[SYMBOL_COLUMN].isna().all():
        if symbol is None:
            raise ValueError("Symbol column missing and no symbol supplied")
        df[SYMBOL_COLUMN] = symbol
    elif symbol is not None:
        df = df[df[SYMBOL_COLUMN] == symbol].copy()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[DATE_COLUMN, *OHLCV_COLUMNS])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df[df["Volume"] >= 0]

    df = df.sort_values(DATE_COLUMN)
    df = df.drop_duplicates(subset=[DATE_COLUMN], keep="last").reset_index(drop=True)

    gap_days = df[DATE_COLUMN].diff().dt.days
    new_block = gap_days.isna() | (gap_days > MAX_CALENDAR_GAP_DAYS)
    df["Sequence_Block"] = new_block.cumsum().astype(int)

    return df


# ---------------------------------------------------------------------------
# Features (the only implementation used by training, live and CLI paths)
# ---------------------------------------------------------------------------

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the canonical 21 model features (plus a few non-model diagnostic
    columns kept for backward compatibility) from ONE symbol's sorted frame.

    Every indicator uses only information available at its own row date:
    rolling windows are trailing, ewm uses adjust=False, lags/shifts look back,
    and intraday values (High-Low, Close-Open) are known at the close.
    """
    df = df.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    close = df["Close"]

    # Trailing moving averages.
    df["MA5"] = close.rolling(window=5, min_periods=5).mean()
    df["MA10"] = close.rolling(window=10, min_periods=10).mean()
    df["MA20"] = close.rolling(window=20, min_periods=20).mean()

    # Past-return / intraday features (manual division avoids fill_method differences).
    prev_close = close.shift(1)
    df["Daily_Returns"] = (close / prev_close - 1.0) * 100.0
    df["Volatility"] = df["High"] - df["Low"]
    df["Price_Change_Pct"] = (close - df["Open"]) / df["Open"] * 100.0
    df["Lag_1"] = prev_close
    df["Lag_3"] = close.shift(3)

    # Wilder RSI(14) — ewm(alpha=1/14, adjust=False), no forward fill.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    # Explicit zero-loss convention keeps valid trading days usable:
    # no losses -> RSI 100; no gains AND no losses -> RSI 50 (neutral).
    no_loss = (avg_loss == 0) & avg_loss.notna()
    df.loc[no_loss & (avg_gain > 0), "RSI"] = 100.0
    df.loc[no_loss & (avg_gain == 0), "RSI"] = 50.0

    # True exponential moving averages (NOT rolling means).
    df["EMA12"] = close.ewm(span=12, min_periods=12, adjust=False).mean()

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, min_periods=9, adjust=False).mean()

    # Bollinger width and position (trailing 20-day mean/std).
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    band_width = upper - lower
    df["BB_Width"] = (band_width / sma20.replace(0, np.nan)) * 100.0

    # Momentum / volume / trend.
    df["Weekly_Momentum"] = (close / close.shift(5) - 1.0) * 100.0
    vol_ma5 = df["Volume"].rolling(window=5, min_periods=5).mean()
    df["Avg_5D_Volume_Trend"] = df["Volume"] / vol_ma5.replace(0, np.nan)
    ma5 = close.rolling(window=5, min_periods=5).mean()
    ma20 = close.rolling(window=20, min_periods=20).mean()
    df["Trend_Strength"] = ((ma5 - ma20).abs() / ma20.replace(0, np.nan)) * 100.0

    # Diagnostic columns (not part of FEATURE_COLUMNS); kept so existing
    # dashboard/chart code and live payloads keep their previous shape.
    df["Volume_Change_Pct"] = (df["Volume"] / df["Volume"].shift(1) - 1.0) * 100.0
    df["BB_Position"] = (close - lower) / band_width.replace(0, np.nan)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    vol_ma10 = df["Volume"].rolling(window=10, min_periods=10).mean()
    df["Avg_Volume_Trend"] = df["Volume"] / vol_ma10.replace(0, np.nan)
    df["Rolling_Std_Returns"] = df["Daily_Returns"].rolling(window=10, min_periods=10).std()

    return df


def add_next_session_target(
    df: pd.DataFrame, threshold: float = TARGET_THRESHOLD_PCT
) -> pd.DataFrame:
    """Attach 3-class Target, exact Target_Date and Next_Return_Pct."""
    df = df.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    next_close = df["Close"].shift(-1)
    df["Next_Return_Pct"] = (next_close / df["Close"] - 1.0) * 100.0
    df["Target_Date"] = df[DATE_COLUMN].shift(-1)

    ret = df["Next_Return_Pct"].to_numpy()
    target = np.select(
        [ret > threshold, ret < -threshold],
        [2, 0],
        default=1,
    ).astype(int)
    target[np.isnan(ret)] = UNKNOWN_TARGET
    df[TARGET_COLUMN] = target
    return df


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def build_labeled_sequences(
    frames: Iterable[pd.DataFrame],
    label_mode: str = "legacy_binary",
    return_regression: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame] | Tuple[
    np.ndarray, np.ndarray, pd.DataFrame, np.ndarray
]:
    """
    Build per-symbol, per-block 60-step sequences.

    label_mode:
      * "three_class"  -> y in {0,1,2} (requires a 3-unit softmax head; Phase 3)
      * "legacy_binary"-> y in {0,1}; sequences whose label is FLAT are not
        emitted as labelled samples (FLAT rows still appear inside other
        sequences' input windows). Keeps the existing binary sigmoid head usable.

    return_regression (Phase 3 multi-task interface):
      When True, also returns y_reg (n,) — the next-session percentage return
      `Next_Return_Pct` of the SAME row that produced the class label (row t,
      the sequence's final input date, predicting session t -> t+1). Units are
      percent (e.g. +1.5 means +1.5%). It is never an input feature.
      Default False keeps every existing caller's 3-value return contract.

    Returns X (n,60,21), y (n,), metadata, and optionally y_reg (n,).
    """
    if label_mode not in ("three_class", "legacy_binary"):
        raise ValueError("label_mode must be 'three_class' or 'legacy_binary'")

    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    y_reg_all: List[float] = []
    meta: List[dict] = []

    for frame in frames:
        frame = frame.copy()
        if "Next_Return_Pct" not in frame.columns:
            frame = add_next_session_target(compute_technical_features(frame))
        frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        symbol = str(frame[SYMBOL_COLUMN].iloc[0])

        for block_id, block in frame.groupby("Sequence_Block", sort=True):
            block = block.sort_values(DATE_COLUMN).reset_index(drop=True)
            # Warmup rows with incomplete indicators cannot be part of a
            # 60-observation valid input window.
            finite = np.isfinite(block[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
            block = block[finite].reset_index(drop=True)
            if len(block) < SEQUENCE_LENGTH + 1:
                continue
            values = block[FEATURE_COLUMNS].values.astype(np.float32)
            targets = block[TARGET_COLUMN].to_numpy()
            target_dates = block["Target_Date"].to_numpy()
            dates = block[DATE_COLUMN].to_numpy()
            # Regression target aligned to the SAME end row as the class label.
            next_returns = block["Next_Return_Pct"].to_numpy(dtype=np.float32)

            for end in range(SEQUENCE_LENGTH - 1, len(block) - 1):
                label = int(targets[end])
                if label == UNKNOWN_TARGET:
                    continue
                if label_mode == "legacy_binary" and label == 1:
                    continue

                # A sequence must contain 60 fully valid observations; warmup
                # windows with NaNs (MA20/RSI/MACD) are not emitted.
                window = values[end - SEQUENCE_LENGTH + 1 : end + 1]
                if not np.isfinite(window).all():
                    continue
                reg = float(next_returns[end])
                if return_regression and not np.isfinite(reg):
                    continue

                X_all.append(window)
                if return_regression:
                    y_reg_all.append(reg)
                if label_mode == "legacy_binary":
                    y_all.append(1 if label == 2 else 0)  # UP->1, DOWN->0
                else:
                    y_all.append(label)
                meta.append({
                    SYMBOL_COLUMN: symbol,
                    "Sequence_Block": int(block_id),
                    "Input_Start_Date": pd.Timestamp(dates[end - SEQUENCE_LENGTH + 1]),
                    "Input_End_Date": pd.Timestamp(dates[end]),
                    "Target_Date": pd.Timestamp(target_dates[end]),
                    "Target": label,
                })

    X = np.asarray(X_all, dtype=np.float32).reshape(
        (-1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    )
    y = np.asarray(y_all, dtype=np.int64)
    meta_df = pd.DataFrame(meta, columns=[
        SYMBOL_COLUMN, "Sequence_Block", "Input_Start_Date",
        "Input_End_Date", "Target_Date", "Target",
    ])
    if return_regression:
        return X, y, meta_df, np.asarray(y_reg_all, dtype=np.float32)
    return X, y, meta_df


# ---------------------------------------------------------------------------
# Chronological partitioning (per symbol, never global)
# ---------------------------------------------------------------------------

def chronological_partition(
    meta: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Assign each sequence to train/validation/test using its own symbol's
    target dates. Boundaries are picked from sorted UNIQUE target dates, so
    train_max_date < val_min_date <= val_max_date < test_min_date per symbol.

    Returns index arrays (train, val, test) and a per-symbol boundary report.
    """
    if meta.empty:
        empty = np.array([], dtype=int)
        return empty, empty, empty, pd.DataFrame()

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    report: List[dict] = []

    for symbol, group in meta.groupby(SYMBOL_COLUMN, sort=True):
        dates = pd.Index(pd.to_datetime(group["Target_Date"]).unique()).sort_values()
        if len(dates) < 3:
            logger.warning(
                "Excluding %s from partitioning: only %d distinct target dates "
                "(need >= 3 for train/validation/test)", symbol, len(dates)
            )
            report.append({
                SYMBOL_COLUMN: symbol, "status": "excluded_short_history",
                "distinct_target_dates": len(dates),
            })
            continue

        train_end = max(1, int(len(dates) * train_frac))
        val_end = min(len(dates) - 1, max(train_end + 1, int(len(dates) * (train_frac + val_frac))))
        train_max = dates[train_end - 1]
        val_min = dates[train_end]
        val_max = dates[val_end - 1]
        test_min = dates[val_end]

        target_dates = pd.to_datetime(group["Target_Date"])
        idx = group.index.to_numpy()
        train_idx.extend(idx[target_dates <= train_max].tolist())
        val_idx.extend(idx[(target_dates >= val_min) & (target_dates <= val_max)].tolist())
        test_idx.extend(idx[target_dates >= test_min].tolist())

        report.append({
            SYMBOL_COLUMN: symbol,
            "status": "included",
            "distinct_target_dates": len(dates),
            "train_max_target_date": train_max,
            "val_min_target_date": val_min,
            "val_max_target_date": val_max,
            "test_min_target_date": test_min,
            "n_train": int((target_dates <= train_max).sum()),
            "n_val": int(((target_dates >= val_min) & (target_dates <= val_max)).sum()),
            "n_test": int((target_dates >= test_min).sum()),
        })

    tr = np.asarray(sorted(train_idx), dtype=int)
    va = np.asarray(sorted(val_idx), dtype=int)
    te = np.asarray(sorted(test_idx), dtype=int)
    return tr, va, te, pd.DataFrame(report)


# ---------------------------------------------------------------------------
# Scaler (fit on training rows only)
# ---------------------------------------------------------------------------

def fit_scaler_on_train(X_train: np.ndarray):
    """Fit MinMaxScaler(feature_range=(0,1)) on TRAIN sequences only."""
    from sklearn.preprocessing import MinMaxScaler

    flat = np.asarray(X_train, dtype=np.float32).reshape(-1, len(FEATURE_COLUMNS))
    if flat.size == 0:
        raise ValueError("Cannot fit scaler: training partition is empty")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(flat)
    return scaler


def transform_sequences(scaler, X: np.ndarray) -> np.ndarray:
    """
    Apply an already-fitted scaler to any partition (train/val/test/live),
    clipping out-of-range values to [0,1] so live outliers cannot explode.
    """
    flat = np.asarray(X, dtype=np.float32).reshape(-1, len(FEATURE_COLUMNS))
    scaled = scaler.transform(flat)
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return scaled.reshape(np.asarray(X).shape)


def split_frames_by_symbol(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Yield one prepared frame per symbol for incremental, low-RAM processing."""
    frames = []
    for symbol in sorted(df[SYMBOL_COLUMN].dropna().unique()):
        frames.append(prepare_symbol_frame(df, symbol=str(symbol)))
    return frames

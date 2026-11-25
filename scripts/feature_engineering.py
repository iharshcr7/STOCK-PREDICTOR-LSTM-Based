"""
feature_engineering.py
======================
Create the canonical model-ready dataset from cleaned stock data.

Features, target and per-symbol handling are defined once in
`lstm_data_pipeline.py`; this script only orchestrates them over every
symbol available in the dataset (no hard-coded stock list).

Usage:
    python scripts/feature_engineering.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

from lstm_data_pipeline import (
    FEATURE_COLUMNS,
    add_next_session_target,
    compute_technical_features,
    prepare_symbol_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_stock_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Backward-compatible single-symbol helpers. All delegate to the canonical
# implementation so their definitions cannot drift from training/live paths.
def _single_symbol_frame(group: pd.DataFrame) -> pd.DataFrame:
    frame = prepare_symbol_frame(group)
    return compute_technical_features(frame)


def compute_moving_averages(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "MA5", "MA10", "MA20"]]


def compute_daily_returns(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Daily_Returns"]]


def compute_volatility(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Volatility"]]


def compute_price_change_pct(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Price_Change_Pct"]]


def compute_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Lag_1", "Lag_3"]]


def compute_rsi(group: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "RSI"]]


def compute_volume_change(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Volume_Change_Pct"]]


def compute_ema(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "EMA12"]]


def compute_bollinger_position(group: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "BB_Position"]]


def compute_macd(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "MACD", "MACD_Signal", "MACD_Hist"]]


def compute_bollinger_width(group: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "BB_Width"]]


def compute_weekly_momentum(group: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Weekly_Momentum"]]


def compute_volume_trend(group: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Avg_Volume_Trend"]]


def compute_avg_5day_volume_trend(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Avg_5D_Volume_Trend"]]


def compute_trend_strength(group: pd.DataFrame) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Trend_Strength"]]


def compute_rolling_std_returns(group: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return _single_symbol_frame(group)[["Symbol", "Date", "Rolling_Std_Returns"]]


def compute_target(group: pd.DataFrame) -> pd.DataFrame:
    return add_next_session_target(_single_symbol_frame(group))[
        ["Symbol", "Date", "Next_Return_Pct", "Target", "Target_Date"]
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per symbol (all symbols, dynamically discovered):
      1. validate / sort / de-duplicate / block-tag the chronological series
      2. compute the canonical trailing indicators
      3. attach the 3-class next-session target and its exact target date
      4. drop only warmup rows with incomplete FEATURE_COLUMNS

    FLAT (neutral) trading days are preserved. The final row of a symbol has
    no t+1 return and is kept as input history with Target = -1 (UNKNOWN);
    it is never emitted as a labelled sequence.
    """
    logger.info("Starting feature engineering on %d rows...", len(df))

    processed = []
    symbols = sorted(df["Symbol"].dropna().unique()) if "Symbol" in df.columns else []
    if not symbols:
        raise ValueError("Input dataset has no Symbol values; cannot process per symbol")

    for symbol in symbols:
        frame = prepare_symbol_frame(df, symbol=str(symbol))
        if len(frame) < len(FEATURE_COLUMNS) + 1:
            logger.warning("Skipping %s: only %d valid rows", symbol, len(frame))
            continue

        frame = compute_technical_features(frame)
        frame = add_next_session_target(frame)

        before = len(frame)
        frame = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        logger.info(
            "Processed %-6s rows=%d (warmup dropped=%d, blocks=%d)",
            symbol, len(frame), before - len(frame), frame["Sequence_Block"].nunique(),
        )
        processed.append(frame)

    if not processed:
        raise ValueError("No symbol had enough valid rows for feature engineering")

    result = pd.concat(processed, ignore_index=True)
    result["Target"] = result["Target"].astype(int)

    counts = result["Target"].value_counts().to_dict()
    logger.info(
        "Target distribution: %s",
        {str(k): counts.get(k, 0) for k in (-1, 0, 1, 2)},
    )
    return result


def main() -> None:
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        logger.error("Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    df = engineer_features(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)

    logger.info("-" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("-" * 60)
    logger.info("Output rows:      %d", len(df))
    logger.info("Model features:   %d (%s)", len(FEATURE_COLUMNS), FEATURE_COLUMNS)
    logger.info("Target:           0=DOWN, 1=FLAT, 2=UP, -1=UNKNOWN (last row)")
    logger.info("Symbols:          %d", df["Symbol"].nunique())
    logger.info("File size:        %.2f MB", size_mb)
    logger.info("Output:           %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

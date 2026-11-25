"""
predict_lstm.py
===============
LSTM live inference engine for stock movement prediction.

Flow:
    Fetch 60+ days of historical data
    → Compute same 21 features as training pipeline
    → Scale with saved MinMaxScaler
    → Build sequence of last SEQUENCE_LENGTH days
    → Run LSTM model
    → Return UP/DOWN prediction + confidence

NOTE: LSTM is LIVE MODE ONLY.
      It requires a sequence of past days — a single manual snapshot
      is not sufficient input for an LSTM model.

Usage (CLI):
    python scripts/predict_lstm.py --symbol AAPL
    python scripts/predict_lstm.py --symbol TSLA --source yfinance
"""

from __future__ import annotations

import logging
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from lstm_data_pipeline import (
    FEATURE_COLUMNS as CANONICAL_FEATURES,
    compute_technical_features,
    prepare_symbol_frame,
    transform_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras"
LSTM_SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"

SEQUENCE_LENGTH = 60  # Must match training

# Single source of truth: canonical 21-feature schema (order included).
FEATURE_COLUMNS = list(CANONICAL_FEATURES)

# Global cache — avoids reloading model on every prediction
_LSTM_MODEL_CACHE = None
_LSTM_SCALER_CACHE = None


# ---------------------------------------------------------------------------
# Model & Scaler Loading
# ---------------------------------------------------------------------------

def is_lstm_available() -> bool:
    """Check if LSTM model and scaler files exist."""
    return LSTM_MODEL_FILE.exists() and LSTM_SCALER_FILE.exists()


def load_lstm_model(use_cache: bool = True):
    """Load LSTM model with optional caching."""
    global _LSTM_MODEL_CACHE

    if use_cache and _LSTM_MODEL_CACHE is not None:
        return _LSTM_MODEL_CACHE

    if not LSTM_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {LSTM_MODEL_FILE}\n"
            f"Run: python scripts/train_lstm_model.py"
        )

    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError(
            "TensorFlow not installed.\n"
            "Run: pip install tensorflow==2.15.0"
        )

    logger.info("Loading LSTM model from: %s", LSTM_MODEL_FILE)
    model = keras.models.load_model(str(LSTM_MODEL_FILE))

    if use_cache:
        _LSTM_MODEL_CACHE = model

    logger.info("LSTM model loaded successfully")
    return model


def load_lstm_scaler(use_cache: bool = True):
    """Load MinMaxScaler with optional caching."""
    global _LSTM_SCALER_CACHE

    if use_cache and _LSTM_SCALER_CACHE is not None:
        return _LSTM_SCALER_CACHE

    if not LSTM_SCALER_FILE.exists():
        raise FileNotFoundError(
            f"LSTM scaler not found: {LSTM_SCALER_FILE}\n"
            f"Run: python scripts/train_lstm_model.py"
        )

    scaler = joblib.load(LSTM_SCALER_FILE)

    if use_cache:
        _LSTM_SCALER_CACHE = scaler

    logger.info("LSTM scaler loaded from: %s", LSTM_SCALER_FILE)
    return scaler


# ---------------------------------------------------------------------------
# Feature Engineering (mirrors fetch_live_data.py)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Compute the canonical 21 features used during LSTM training.

    Delegates to lstm_data_pipeline.compute_technical_features — the same
    implementation used by feature_engineering.py and fetch_live_data.py.
    """
    prepared = prepare_symbol_frame(df, symbol=symbol)
    return compute_technical_features(prepared)


# ---------------------------------------------------------------------------
# Sequence Builder
# ---------------------------------------------------------------------------

def build_prediction_sequence(df: pd.DataFrame, scaler, symbol: str | None = None) -> np.ndarray:
    """
    Build one (1, 60, 21) input sequence from a SINGLE symbol's history.

    Steps:
        1. Validate/sort/de-duplicate the symbol's chronological rows
        2. Compute canonical trailing features
        3. Drop warmup rows with incomplete features
        4. Take the last SEQUENCE_LENGTH consecutive observations
        5. Apply the already-fitted scaler (never re-fit) with [0,1] clipping

    The sequence ends at the latest available observation; the model predicts
    the session immediately after it. No target column is used at inference.
    """
    prepared = prepare_symbol_frame(df, symbol=symbol) if symbol or "Symbol" in df.columns else df.copy()
    df_feat = compute_technical_features(prepared)[FEATURE_COLUMNS].dropna()

    if len(df_feat) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data to build LSTM sequence for {symbol or 'symbol'}. "
            f"Need at least {SEQUENCE_LENGTH} valid rows after feature computation, "
            f"got {len(df_feat)}. "
            f"Try fetching more historical data (use outputsize='full' for Alpha Vantage "
            f"or period='1y' for yfinance)."
        )

    sequence = df_feat.tail(SEQUENCE_LENGTH).values.astype(np.float32)
    scaled = transform_sequences(scaler, sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
    return scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------

def predict_lstm_live(
    symbol: str,
    source: str = "auto",
    api_key: str = "",
) -> Dict:
    """
    Run LSTM prediction for a given stock symbol using live historical data.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    source : str
        Data source: 'alpha_vantage', 'yfinance', or 'auto'.
    api_key : str
        Alpha Vantage API key (optional if set in .env).

    Returns
    -------
    dict with keys:
        symbol, prediction, confidence, probabilities,
        latest_date, source, sequence_length, model_file
    """
    logger.info("=" * 60)
    logger.info("LSTM PREDICTION PIPELINE — %s", symbol)
    logger.info("=" * 60)

    # 1. Load model and scaler
    model  = load_lstm_model()
    scaler = load_lstm_scaler()

    # 2. Fetch historical data (need enough rows for sequence + warmup).
    #    Yahoo Finance is primary; Alpha Vantage uses the free-tier 'compact'
    #    endpoint (~100 sessions, enough for the 60-session window + warmup).
    logger.info("Fetching historical data for %s...", symbol)

    try:
        from fetch_live_data import fetch_alpha_vantage, fetch_yfinance
    except ImportError:
        raise ImportError("fetch_live_data.py not found in scripts/")

    df = None
    used_source = None

    if source in ("yfinance", "auto"):
        try:
            df = fetch_yfinance(symbol, period="1y")
            used_source = "yfinance"
        except Exception as e:
            logger.warning("yfinance failed for %s: %s", symbol, e)
            if source == "yfinance":
                raise

    if df is None and source in ("alpha_vantage", "auto"):
        import os
        _api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        df = fetch_alpha_vantage(symbol, _api_key, outputsize="compact")
        used_source = "alpha_vantage"

    logger.info("Fetched %d rows from %s", len(df), used_source)

    # 3. Build prediction sequence
    logger.info("Building LSTM input sequence (last %d days)...", SEQUENCE_LENGTH)
    X = build_prediction_sequence(df, scaler, symbol=symbol)
    logger.info("Sequence shape: %s", X.shape)

    # 4. Run prediction
    logger.info("Running LSTM inference...")
    # sigmoid output: y_proba = P(UP) since Target=1 means UP in training
    y_proba = float(model.predict(X, verbose=0).flatten()[0])
    y_pred  = 1 if y_proba >= 0.5 else 0

    p_up   = y_proba          # probability of UP
    p_down = 1.0 - y_proba    # probability of DOWN

    direction  = "UP" if y_pred == 1 else "DOWN"
    confidence = p_up * 100 if y_pred == 1 else p_down * 100

    logger.info("Prediction: %s | P(UP)=%.4f | P(DOWN)=%.4f | Confidence: %.2f%%",
                direction, p_up, p_down, confidence)

    # 5. Get latest date from data
    latest_date = str(df["Date"].iloc[-1].date()) if hasattr(df["Date"].iloc[-1], "date") else str(df["Date"].iloc[-1])

    result = {
        "symbol":          symbol,
        "prediction":      direction,
        "confidence":      round(confidence, 2),
        "probabilities": {
            "DOWN": round(p_down * 100, 2),
            "UP":   round(p_up   * 100, 2),
        },
        "latest_date":     latest_date,
        "source":          used_source,
        "sequence_length": SEQUENCE_LENGTH,
        "model_file":      str(LSTM_MODEL_FILE),
        "model_type":      "LSTM",
    }

    logger.info("=" * 60)
    logger.info("LSTM PREDICTION COMPLETE")
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM live stock prediction")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--api-key", type=str, default="", help="Alpha Vantage API key")
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["alpha_vantage", "yfinance", "auto"],
        help="Data source",
    )
    args = parser.parse_args()

    result = predict_lstm_live(
        symbol=args.symbol,
        source=args.source,
        api_key=args.api_key,
    )

    print("\n" + "=" * 60)
    print(f"  LSTM PREDICTION — {result['symbol']}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

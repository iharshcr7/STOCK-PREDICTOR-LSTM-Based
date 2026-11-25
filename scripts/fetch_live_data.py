"""
fetch_live_data.py
===================
Fetch real-time stock data from Alpha Vantage API with yfinance fallback.
Computes the same engineered features used in training so that live data
is model-ready.

Usage:
    python scripts/fetch_live_data.py --symbol AAPL
    python scripts/fetch_live_data.py --symbol TSLA --source yfinance
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from model_config import get_expected_features, validate_feature_schema
from lstm_data_pipeline import compute_technical_features, prepare_symbol_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs when python-dotenv is unavailable."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    load_env_file(PROJECT_ROOT / ".env")

# API key from environment variable (preferred) or fallback
DEFAULT_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Features the trained model expects (must match train_model.py)
MODEL_FEATURES = get_expected_features()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Fetching — Alpha Vantage
# ---------------------------------------------------------------------------


def fetch_alpha_vantage(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage TIME_SERIES_DAILY endpoint.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    api_key : str
        Alpha Vantage API key.
    outputsize : str
        'compact' (last 100 days) or 'full' (20+ years).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not provided. "
            "Set ALPHA_VANTAGE_API_KEY environment variable or pass --api-key."
        )

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    logger.info("Fetching %s from Alpha Vantage (outputsize=%s)...", symbol, outputsize)

    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Check for API errors and rate limits
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

    # Premium-only endpoints (e.g. outputsize=full) are NOT rate limits. Keep
    # them distinct so callers can choose a free-tier strategy instead of
    # retrying and burning the daily quota.
    for _key in ("Note", "Information"):
        if _key in data and "premium" in str(data[_key]).lower():
            raise ValueError(f"Alpha Vantage premium restriction: {data[_key]}")

    # Check for rate limit messages (both "Note" and "Information" keys)
    if "Note" in data:
        logger.error(f"Alpha Vantage rate limit hit: {data['Note']}")
        raise RuntimeError(
            f"Alpha Vantage API rate limit exceeded.\n"
            f"Message: {data['Note']}\n"
            f"Free tier limit: 25 requests/day, 5 requests/minute.\n"
            f"Solution: Wait a few minutes or use yfinance as fallback."
        )
    
    if "Information" in data:
        logger.error(f"Alpha Vantage API limit: {data['Information']}")
        raise RuntimeError(
            f"Alpha Vantage API rate limit exceeded.\n"
            f"Message: {data['Information']}\n"
            f"Free tier limit: 25 requests/day, 5 requests/minute.\n"
            f"Solution: Wait a few minutes or use yfinance as fallback."
        )
    
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected API response format. Keys: {list(data.keys())}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rename columns
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.insert(0, "Date", df.index)
    df = df.reset_index(drop=True)

    logger.info(
        "Alpha Vantage: %d rows fetched | %s → %s",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Data Fetching — yfinance (backup)
# ---------------------------------------------------------------------------


def fetch_yfinance(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch stock data using yfinance as a backup source.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    period : str
        Data period: '1mo', '3mo', '6mo', '1y', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    logger.info("Fetching %s from yfinance (period=%s)...", symbol, period)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError(
            f"No data returned from yfinance for {symbol} — symbol may be "
            "delisted, invalid, or not traded on the configured exchange"
        )

    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.insert(0, "Date", df.index)
    df = df.reset_index(drop=True)

    logger.info(
        "yfinance: %d rows fetched | %s → %s",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Feature Computation (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------


def compute_live_features(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """
    Compute the canonical features used during training.

    Delegates to lstm_data_pipeline.compute_technical_features so historical
    and live indicator values are computed by one implementation. The frame
    is sorted chronologically and de-duplicated first; the Symbol column is
    preserved (or added from `symbol` for live single-stock frames).
    """
    if symbol is not None and ("Symbol" not in df.columns or df["Symbol"].isna().all()):
        df = df.copy()
        df["Symbol"] = symbol
    prepared = prepare_symbol_frame(df, symbol=symbol)
    return compute_technical_features(prepared)


def validate_live_feature_schema(feature_columns: list[str]) -> None:
    validate_feature_schema(feature_columns)


# ---------------------------------------------------------------------------
# Unified Fetch + Feature Pipeline
# ---------------------------------------------------------------------------


def fetch_live_stock_data(
    symbol: str,
    api_key: str = "",
    source: str = "auto",
    return_full_history: bool = True,
) -> dict:
    """
    Fetch live stock data and compute model-ready features.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    api_key : str
        Alpha Vantage API key. Ignored if source='yfinance'.
    source : str
        'alpha_vantage', 'yfinance', or 'auto' (try AV first, fallback to yf).
    return_full_history : bool
        If True, includes full historical DataFrame for Spark processing.

    Returns
    -------
    dict
        Contains: symbol, latest quote data, computed features, source used,
        feature vector ready for model prediction, and optionally full historical DataFrame.
    """
    df = None
    used_source = None

    # --- Fetch raw data ---
    if source in ("alpha_vantage", "auto"):
        try:
            df = fetch_alpha_vantage(symbol, api_key or DEFAULT_API_KEY)
            used_source = "alpha_vantage"
        except (RuntimeError, ValueError) as exc:
            logger.warning("Alpha Vantage failed: %s", exc)
            if source == "alpha_vantage":
                # User explicitly requested Alpha Vantage, so raise the error
                raise
            # Otherwise, fall back to yfinance
            logger.info("Falling back to Yahoo Finance...")
        except Exception as exc:
            logger.warning("Alpha Vantage unexpected error: %s", exc)
            if source == "alpha_vantage":
                raise
            logger.info("Falling back to Yahoo Finance...")

    if df is None:
        df = fetch_yfinance(symbol)
        used_source = "yfinance"

    # --- Compute features (single symbol, chronological, Symbol preserved) ---
    df = compute_live_features(df, symbol=symbol)

    validate_live_feature_schema(MODEL_FEATURES)

    # Get the latest complete row (with all features computed)
    latest = df.dropna(subset=MODEL_FEATURES).iloc[-1]

    result = {
        "symbol": symbol,
        "source": used_source,
        "latest_date": str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]),
        "quote": {
            "open": round(float(latest["Open"]), 4),
            "high": round(float(latest["High"]), 4),
            "low": round(float(latest["Low"]), 4),
            "close": round(float(latest["Close"]), 4),
            "volume": int(latest["Volume"]),
        },
        "features": {col: round(float(latest[col]), 6) for col in MODEL_FEATURES},
        "feature_vector": [float(latest[col]) for col in MODEL_FEATURES],
    }
    
    # Include full historical DataFrame for Spark processing
    if return_full_history:
        # Prepare DataFrame for Spark: keep only essential columns
        spark_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_for_spark = df[spark_columns].copy()
        df_for_spark["Symbol"] = symbol
        result["historical_df"] = df_for_spark

    logger.info(
        "Live data ready for %s (source=%s, date=%s, rows=%d)",
        symbol, used_source, result["latest_date"], len(df) if return_full_history else 1,
    )
    return result


def get_historical_for_chart(
    symbol: str,
    api_key: str = "",
    source: str = "auto",
    period: str = "3mo",
) -> pd.DataFrame:
    """
    Fetch recent historical data suitable for charting (candlestick / line).
    Returns a clean DataFrame with Date, Open, High, Low, Close, Volume.
    """
    df = None

    if source in ("alpha_vantage", "auto"):
        try:
            # Free tier only supports outputsize='compact'; 'full' is premium.
            df = fetch_alpha_vantage(symbol, api_key or DEFAULT_API_KEY, outputsize="compact")
        except Exception:
            pass

    if df is None:
        df = fetch_yfinance(symbol, period=period)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live stock data")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--api-key", type=str, default="", help="Alpha Vantage API key")
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["alpha_vantage", "yfinance", "auto"],
        help="Data source",
    )
    args = parser.parse_args()

    result = fetch_live_stock_data(
        symbol=args.symbol,
        api_key=args.api_key,
        source=args.source,
    )

    print("\n" + "=" * 60)
    print(f"  LIVE DATA — {result['symbol']}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

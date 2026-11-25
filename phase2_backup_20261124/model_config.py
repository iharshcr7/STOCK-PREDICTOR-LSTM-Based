"""
model_config.py
===============
Shared feature-schema configuration and validation for the LSTM deep-learning
prediction pipeline (also used by the live-data and Spark processing modules).

UPDATED: Removed Random Forest model paths and RF metadata dependency;
the 21-feature schema used by the LSTM is preserved unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"

SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"

# REMOVED: STABLE_SYMBOLS restriction
# The system now supports all stocks dynamically from the dataset
# No hardcoded symbol filtering needed

FINAL_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA10",
    "MA20",
    "EMA12",
    "RSI",
    "MACD",
    "MACD_Signal",
    "Daily_Returns",
    "Volatility",
    "Price_Change_Pct",
    "Weekly_Momentum",
    "Trend_Strength",
    "BB_Width",
    "Avg_5D_Volume_Trend",
    "Lag_1",
    "Lag_3",
]


def read_model_metadata() -> dict:
    """Read LSTM metadata (same feature schema as the shared config below)."""
    if not LSTM_METADATA_FILE.exists():
        return {}
    try:
        return json.loads(LSTM_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_expected_features() -> list[str]:
    metadata = read_model_metadata()
    features = metadata.get("feature_columns")
    if isinstance(features, list) and features:
        return features
    return FINAL_FEATURE_COLUMNS


def validate_feature_schema(feature_columns: list[str]) -> None:
    expected = get_expected_features()
    if feature_columns != expected:
        raise ValueError(
            "Feature schema mismatch. "
            f"Expected {expected} but got {feature_columns}."
        )


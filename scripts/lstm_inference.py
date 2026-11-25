"""
lstm_inference.py
=================
Phase 4 — inference-only integration for the EXISTING NASDAQ-trained LSTM.

This module never trains, retrains, fine-tunes or rebuilds a model. It composes
the already-completed pipeline modules:

    predict_lstm.py       -> cached model/scaler loading, sequence building
    lstm_data_pipeline.py -> canonical feature calculation, scaling contract
    model_config.py       -> canonical feature schema (names AND order)
    fetch_live_data.py    -> live OHLCV retrieval

and adds the Phase 4 concerns: artifact discovery/validation, symbol
compatibility preflight, multi-stock inference, and storage-record building.

Discovered capability of the shipped checkpoint (verified from the .keras
config, not assumed):

    input  : (None, 60, 21)   -> 60 sessions x 21 canonical features
    output : (None, 1)        -> sigmoid P(UP); 1 = UP, 0 = DOWN (threshold 0.5)

There is NO trained 3-class softmax head and NO regression head in the project.
The Phase 3 scripts describe a future multi-task model but no phase3_*.keras
artifact exists. Therefore this module reports ``three_class=False`` and
``next_session_return=False`` explicitly, and leaves the return fields empty
instead of inventing predictions. If such a checkpoint is ever added, discovery
will detect and report it without changing inference behaviour here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import predict_lstm as _legacy
from lstm_data_pipeline import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    compute_technical_features,
    prepare_symbol_frame,
    transform_sequences,
)
from model_config import FINAL_FEATURE_COLUMNS, read_model_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from the shipped artifacts
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODELS_DIR / "lstm_model.keras"
SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
METADATA_FILE = MODELS_DIR / "lstm_metadata.json"

MODEL_ID = "lstm_model.keras"
MODEL_TYPE = "LSTM"
REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")

# Fraction of scaled input values allowed outside [0, 1] before we warn that a
# symbol's distribution differs materially from the scaler's training range.
OUT_OF_RANGE_WARN_FRACTION = 0.25

# Feature schema version is derived from the artifact itself, so a future
# schema change can never silently reuse an old dedup key.
def _feature_config_version(features: Iterable[str]) -> str:
    payload = json.dumps(
        {"features": list(features), "sequence_length": SEQUENCE_LENGTH},
        sort_keys=True,
    )
    return "lstm21-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Discovery / compatibility report
# ---------------------------------------------------------------------------

@dataclass
class ArtifactReport:
    """Read-only description of the trained artifacts that are actually present."""

    model_file: str
    scaler_file: str
    metadata_file: str
    model_format: str = ""
    keras_version: str = ""
    input_shape: tuple = ()
    output_shape: tuple = ()
    output_units: int | None = None
    output_activation: str = ""
    sequence_length: int = 0
    n_features: int = 0
    feature_columns: list = field(default_factory=list)
    feature_config_version: str = ""
    label_mapping: dict = field(default_factory=dict)
    threshold: float = 0.5
    scaler_type: str = ""
    scaler_n_features: int = 0
    scaler_feature_range: tuple = ()
    model_sha256: str = ""
    capabilities: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = {}
        for key, value in self.__dict__.items():
            data[key] = list(value) if isinstance(value, tuple) else value
        return data


_ARTIFACT_REPORT_CACHE: ArtifactReport | None = None


def _read_keras_config(path: Path) -> dict:
    """Read architecture metadata from a .keras zip without importing TensorFlow."""
    if not path.exists():
        raise FileNotFoundError(f"LSTM model artifact not found: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "config.json" not in names:
                raise ValueError("not a .keras archive (missing config.json)")
            config = json.loads(archive.read("config.json"))
            keras_meta = {}
            if "metadata.json" in names:
                keras_meta = json.loads(archive.read("metadata.json"))
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable Keras artifact {path}: {exc}") from exc
    return {"config": config, "metadata": keras_meta}


def discover_artifacts(strict: bool = True) -> ArtifactReport:
    """
    Discover and cross-check the trained LSTM, its scaler and its metadata.

    All checks are read-only. Nothing is fitted, compiled or trained here.

    Raises
    ------
    FileNotFoundError
        When an essential trained artifact is missing.
    ValueError
        When artifacts disagree on feature count, ordering, sequence length or
        output interpretation.
    """
    global _ARTIFACT_REPORT_CACHE
    if _ARTIFACT_REPORT_CACHE is not None and strict:
        return _ARTIFACT_REPORT_CACHE

    issues: list[str] = []
    warnings: list[str] = []

    for label, path in (
        ("model", MODEL_FILE),
        ("scaler", SCALER_FILE),
        ("metadata", METADATA_FILE),
    ):
        if not path.exists():
            issues.append(f"{label} artifact missing: {path}")

    if issues:
        # Do NOT create a substitute artifact — report the real problem.
        raise FileNotFoundError(
            "Existing LSTM artifacts are incomplete; refusing to create a "
            "replacement model.\n  - " + "\n  - ".join(issues)
        )

    metadata = read_model_metadata()
    if not metadata:
        issues.append(f"Metadata could not be read: {METADATA_FILE}")

    features = metadata.get("feature_columns", list(FINAL_FEATURE_COLUMNS))
    if not isinstance(features, list) or not features:
        issues.append("Metadata has no usable feature_columns list")
        features = list(FINAL_FEATURE_COLUMNS)
    if features != list(FINAL_FEATURE_COLUMNS):
        issues.append(
            "Feature ordering differs between metadata and model_config. "
            f"metadata={features} canonical={list(FINAL_FEATURE_COLUMNS)}"
        )
    if features != list(FEATURE_COLUMNS):
        issues.append("Feature ordering differs between metadata and lstm_data_pipeline")

    keras = _read_keras_config(MODEL_FILE)
    layers = keras["config"].get("config", {}).get("layers", [])
    input_layers = [l for l in layers if l.get("class_name") == "InputLayer"]
    if not input_layers:
        issues.append("Keras config has no InputLayer")
        input_shape = ()
    else:
        batch_shape = input_layers[0]["config"].get("batch_input_shape", [])
        input_shape = tuple(batch_shape)

    dense_layers = [l for l in layers if l.get("class_name") == "Dense"]
    if not dense_layers:
        issues.append("Keras config has no Dense output layer")
        output_units, output_activation = None, ""
    else:
        head = dense_layers[-1]["config"]
        output_units = int(head.get("units", 0))
        output_activation = str(head.get("activation", ""))
    output_shape: tuple = (None, output_units) if output_units else ()

    seq_len = int(metadata.get("sequence_length", SEQUENCE_LENGTH) or SEQUENCE_LENGTH)
    n_features = len(features)

    if input_shape and tuple(input_shape[1:]) != (seq_len, n_features):
        issues.append(
            f"Input shape mismatch: model={tuple(input_shape[1:])} "
            f"expected=({seq_len}, {n_features})"
        )

    pred_cfg = metadata.get("prediction_config", {})
    labels = pred_cfg.get("labels", {"0": "DOWN", "1": "UP"})
    labels = {str(k): str(v) for k, v in labels.items()}
    threshold = float(pred_cfg.get("threshold", 0.5))

    binary_head = (
        output_units == 1
        and output_activation == "sigmoid"
        and set(labels.values()) == {"DOWN", "UP"}
    )
    if not binary_head:
        issues.append(
            "Unexpected output head: this integration supports only the "
            f"shipped binary sigmoid head (units={output_units}, "
            f"activation={output_activation}, labels={labels})."
        )

    # Scaler cross-check (never re-fit).
    try:
        import joblib

        scaler = joblib.load(SCALER_FILE)
        scaler_type = type(scaler).__name__
        scaler_n_features = int(getattr(scaler, "n_features_in_", 0))
        scaler_range = tuple(getattr(scaler, "feature_range", ()))
        if scaler_n_features and scaler_n_features != n_features:
            issues.append(
                f"Scaler expects {scaler_n_features} features, metadata has {n_features}"
            )
        if scaler_range and tuple(scaler_range) != (0, 1):
            issues.append(f"Scaler feature_range is {scaler_range}, expected (0, 1)")
    except Exception as exc:  # pragma: no cover - dependency failure path
        issues.append(f"Scaler could not be loaded: {exc}")
        scaler_type, scaler_n_features, scaler_range = "", 0, ()

    model_sha256 = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()

    # Optional forward-compatibility detection. Presence is reported, never used.
    phase3_models = sorted(MODELS_DIR.glob("phase3_*.keras"))
    if phase3_models:
        warnings.append(
            "A versioned 3-class checkpoint exists but Phase 4 inference still "
            "targets the legacy binary checkpoint: " + ", ".join(p.name for p in phase3_models)
        )
    else:
        warnings.append(
            "No 3-class / return-regression checkpoint found. Direction is "
            "binary (UP/DOWN) and predicted next-session return is unavailable."
        )

    if strict and issues:
        raise ValueError(
            "Existing LSTM artifacts are incompatible; refusing to substitute "
            "another model.\n  - " + "\n  - ".join(issues)
        )

    report = ArtifactReport(
        model_file=str(MODEL_FILE),
        scaler_file=str(SCALER_FILE),
        metadata_file=str(METADATA_FILE),
        model_format=".keras",
        keras_version=str(keras["metadata"].get("keras_version", "")),
        input_shape=input_shape,
        output_shape=output_shape,
        output_units=output_units,
        output_activation=output_activation,
        sequence_length=seq_len,
        n_features=n_features,
        feature_columns=list(features),
        feature_config_version=_feature_config_version(features),
        label_mapping=labels,
        threshold=threshold,
        scaler_type=scaler_type,
        scaler_n_features=scaler_n_features,
        scaler_feature_range=scaler_range,
        model_sha256=model_sha256,
        capabilities={
            "direction_binary": bool(binary_head),
            "three_class": False,
            "next_session_return": False,
        },
        warnings=warnings,
    )
    if strict:
        _ARTIFACT_REPORT_CACHE = report
    return report


def validate_artifacts(force: bool = False) -> ArtifactReport:
    """Validate trained artifacts, raising a meaningful error on any problem."""
    global _ARTIFACT_REPORT_CACHE
    if force:
        _ARTIFACT_REPORT_CACHE = None
    return discover_artifacts(strict=True)


def load_bundle() -> tuple[object, object, ArtifactReport]:
    """
    Load (and cache) the existing model + fitted scaler, validating live shapes.

    Model/scaler loading is delegated to predict_lsm's cached loaders so the
    model is read from disk only once per process.
    """
    report = validate_artifacts()
    model = _legacy.load_lstm_model()      # cached, inference-only
    scaler = _legacy.load_lstm_scaler()    # cached, never re-fitted

    actual_input = tuple(model.input_shape[1:])
    actual_output = tuple(model.output_shape[1:])
    if actual_input != (report.sequence_length, report.n_features):
        raise ValueError(
            f"Loaded model input {actual_input} does not match validated "
            f"artifact contract {(report.sequence_length, report.n_features)}"
        )
    if actual_output != (1,):
        raise ValueError(
            f"Loaded model output {actual_output} is not the single sigmoid "
            "probability expected by this integration."
        )
    return model, scaler, report


# ---------------------------------------------------------------------------
# Per-symbol compatibility preflight
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityReport:
    symbol: str
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n_rows: int = 0
    n_valid_rows: int = 0
    latest_date: str = ""
    out_of_range_fraction: float = 0.0
    blocks_in_window: int = 1
    max_gap_days: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def preflight_symbol(
    df: pd.DataFrame | None,
    symbol: str,
    scaler: object | None = None,
) -> CompatibilityReport:
    """Validate one symbol's OHLCV history before it reaches the model."""
    result = CompatibilityReport(symbol=str(symbol), ok=True)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        result.ok = False
        result.errors.append("No historical data was retrieved for this symbol")
        return result

    missing = (set(REQUIRED_OHLCV) | {"Date"}) - set(df.columns)
    if missing:
        result.ok = False
        result.errors.append(f"Missing required columns: {sorted(missing)}")
        return result

    try:
        prepared = prepare_symbol_frame(df, symbol=symbol)
    except Exception as exc:
        result.ok = False
        result.errors.append(f"OHLCV validation failed: {exc}")
        return result

    result.n_rows = int(len(df))
    if prepared.empty:
        result.ok = False
        result.errors.append("No valid OHLCV rows remain after cleaning")
        return result

    result.n_valid_rows = int(len(prepared))
    result.latest_date = str(prepared["Date"].iloc[-1].date())

    tail = prepared.tail(SEQUENCE_LENGTH)
    result.blocks_in_window = int(tail["Sequence_Block"].nunique())
    gaps = tail["Date"].diff().dt.days
    result.max_gap_days = round(float(gaps.max()), 1) if gaps.notna().any() else 0.0
    if result.blocks_in_window > 1:
        if result.max_gap_days > 30:
            result.ok = False
            result.errors.append(
                f"The last {SEQUENCE_LENGTH} rows contain a {result.max_gap_days:.0f}-day "
                "data gap; the sequence is too incomplete for a meaningful prediction"
            )
        else:
            result.warnings.append(
                f"The {SEQUENCE_LENGTH}-session input window contains "
                f"{result.blocks_in_window - 1} historical gap(s) of up to "
                f"{result.max_gap_days:.0f} calendar days (common in the bundled "
                "historical CSV). Live API data is normally gap-free."
            )

    try:
        features = compute_technical_features(prepared)[FEATURE_COLUMNS]
        valid_rows = int(features.dropna().shape[0])
        if valid_rows < SEQUENCE_LENGTH:
            result.ok = False
            result.errors.append(
                f"Need at least {SEQUENCE_LENGTH} complete feature rows, got {valid_rows}"
            )
        elif scaler is not None:
            window = features.tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
            scaled = np.asarray(scaler.transform(window), dtype=np.float32)
            if not np.isfinite(scaled).all():
                result.ok = False
                result.errors.append("Scaled input contains non-finite values")
            else:
                oor = float(np.mean((scaled < 0.0) | (scaled > 1.0)))
                result.out_of_range_fraction = round(oor, 4)
                if oor > OUT_OF_RANGE_WARN_FRACTION:
                    result.warnings.append(
                        f"{oor:.0%} of scaled input values fall outside the scaler's "
                        "training range; this symbol's distribution differs from the "
                        "NASDAQ training data and the prediction may be unreliable."
                    )
    except Exception as exc:
        result.ok = False
        result.errors.append(f"Feature computation failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _base_result(symbol: str, source: str, report: ArtifactReport) -> dict:
    return {
        "symbol": str(symbol),
        "source": source,
        "model_id": MODEL_ID,
        "model_type": MODEL_TYPE,
        "model_file": report.model_file,
        "model_sha256": report.model_sha256,
        "sequence_length": report.sequence_length,
        "n_features": report.n_features,
        "feature_config": report.feature_config_version,
        "capabilities": dict(report.capabilities),
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def predict_from_frame(
    symbol: str,
    df: pd.DataFrame | None,
    *,
    source: str = "local",
) -> dict:
    """
    Run inference on an already-retrieved OHLCV frame.

    Returns a dict with ``ok=True`` and the same display keys as
    ``predict_lstm_live`` (prediction, confidence, probabilities, latest_date,
    sequence_length, model_file) plus Phase 4 metadata. On unusable input an
    ``ok=False`` dict with ``errors`` is returned instead of raising, so batch
    runs can continue for other symbols.
    """
    model, scaler, report = load_bundle()   # cached; no training anywhere
    result = _base_result(symbol, source, report)

    compat = preflight_symbol(df, symbol, scaler=scaler)
    result["compatibility"] = compat.to_dict()
    result["warnings"] = list(compat.warnings)
    if not compat.ok:
        result.update({
            "ok": False,
            "errors": compat.errors,
            "prediction": None,
            "probabilities": {},
            "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: inference was not run",
            "latest_date": compat.latest_date,
            "input_data_timestamp": compat.latest_date or None,
        })
        return result

    try:
        X = _legacy.build_prediction_sequence(df, scaler, symbol=symbol)
        raw = np.asarray(model.predict(X, verbose=0)).reshape(-1)
    except Exception as exc:
        result.update({
            "ok": False,
            "errors": [f"Inference failed: {exc}"],
            "prediction": None,
            "probabilities": {},
            "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: inference failed",
            "latest_date": compat.latest_date,
            "input_data_timestamp": compat.latest_date or None,
        })
        return result

    if raw.size != 1:
        result.update({
            "ok": False,
            "errors": [f"Expected 1 model output, got {raw.size}"],
            "prediction": None,
            "probabilities": {},
            "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: unexpected output shape",
            "latest_date": compat.latest_date,
            "input_data_timestamp": compat.latest_date or None,
        })
        return result

    prob_up = float(np.clip(raw[0], 0.0, 1.0))
    prob_down = 1.0 - prob_up
    label = 1 if prob_up >= report.threshold else 0
    direction = report.label_mapping.get(str(label), "UP" if label else "DOWN")
    confidence = max(prob_up, prob_down) * 100.0

    result.update({
        "ok": True,
        "errors": [],
        "prediction": direction,
        "predicted_label": label,
        "confidence": round(confidence, 2),
        "probabilities": {
            "DOWN": round(prob_down * 100.0, 2),
            "UP": round(prob_up * 100.0, 2),
        },
        "raw_output": round(prob_up, 6),
        "predicted_next_return_pct": None,
        "next_return_status": (
            "unavailable: the existing NASDAQ LSTM has a binary sigmoid head "
            "(P(UP)) only — no regression output and no 3-class checkpoint exists"
        ),
        "latest_date": compat.latest_date,
        "input_data_timestamp": compat.latest_date or None,
    })
    return result


def predict_from_features(
    symbol: str,
    feature_frame: pd.DataFrame | None,
    *,
    source: str = "hdfs_parquet",
    run_id: str | None = None,
    hdfs_parquet_reference: str | None = None,
) -> dict:
    """
    Infer from the authoritative Spark-processed Parquet feature rows.

    The frame must already contain the exact 21 canonical feature columns
    (computed in Spark). No OHLCV-derived feature is recomputed here: this
    function validates the processed schema, takes the last 60 ordered rows,
    applies the ORIGINAL fitted scaler with ``transform()`` only, and runs the
    ORIGINAL cached Keras model. No training or re-fitting occurs.

    Returns the same result shape as :func:`predict_from_frame`, extended with
    HDFS provenance fields used for MongoDB storage.
    """
    model, scaler, report = load_bundle()
    result = _base_result(symbol, source, report)
    result.update({
        "run_id": run_id,
        "hdfs_parquet_reference": hdfs_parquet_reference,
    })

    if feature_frame is None or not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        result.update({
            "ok": False,
            "errors": ["No processed Parquet feature rows were available for this symbol"],
            "prediction": None, "probabilities": {}, "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: inference was not run",
            "latest_date": None, "input_data_timestamp": None, "raw_output": 0.0,
        })
        return result

    missing = [c for c in FEATURE_COLUMNS if c not in feature_frame.columns]
    if missing:
        result.update({
            "ok": False,
            "errors": [f"Processed features missing required columns: {missing}"],
            "prediction": None, "probabilities": {}, "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: incompatible feature schema",
            "latest_date": None, "input_data_timestamp": None, "raw_output": 0.0,
        })
        return result

    ordered = feature_frame.copy()
    if "Date" in ordered.columns:
        ordered["Date"] = pd.to_datetime(ordered["Date"], errors="coerce")
        ordered = ordered.sort_values("Date")
    valid_frame = ordered[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_frame) < SEQUENCE_LENGTH:
        result.update({
            "ok": False,
            "errors": [
                f"Need at least {SEQUENCE_LENGTH} complete processed feature rows, "
                f"got {len(valid_frame)}"
            ],
            "prediction": None, "probabilities": {}, "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: insufficient processed history",
            "latest_date": None, "input_data_timestamp": None, "raw_output": 0.0,
        })
        return result

    window = valid_frame.tail(SEQUENCE_LENGTH).to_numpy(dtype=np.float32)
    X = transform_sequences(scaler, window.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))

    # Distribution check: the scaler is fixed; never refit it.
    scaled_raw = np.asarray(scaler.transform(window), dtype=np.float32)
    oor = float(np.mean((scaled_raw < 0.0) | (scaled_raw > 1.0)))
    warnings = []
    if oor > OUT_OF_RANGE_WARN_FRACTION:
        warnings.append(
            f"{oor:.0%} of processed features fall outside the fitted scaler's "
            "training range; prediction confidence may be unreliable."
        )

    try:
        raw = np.asarray(model.predict(X, verbose=0)).reshape(-1)
    except Exception as exc:
        result.update({
            "ok": False, "errors": [f"Inference failed: {exc}"],
            "prediction": None, "probabilities": {}, "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: inference failed",
            "latest_date": None, "input_data_timestamp": None, "raw_output": 0.0,
        })
        return result

    if raw.size != 1:
        result.update({
            "ok": False, "errors": [f"Expected 1 sigmoid output, got {raw.size}"],
            "prediction": None, "probabilities": {}, "confidence": 0.0,
            "predicted_next_return_pct": None,
            "next_return_status": "unavailable: unexpected output shape",
            "latest_date": None, "input_data_timestamp": None, "raw_output": 0.0,
        })
        return result

    prob_up = float(np.clip(raw[0], 0.0, 1.0))
    prob_down = 1.0 - prob_up
    label = 1 if prob_up >= report.threshold else 0
    direction = report.label_mapping.get(str(label), "UP" if label else "DOWN")
    latest_date = (
        str(ordered["Date"].dropna().iloc[-1].date())
        if "Date" in ordered.columns and ordered["Date"].notna().any()
        else None
    )

    result.update({
        "ok": True,
        "errors": [],
        "prediction": direction,
        "predicted_label": label,
        "raw_output": round(prob_up, 6),
        "confidence": round(max(prob_up, prob_down) * 100.0, 2),
        "probabilities": {
            "DOWN": round(prob_down * 100.0, 2),
            "UP": round(prob_up * 100.0, 2),
        },
        "predicted_next_return_pct": None,
        "next_return_status": (
            "unavailable: binary LSTM has no regression head; no three-class "
            "checkpoint exists"
        ),
        "latest_date": latest_date,
        "input_data_timestamp": latest_date,
        "warnings": warnings,
        "compatibility": {
            "out_of_range_fraction": round(oor, 4),
            "blocks_in_window": 1,
            "max_gap_days": 0.0,
        },
    })
    return result


def _fetch_history(
    symbol: str,
    source: str = "auto",
    api_key: str = "",
) -> tuple[pd.DataFrame, str]:
    """Retrieve enough history for one 60-session sequence (reuses live module)."""
    try:
        from fetch_live_data import fetch_alpha_vantage, fetch_yfinance
    except ImportError as exc:
        raise ImportError("fetch_live_data.py not found in scripts/") from exc

    df = None
    used_source = source

    # Provider strategy: Yahoo Finance first (no API quota), then the
    # free-tier-compatible Alpha Vantage 'compact' endpoint.
    if source in ("yfinance", "auto"):
        try:
            df = fetch_yfinance(symbol, period="1y")
            used_source = "yfinance"
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", symbol, exc)
            if source == "yfinance":
                raise

    if df is None and source in ("alpha_vantage", "auto"):
        try:
            key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            df = fetch_alpha_vantage(symbol, key, outputsize="compact")
            used_source = "alpha_vantage"
        except Exception as exc:
            logger.warning("Alpha Vantage failed for %s: %s", symbol, exc)
            if source == "alpha_vantage":
                raise

    if df is None:
        raise RuntimeError(f"No provider returned usable history for {symbol}")

    return df, used_source


def predict_live(
    symbol: str,
    source: str = "auto",
    api_key: str = "",
) -> dict:
    """Fetch history for ``symbol`` and run inference with the cached model."""
    df, used_source = _fetch_history(symbol, source=source, api_key=api_key)
    result = predict_from_frame(symbol, df, source=used_source)
    return result


def infer_symbols(
    symbols: Iterable[str],
    history_loader: Callable[[str], pd.DataFrame] | None = None,
    *,
    source: str = "auto",
    api_key: str = "",
) -> list[dict]:
    """
    Multi-stock inference. The model and scaler are loaded once for the whole
    batch; one incompatible symbol never aborts the others.
    """
    validate_artifacts()  # fail fast if the trained checkpoint is missing
    results: list[dict] = []
    for symbol in symbols:
        try:
            df = (
                history_loader(symbol)
                if history_loader is not None
                else _fetch_history(symbol, source=source, api_key=api_key)[0]
            )
            result = predict_from_frame(symbol, df, source=source)
        except Exception as exc:
            result = _base_result(str(symbol), source, validate_artifacts())
            result.update({
                "ok": False,
                "errors": [f"Data retrieval failed: {exc}"],
                "prediction": None,
                "probabilities": {},
                "confidence": 0.0,
                "predicted_next_return_pct": None,
                "next_return_status": "unavailable: inference was not run",
                "latest_date": "",
                "input_data_timestamp": None,
                "warnings": [],
            })
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# MongoDB record builder
# ---------------------------------------------------------------------------

def to_storage_record(result: dict) -> dict:
    """
    Convert a successful inference result into a MongoDB document.

    ``dedup_key`` is deterministic: the same symbol + input session + exact
    trained weights + feature schema always maps to one stored prediction.
    """
    if not result.get("ok"):
        raise ValueError("Refusing to store an unsuccessful inference result")

    key_source = "|".join([
        str(result["symbol"]),
        str(result.get("model_sha256", "")),
        str(result.get("input_data_timestamp", "")),
        str(result.get("feature_config", "")),
    ])
    return {
        "symbol": str(result["symbol"]),
        "prediction": result["prediction"],
        "confidence": round(float(result["confidence"]), 4),
        "timestamp": result["prediction_timestamp"],
        "source": result.get("source", ""),
        "model": MODEL_TYPE,
        "dedup_key": hashlib.sha256(key_source.encode("utf-8")).hexdigest(),
        "meta": {
            "model_id": result.get("model_id", MODEL_ID),
            "model_sha256": result.get("model_sha256", ""),
            "model_type": MODEL_TYPE,
            "feature_config_version": result.get("feature_config", ""),
            "feature_columns": list(FEATURE_COLUMNS),
            "input_data_timestamp": result.get("input_data_timestamp"),
            "direction_probabilities": dict(result.get("probabilities", {})),
            "raw_sigmoid_output": result.get("raw_output"),
            "predicted_next_return_pct": result.get("predicted_next_return_pct"),
            "next_return_status": result.get("next_return_status", ""),
            "sequence_length": result.get("sequence_length", SEQUENCE_LENGTH),
            "capabilities": dict(result.get("capabilities", {})),
            "compatibility_warnings": list(result.get("warnings", [])),
            "out_of_range_fraction": result.get("compatibility", {}).get(
                "out_of_range_fraction", 0.0
            ),
            "blocks_in_input_window": result.get("compatibility", {}).get(
                "blocks_in_window", 1
            ),
            "max_gap_days": result.get("compatibility", {}).get("max_gap_days", 0.0),
        },
    }


__all__ = [
    "ArtifactReport",
    "CompatibilityReport",
    "MODEL_ID",
    "discover_artifacts",
    "validate_artifacts",
    "load_bundle",
    "preflight_symbol",
    "predict_from_frame",
    "predict_from_features",
    "predict_live",
    "infer_symbols",
    "to_storage_record",
]

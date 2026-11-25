"""
stock_intelligence_pipeline.py
==============================
Phase 5.2 — end-to-end Stock Market Intelligence Engine orchestration.

Enforced data flow (no API-to-LSTM shortcut)::

    providers -> acquisition -> HDFS raw -> Spark -> HDFS Parquet
              -> existing LSTM inference -> MongoDB -> Streamlit

The module never trains, retrains, fine-tunes or replaces the existing NASDAQ
LSTM. All inference goes through ``lstm_inference.predict_from_features`` on
Spark-computed, Parquet-persisted features.

Usage::

    python scripts/stock_intelligence_pipeline.py --symbols AAPL,MSFT
    python scripts/stock_intelligence_pipeline.py --asset-type stocks \\
        --full-universe --batch-size 100 --max-symbols 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

import hdfs_market_store as hdfs
import lstm_inference as inference
import spark_env
from lstm_data_pipeline import FEATURE_COLUMNS, SEQUENCE_LENGTH
from mongo_store import (
    lstm_prediction_exists,
    save_lstm_prediction,
    update_lstm_prediction_meta,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STOCK_DIR = DATA_DIR / "stock_market_dataset" / "stocks"
ETF_DIR = DATA_DIR / "stock_market_dataset" / "etfs"
META_CSV = DATA_DIR / "stock_market_dataset" / "symbols_valid_meta.csv"

_SYMBOL_ASSET_MAP: dict[str, str] | None = None

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ROWS = 300  # rows retained per symbol for dashboard analytics

_SPARK_CACHE = {}


@dataclass
class SymbolInfo:
    symbol: str
    asset_type: str  # stocks | etfs
    security_name: str = ""
    in_metadata: bool = False
    financial_status: str = ""
    local_path: str | None = None

    @property
    def has_local_data(self) -> bool:
        return bool(self.local_path and Path(self.local_path).exists())

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PipelineRun:
    run_id: str
    mode: str
    asset_type: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    status: str = "initialized"
    symbols_requested: int = 0
    symbols_discovered: int = 0
    symbols_ingested: int = 0
    symbols_processed: int = 0
    input_records: int = 0
    output_records: int = 0
    failed_symbols: dict = field(default_factory=dict)
    incompatible_symbols: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    hdfs_paths: dict = field(default_factory=dict)
    spark_info: dict = field(default_factory=dict)
    mongo_ids: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    ingestion: dict = field(default_factory=dict)
    spark: dict = field(default_factory=dict)
    parquet: dict = field(default_factory=dict)
    inference: dict = field(default_factory=dict)
    mongodb: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Authoritative symbol / asset-type inventory
# ---------------------------------------------------------------------------

def _load_symbol_metadata(base: Path | None = None) -> dict[str, dict]:
    base = base or DATA_DIR
    meta_csv = base / "stock_market_dataset" / "symbols_valid_meta.csv"
    metadata: dict[str, dict] = {}
    if not meta_csv.exists():
        return metadata
    try:
        frame = pd.read_csv(
            meta_csv,
            usecols=["Symbol", "ETF", "Test Issue", "Financial Status", "Security Name"],
            dtype=str,
        )
        for row in frame.to_dict("records"):
            symbol = str(row.get("Symbol") or "").strip().upper()
            if symbol:
                metadata[symbol] = row
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read symbol metadata: %s", exc)
    return metadata


def _build_asset_map_from_metadata(
    metadata: dict[str, dict],
    stock_dir: Path,
    etf_dir: Path,
) -> dict[str, str]:
    """
    Build ``symbol -> asset_type`` from the existing inventory.

    The metadata CSV's ``ETF`` flag is authoritative. Local directory
    membership is used only as a secondary source for symbols absent from the
    metadata. Ticker text is NEVER used to guess an asset type.
    """
    mapping: dict[str, str] = {}
    for symbol, row in metadata.items():
        flag = str(row.get("ETF") or "N").strip().upper()
        mapping[symbol] = "etfs" if flag == "Y" else "stocks"
    for directory, kind in ((etf_dir, "etfs"), (stock_dir, "stocks")):
        if not directory.exists():
            continue
        for csv_file in directory.glob("*.csv"):
            mapping.setdefault(csv_file.stem.upper(), kind)
    return mapping


def symbol_asset_type(symbol: str, *, data_dir: Path | None = None) -> str | None:
    """Return the authoritative asset type for a ticker, or None if unknown."""
    global _SYMBOL_ASSET_MAP
    if _SYMBOL_ASSET_MAP is None:
        base = data_dir or DATA_DIR
        _SYMBOL_ASSET_MAP = _build_asset_map_from_metadata(
            _load_symbol_metadata(base),
            base / "stock_market_dataset" / "stocks",
            base / "stock_market_dataset" / "etfs",
        )
    return _SYMBOL_ASSET_MAP.get(str(symbol).strip().upper())


def resolve_symbol_info(
    requested: str,
    *,
    asset_type_hint: str | None = None,
    data_dir: Path | None = None,
) -> SymbolInfo:
    """
    Resolve a requested ticker to an authoritative :class:`SymbolInfo`.

    An unknown ticker is returned with ``asset_type=None``-equivalent status
    (empty string) so ingestion can fail loudly instead of writing it to the
    wrong HDFS directory.
    """
    symbol = str(requested).strip().upper()
    base = data_dir or DATA_DIR
    full_universe = discover_universe("all", data_dir=base)
    for info in full_universe:
        if info.symbol == symbol:
            return info
    resolved = symbol_asset_type(symbol, data_dir=base)
    return SymbolInfo(
        symbol=symbol,
        asset_type=resolved or "",
        in_metadata=bool(resolved),
        security_name=f"unclassified (hint={asset_type_hint})" if not resolved else "",
    )


# ---------------------------------------------------------------------------
# Full-universe discovery and coverage
# ---------------------------------------------------------------------------

def discover_universe(
    asset_type: str = "all",
    *,
    data_dir: Path | None = None,
) -> list[SymbolInfo]:
    """
    Discover stocks and ETFs from the existing dataset and metadata CSV.

    Metadata is used to flag test issues / deficient financial status; file
    presence establishes local historical availability. No files are opened.
    """
    base = data_dir or DATA_DIR
    stock_dir = base / "stock_market_dataset" / "stocks"
    etf_dir = base / "stock_market_dataset" / "etfs"
    meta_csv = base / "stock_market_dataset" / "symbols_valid_meta.csv"

    metadata = _load_symbol_metadata(base)
    global _SYMBOL_ASSET_MAP
    _SYMBOL_ASSET_MAP = _build_asset_map_from_metadata(metadata, stock_dir, etf_dir)

    wanted = ("stocks", "etfs") if asset_type == "all" else (asset_type,)
    universe: list[SymbolInfo] = []
    for kind, directory in (("stocks", stock_dir), ("etfs", etf_dir)):
        if kind not in wanted or not directory.exists():
            continue
        for csv_file in sorted(directory.glob("*.csv")):
            symbol = csv_file.stem.upper()
            meta = metadata.get(symbol, {})
            test_issue = str(meta.get("Test Issue") or "N").strip().upper() == "Y"
            status = str(meta.get("Financial Status") or "").strip()
            # Asset type comes from the inventory metadata, never from the
            # ticker text or from the user's UI selection.
            resolved_type = _SYMBOL_ASSET_MAP.get(symbol, kind)
            if resolved_type != kind:
                continue  # keep the inventory's authoritative classification
            # Test issues and financially deficient symbols are kept in the
            # coverage report but marked incompatible, never silently dropped.
            universe.append(SymbolInfo(
                symbol=symbol,
                asset_type=resolved_type,
                security_name=str(meta.get("Security Name") or "").strip(),
                in_metadata=bool(meta),
                financial_status=status,
                local_path=str(csv_file) if not test_issue else None,
            ))
    return universe


def classify_coverage(universe: Iterable[SymbolInfo]) -> dict:
    """Summarise discovery categories without claiming processing success."""
    rows = list(universe)
    return {
        "discovered": len(rows),
        "stocks": sum(1 for s in rows if s.asset_type == "stocks"),
        "etfs": sum(1 for s in rows if s.asset_type == "etfs"),
        "in_metadata": sum(1 for s in rows if s.in_metadata),
        "test_issues_excluded": sum(1 for s in rows if not s.has_local_data),
        "with_local_data": sum(1 for s in rows if s.has_local_data),
        "financially_deficient": sum(
            1 for s in rows if s.financial_status in {"D", "E", "H"}
        ),
        "symbols": [s.symbol for s in rows],
    }


# ---------------------------------------------------------------------------
# Spark processing (real implementation; injectable for tests)
# ---------------------------------------------------------------------------

def get_spark(app_name: str = "StockIntelligencePipeline", *, use_hdfs: bool = True):
    """Create/reuse one local Spark session (never one per Streamlit rerun)."""
    key = (app_name, use_hdfs)
    if key in _SPARK_CACHE:
        return _SPARK_CACHE[key]
    # Windows: patch the Java gateway launch cwd BEFORE any SparkContext exists.
    spark_env.ensure_spark_ready()
    from spark_processing import build_spark

    spark = build_spark(app_name=app_name, use_hdfs=use_hdfs)
    _SPARK_CACHE[key] = spark
    return spark


def process_raw_with_spark(
    spark,
    input_paths: list[str],
    run_id: str,
    asset_type: str,
) -> dict:
    """
    Read raw HDFS CSVs with Spark, validate/clean, compute the exact 21
    features with per-symbol windows, and write partitioned Snappy Parquet.

    Rolling calculations are performed by ``compute_spark_features`` from the
    existing ``spark_live_processing`` module, whose windows are always
    ``partitionBy("Symbol").orderBy("Date")`` — no cross-stock leakage.
    """
    from pyspark.sql import functions as F
    from spark_live_processing import compute_spark_features, validate_spark_features

    if not input_paths:
        raise ValueError("No raw HDFS input paths were provided to Spark")

    hdfs_paths = [hdfs.hdfs_uri(p) for p in input_paths]
    logger.info("Spark reading %d raw HDFS paths", len(hdfs_paths))
    raw = spark.read.option("header", True).option("inferSchema", True).csv(hdfs_paths)

    # HDFS paths are constructed by ingestion as
    # /stock_data/raw/{stocks|etfs}/<SYMBOL>/<DATE>/data.csv or
    # /stock_data/raw/live_updates/{stocks|etfs}/<SYMBOL>/<DATE>/data.csv.
    from_file = F.upper(F.regexp_extract(
        F.input_file_name(),
        r"/(?:stocks|etfs)/([^/]+)/",
        1,
    ))
    from_asset = F.regexp_extract(F.input_file_name(), r"/(stocks|etfs)/", 1)
    if "Symbol" in raw.columns:
        raw = raw.withColumn(
            "Symbol",
            F.when(
                F.col("Symbol").isNotNull() & (F.length(F.col("Symbol")) > 0),
                F.upper(F.col("Symbol")),
            ).otherwise(from_file),
        )
    else:
        # Local historical CSVs contain OHLCV only; derive the ticker from the
        # HDFS path so rolling windows stay correctly partitioned per symbol.
        raw = raw.withColumn("Symbol", from_file)
    raw = raw.filter(F.length(F.col("Symbol")) > 0)

    input_records = raw.count()
    cleaned = raw.dropDuplicates(["Symbol", "Date"]).dropna(
        subset=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    )
    cleaned = cleaned.withColumn("Date", F.to_date("Date"))
    cleaned = cleaned.filter(F.col("Date").isNotNull())

    featured = compute_spark_features(cleaned)
    validate_spark_features(featured)

    # Open/High/Low/Close/Volume are already the first five canonical feature
    # columns — listing them again would create duplicate Parquet fields.
    select_columns = ["Symbol", "Date"]
    if "Adj Close" in featured.columns:
        select_columns.append("Adj Close")
    select_columns += list(FEATURE_COLUMNS)
    output = featured.select(*select_columns).dropna(subset=list(FEATURE_COLUMNS))

    output = (
        output.withColumn("run_id", F.lit(run_id))
        .withColumn("source", F.lit("hdfs_raw"))
        .withColumn("processing_timestamp", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn(
            "asset_type",
            F.when(F.length(from_asset) > 0, from_asset).otherwise(F.lit(asset_type)),
        )
        .withColumn("year", F.year("Date"))
    )

    output_records = output.count()
    target = hdfs.features_path(run_id, asset_type)
    (
        output.write.mode("overwrite")
        .partitionBy("asset_type", "year")
        .option("compression", "snappy")
        .parquet(hdfs.hdfs_uri(target))
    )
    logger.info("Wrote %d rows to %s", output_records, target)

    return {
        "success": True,
        "input_records": input_records,
        "output_records": output_records,
        "parquet_path": target,
        "parquet_uri": hdfs.hdfs_uri(target),
        "compression": "snappy",
        "partitions": ["asset_type", "year"],
        "feature_count": len(FEATURE_COLUMNS),
        "spark_version": spark.version,
        "spark_ui_url": getattr(spark.sparkContext, "uiWebUrl", None),
    }


def read_processed_features(
    spark,
    run_id: str,
    asset_type: str,
    symbol: str,
    *,
    limit: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    """Read one symbol's processed Parquet rows into a bounded pandas frame."""
    from pyspark.sql import functions as F

    path = hdfs.hdfs_uri(hdfs.features_path(run_id, asset_type))
    frame = spark.read.parquet(path)
    frame = (
        frame.filter(F.col("Symbol") == symbol)
        .orderBy(F.col("Date").desc())
        .limit(limit)
        .toPandas()
    )
    if "Date" in frame.columns:
        frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class StockIntelligencePipeline:
    """Coordinate ingestion, Spark processing, inference and persistence."""

    def __init__(
        self,
        *,
        hdfs_bin: str = "hdfs",
        hdfs_runner: Callable | None = None,
        spark_factory: Callable | None = None,
        spark_processor: Callable | None = None,
        feature_reader: Callable | None = None,
        mongo_collection=None,
        fetcher=None,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self.hdfs_bin = hdfs_bin
        self.hdfs_runner = hdfs_runner
        self.spark_factory = spark_factory or get_spark
        self.spark_processor = spark_processor or process_raw_with_spark
        self.feature_reader = feature_reader or read_processed_features
        self.mongo_collection = mongo_collection
        self.fetcher = fetcher
        self.max_rows = int(max_rows)
        self.last_run: PipelineRun | None = None

    # -- stage 1: acquisition + HDFS raw ingestion ------------------------

    def ingest_local(
        self,
        symbols: Iterable[SymbolInfo],
        *,
        run_id: str,
        force: bool = False,
    ) -> list[dict]:
        statuses = []
        for info in symbols:
            if not info.has_local_data:
                statuses.append({
                    "symbol": info.symbol, "status": "unavailable_local_history",
                    "hdfs_path": None, "bytes": 0, "asset_type": info.asset_type,
                })
                continue
            status = hdfs.ingest_local_csv(
                info.symbol,
                Path(info.local_path),
                asset_type=info.asset_type,
                hdfs_bin=self.hdfs_bin,
                runner=self.hdfs_runner,
                force=force,
            )
            status["asset_type"] = info.asset_type
            statuses.append(status)
        return statuses

    def ingest_live(self, symbol: str, asset_type: str | None = None) -> dict:
        """
        Acquire a fresh update via the existing cached provider, merge local
        warm-up history, then persist to the HDFS raw zone.

        Asset type is always resolved from the inventory — never from the
        caller's UI selection or from ticker text.
        """
        from live_prediction_pipeline import (
            WARMUP_MIN_ROWS, MarketDataFetcher, merge_warmup_history,
        )

        resolved = asset_type or symbol_asset_type(symbol)
        if resolved not in {"stocks", "etfs"}:
            return {
                "symbol": symbol.upper(), "status": "unclassified_symbol",
                "hdfs_path": None, "bytes": 0,
                "error": "Symbol is not present in the stock/ETF inventory; "
                         "asset type cannot be determined reliably",
            }

        if self.fetcher is None:
            self.fetcher = MarketDataFetcher()
        market_data = self.fetcher.fetch(symbol)

        # AV compact returns ~100 rows; long-window indicators need history.
        merged, warmup = merge_warmup_history(
            market_data.df, symbol, asset_type=resolved
        )
        if merged.empty:
            return {
                "symbol": symbol.upper(), "status": "empty_after_warmup_merge",
                "hdfs_path": None, "bytes": 0, "warmup": warmup,
            }

        status = hdfs.ingest_dataframe(
            merged,
            symbol,
            asset_type=resolved,
            hdfs_bin=self.hdfs_bin,
            runner=self.hdfs_runner,
            live=True,
            force=True,
        )
        status["warmup"] = warmup
        status["asset_type"] = resolved
        if not warmup["sufficient"]:
            status["warning"] = (
                f"Only {warmup['merged_rows']} merged rows are available; "
                f"{WARMUP_MIN_ROWS} are recommended for SMA200 + 60-session inference"
            )
        return status

    # -- stage 2/3: Spark + Parquet ---------------------------------------

    def run_spark(self, run: PipelineRun, ingest_statuses: list[dict]) -> dict:
        # The ingestion manifest — not a guessed path — is the authoritative
        # index of what was actually written to HDFS.
        manifest = hdfs.read_ingestion_manifest(
            run.run_id, hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner
        )
        statuses = manifest.get("statuses", ingest_statuses) if manifest else ingest_statuses
        successful = [
            s["hdfs_path"] for s in statuses
            if s.get("status") in {"ingested", "already_ingested"} and s.get("hdfs_path")
        ]
        if not successful:
            raise ValueError(
                "No HDFS raw data was available for Spark processing; "
                "check the ingestion manifest and per-symbol ingestion status"
            )
        missing = [
            path for path in successful
            if not hdfs.path_exists(
                path, hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner
            )
        ]
        if missing:
            raise ValueError(
                "Ingestion claimed success but these HDFS raw paths are missing: "
                + ", ".join(missing)
            )
        run.hdfs_paths["raw_manifest"] = hdfs.ingest_manifest_path(run.run_id)
        spark = self.spark_factory()
        result = self.spark_processor(
            spark, successful, run.run_id, run.asset_type
        )
        run.spark_info = {k: v for k, v in result.items() if k != "parquet_uri"}
        run.input_records = int(result.get("input_records", 0))
        run.output_records = int(result.get("output_records", 0))
        run.hdfs_paths["raw"] = successful
        run.hdfs_paths["parquet"] = result.get("parquet_path")
        return result

    # -- stage 4: inference from processed parquet ------------------------

    def predict_symbol(
        self,
        run: PipelineRun,
        symbol: str,
        *,
        feature_frame: pd.DataFrame | None = None,
    ) -> dict:
        if feature_frame is None:
            spark = self.spark_factory()
            feature_frame = self.feature_reader(
                spark, run.run_id, run.asset_type, symbol, limit=self.max_rows
            )
        parquet_ref = (
            f"{hdfs.features_path(run.run_id, run.asset_type)} "
            f"(symbol={symbol}, rows={len(feature_frame)})"
        )
        return inference.predict_from_features(
            symbol,
            feature_frame,
            source="hdfs_parquet",
            run_id=run.run_id,
            hdfs_parquet_reference=parquet_ref,
        )

    # -- stage 5: MongoDB --------------------------------------------------

    def persist_prediction(self, run: PipelineRun, prediction: dict) -> str:
        record = inference.to_storage_record(prediction)
        record["meta"].update({
            "hdfs_parquet_reference": prediction.get("hdfs_parquet_reference"),
            "processing_run_id": prediction.get("run_id"),
            "latest_market_session": prediction.get("latest_date"),
            "data_freshness": "historical_hdfs_parquet",
            "compatibility_status": "compatible" if prediction.get("ok") else "incompatible",
            "dataset_source": "hdfs_features_daily",
            "asset_type": run.asset_type,
        })
        already_stored = lstm_prediction_exists(
            record["dedup_key"], collection=self.mongo_collection
        )
        mongo_id = save_lstm_prediction(record, collection=self.mongo_collection)
        if mongo_id:
            run.mongo_ids.append(mongo_id)
        if already_stored or not mongo_id:
            # Dedup hit: keep one record but refresh its HDFS/Spark provenance.
            update_lstm_prediction_meta(
                record["dedup_key"],
                {
                    "hdfs_parquet_reference": prediction.get("hdfs_parquet_reference"),
                    "processing_run_id": prediction.get("run_id"),
                    "latest_market_session": prediction.get("latest_date"),
                },
                collection=self.mongo_collection,
            )
        return mongo_id

    # -- full run ----------------------------------------------------------

    def run(
        self,
        symbols: Iterable[str] | Iterable[SymbolInfo],
        *,
        mode: str = "local_historical",
        asset_type: str = "stocks",
        force_ingest: bool = False,
    ) -> PipelineRun:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run = PipelineRun(
            run_id=run_id, mode=mode, asset_type=asset_type,
            status="initialized",
        )
        self.last_run = run

        infos: list[SymbolInfo] = []
        for item in symbols:
            if isinstance(item, SymbolInfo):
                info = item
                # Even when a SymbolInfo is supplied, re-check its asset type
                # against the authoritative inventory.
                authoritative = symbol_asset_type(info.symbol)
                if authoritative and info.asset_type != authoritative:
                    info = SymbolInfo(
                        symbol=info.symbol, asset_type=authoritative,
                        security_name=info.security_name, in_metadata=info.in_metadata,
                        financial_status=info.financial_status,
                        local_path=str(
                            (ETF_DIR if authoritative == "etfs" else STOCK_DIR) / f"{info.symbol}.csv"
                        ),
                    )
                infos.append(info)
                continue
            requested = str(item).strip().upper()
            infos.append(resolve_symbol_info(requested, asset_type_hint=asset_type))

        unclassified = [i.symbol for i in infos if i.asset_type not in {"stocks", "etfs"}]
        if unclassified:
            run.status = "failed"
            run.errors.append(
                "Unclassified symbols (not present in stock/ETF inventory): "
                + ", ".join(unclassified)
            )
            run.finished_at = datetime.now(timezone.utc).isoformat()
            return run
        run.symbols_requested = len(infos)
        run.symbols_discovered = len(infos)
        run.coverage = classify_coverage(infos)

        try:
            hdfs.ensure_directories(hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner)

            if mode == "live_incremental":
                ingest_statuses = [self.ingest_live(i.symbol, i.asset_type) for i in infos]
            else:
                ingest_statuses = self.ingest_local(infos, run_id=run_id, force=force_ingest)

            hdfs.write_ingestion_manifest(
                run_id, ingest_statuses,
                hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner,
            )
            run.hdfs_paths["raw_manifest"] = hdfs.ingest_manifest_path(run_id)

            for status in ingest_statuses:
                if status.get("status") not in {"ingested", "already_ingested"}:
                    run.failed_symbols[status["symbol"]] = status.get("status", "ingestion_failed")
            run.symbols_ingested = sum(
                1 for s in ingest_statuses
                if s.get("status") in {"ingested", "already_ingested"}
            )
            run.ingestion = {
                "statuses": ingest_statuses,
                "ingested": run.symbols_ingested,
                "failed": dict(run.failed_symbols),
                "manifest": hdfs.ingest_manifest_path(run_id),
            }

            spark_result = self.run_spark(run, ingest_statuses)
            run.spark = spark_result
            run.parquet = {
                "path": spark_result.get("parquet_path"),
                "uri": spark_result.get("parquet_uri"),
                "rows": spark_result.get("output_records", 0),
                "compression": spark_result.get("compression"),
                "partitions": spark_result.get("partitions", []),
            }
            run.symbols_processed = run.symbols_ingested

            for info in infos:
                if info.symbol in run.failed_symbols:
                    continue
                try:
                    prediction = self.predict_symbol(run, info.symbol)
                except Exception as exc:
                    run.failed_symbols[info.symbol] = f"inference_error: {exc}"
                    continue
                if not prediction.get("ok"):
                    run.incompatible_symbols[info.symbol] = "; ".join(
                        prediction.get("errors", [])
                    ) or prediction.get("status", "incompatible")
                    continue
                self.persist_prediction(run, prediction)
                run.predictions.append(prediction)

            run.inference = {
                "attempted": len(infos) - len(run.failed_symbols),
                "succeeded": len(run.predictions),
                "incompatible": dict(run.incompatible_symbols),
            }
            run.mongodb = {
                "written": len(run.mongo_ids),
                "ids": run.mongo_ids,
            }
            run.status = (
                "completed" if not run.failed_symbols and not run.incompatible_symbols
                else "partial" if run.predictions else "failed"
            )
            manifest = run.to_dict()
            manifest["spark_processor_result"] = spark_result
            hdfs.write_manifest(
                run_id, manifest, hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner
            )
            run.hdfs_paths["manifest"] = hdfs.manifest_path(run_id)
        except Exception as exc:
            run.status = "failed"
            run.errors.append(str(exc))
            logger.exception("Pipeline run %s failed", run_id)

        run.finished_at = datetime.now(timezone.utc).isoformat()
        return run

    def run_full_universe(
        self,
        *,
        asset_type: str = "stocks",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_symbols: int | None = None,
        mode: str = "local_historical",
    ) -> list[PipelineRun]:
        """Process the discovered universe in bounded, resumable batches."""
        universe = [
            s for s in discover_universe(asset_type)
            if s.has_local_data and s.financial_status not in {"D", "E", "H"}
        ]
        if max_symbols:
            universe = universe[:max_symbols]
        runs = []
        for start in range(0, len(universe), batch_size):
            batch = universe[start:start + batch_size]
            logger.info(
                "Full-universe batch %d-%d of %d", start + 1, start + len(batch), len(universe)
            )
            runs.append(self.run(batch, mode=mode, asset_type=asset_type))
        return runs

    def resume(self, run_id: str) -> PipelineRun | None:
        """Re-run failed/incompatible symbols from an existing manifest."""
        manifest = hdfs.read_manifest(
            run_id, hdfs_bin=self.hdfs_bin, runner=self.hdfs_runner
        )
        if not manifest:
            return None
        failed = list(manifest.get("failed_symbols", {}))
        failed += list(manifest.get("incompatible_symbols", {}))
        if not failed:
            return None
        return self.run(
            failed, mode=manifest.get("mode", "local_historical"),
            asset_type=manifest.get("asset_type", "stocks"),
        )


# ---------------------------------------------------------------------------
# Streamlit-safe result normalisation
# ---------------------------------------------------------------------------

def normalize_run_result(value) -> dict:
    """
    Normalise any session-state value into a consistent run-result dictionary.

    Streamlit can overwrite a widget-backed key with a boolean; this function
    guarantees that callers always get the documented structure with correct
    types, including for failed and partial runs.
    """
    template = {
        "run_id": "", "status": "unknown", "mode": "", "asset_type": "stocks",
        "started_at": "", "finished_at": "", "symbols_requested": 0,
        "symbols_discovered": 0, "symbols_ingested": 0, "symbols_processed": 0,
        "input_records": 0, "output_records": 0,
        "failed_symbols": {}, "incompatible_symbols": {}, "predictions": [],
        "coverage": {}, "hdfs_paths": {}, "spark_info": {}, "mongo_ids": [],
        "errors": [], "ingestion": {}, "spark": {}, "parquet": {},
        "inference": {}, "mongodb": {},
    }
    if isinstance(value, PipelineRun):
        value = value.to_dict()
    if not isinstance(value, dict):
        return template
    for key in template:
        if key not in value:
            continue
        expected = template[key]
        supplied = value[key]
        if isinstance(expected, dict) and not isinstance(supplied, dict):
            continue
        if isinstance(expected, list) and not isinstance(supplied, list):
            continue
        template[key] = supplied
    return template


# ---------------------------------------------------------------------------
# Dashboard analytics payload
# ---------------------------------------------------------------------------

def build_market_payload(feature_frame: pd.DataFrame, symbol: str) -> dict:
    """Build per-symbol OHLCV + indicator payload for Streamlit charts."""
    if feature_frame is None or feature_frame.empty:
        return {}
    frame = feature_frame.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).sort_values("Date")

    def series(name: str) -> list:
        if name not in frame.columns:
            return []
        return [
            None if pd.isna(v) else float(v) for v in frame[name]
        ]

    close = series("Close")
    dates = [d.date().isoformat() for d in frame["Date"]]
    ma20 = series("MA20")

    payload = {
        "symbol": symbol,
        "Date": dates,
        "Open": series("Open"),
        "High": series("High"),
        "Low": series("Low"),
        "Close": close,
        "Adj Close": series("Adj Close"),
        "Volume": series("Volume"),
        "MA20": ma20,
        "RSI": series("RSI"),
        "MACD": series("MACD"),
        "MACD_Signal": series("MACD_Signal"),
        "BB_Width": series("BB_Width"),
        "Daily_Returns": series("Daily_Returns"),
    }
    if close:
        s = pd.Series(close)
        payload["SMA50"] = [
            None if pd.isna(v) else float(v) for v in s.rolling(50, min_periods=50).mean()
        ]
        payload["SMA200"] = [
            None if pd.isna(v) else float(v) for v in s.rolling(200, min_periods=200).mean()
        ]
    if ma20 and payload.get("BB_Width"):
        upper, lower = [], []
        for m, w in zip(ma20, payload["BB_Width"]):
            if m is None or w is None or pd.isna(w):
                upper.append(None); lower.append(None)
            else:
                half = abs(float(m)) * float(w) / 200.0
                upper.append(float(m) + half); lower.append(float(m) - half)
        payload["BB_Upper"], payload["BB_Lower"] = upper, lower

    last = frame.iloc[-1]
    prev_close = float(frame["Close"].iloc[-2]) if len(frame) > 1 else None
    latest_close = float(last["Close"]) if pd.notna(last.get("Close")) else None
    payload["latest_metrics"] = {
        "date": str(frame["Date"].iloc[-1].date()),
        "open": float(last["Open"]) if pd.notna(last.get("Open")) else None,
        "high": float(last["High"]) if pd.notna(last.get("High")) else None,
        "low": float(last["Low"]) if pd.notna(last.get("Low")) else None,
        "close": latest_close,
        "adj_close": float(last["Adj Close"]) if "Adj Close" in frame.columns and pd.notna(last.get("Adj Close")) else None,
        "volume": float(last["Volume"]) if pd.notna(last.get("Volume")) else None,
        "previous_close": prev_close,
        "daily_change": (latest_close - prev_close) if (latest_close and prev_close) else None,
        "daily_change_pct": (
            (latest_close / prev_close - 1.0) * 100.0 if (latest_close and prev_close) else None
        ),
    }
    return payload


_CONTAINER_KEYS = ("data", "records", "ohlcv", "prices")
_METADATA_KEYS = frozenset({
    "symbol", "provider", "source", "latest_metrics", "run_id",
    "asset_type", "metadata", "meta", "freshness", "timestamp",
    "status", "ok", "error", "errors",
})
_SENSITIVE_KEY_TOKENS = ("key", "secret", "token", "password", "credential", "auth")
_CANONICAL_COLUMNS = {
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
    "symbol": "Symbol",
    "ticker": "Symbol",
    "ma20": "MA20",
    "sma20": "MA20",
    "sma50": "SMA50",
    "sma200": "SMA200",
    "rsi": "RSI",
    "macd": "MACD",
    "macd_signal": "MACD_Signal",
    "bb_width": "BB_Width",
    "bb_upper": "BB_Upper",
    "bb_lower": "BB_Lower",
    "daily_returns": "Daily_Returns",
}
_REQUIRED_OHLCV = ("Date", "Open", "High", "Low", "Close")
_NUMERIC_COLUMNS = (
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "MA20", "SMA50", "SMA200", "RSI", "MACD", "MACD_Signal",
    "BB_Width", "BB_Upper", "BB_Lower", "Daily_Returns",
)


def _is_sequence_like(value) -> bool:
    """Return True for list/tuple/ndarray/Series/Index sequence values."""
    return isinstance(value, (list, tuple, pd.Series, pd.Index)) or (
        hasattr(value, "ndim") and getattr(value, "ndim", 0) == 1
    )


def _safe_metadata_dict(raw: dict) -> dict:
    """Copy non-sensitive metadata fields without exposing credentials."""
    cleaned = {}
    for k, v in raw.items():
        if any(tok in str(k).lower() for tok in _SENSITIVE_KEY_TOKENS):
            continue
        cleaned[k] = v
    return cleaned


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map case-insensitive OHLCV and indicator column aliases to canonical names."""
    rename_map = {}
    existing = set(frame.columns)
    for col in frame.columns:
        key = str(col).strip().lower()
        target = _CANONICAL_COLUMNS.get(key)
        if target and col != target and target not in existing and target not in rename_map.values():
            rename_map[col] = target
    return frame.rename(columns=rename_map) if rename_map else frame


def _extract_raw_frame(payload, depth: int = 0) -> tuple[pd.DataFrame, dict]:
    """Extract raw tabular records and separate metadata from supported payload shapes."""
    if depth > 4 or payload is None:
        return pd.DataFrame(), {}

    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
        if "Date" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
            if "index" in frame.columns and "Date" not in frame.columns:
                frame = frame.rename(columns={"index": "Date"})
        meta = {}
        if "Symbol" in frame.columns:
            syms = [str(s) for s in frame["Symbol"].dropna().unique().tolist()]
            if len(syms) == 1:
                meta["symbol"] = syms[0]
            if syms:
                meta["symbols"] = syms
        return frame, meta

    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame(), {}
        dict_items = [item for item in payload if isinstance(item, dict)]
        if not dict_items:
            return pd.DataFrame(), {"validation_error": "List payload contains no dictionary records."}
        return pd.DataFrame(dict_items), {}

    if isinstance(payload, dict):
        if not payload:
            return pd.DataFrame(), {}

        lower_map = {str(k).lower(): k for k in payload.keys()}

        # 3. Nested market-data dictionary (data / records / ohlcv / prices)
        for container_key in _CONTAINER_KEYS:
            if container_key in lower_map:
                actual_key = lower_map[container_key]
                outer_meta = _safe_metadata_dict({
                    k: v for k, v in payload.items() if k != actual_key
                })
                inner_frame, inner_meta = _extract_raw_frame(payload[actual_key], depth + 1)
                merged_meta = {**inner_meta, **outer_meta}
                return inner_frame, merged_meta

        # Separate metadata vs sequence columns vs scalar OHLCV fields
        seq_cols = {}
        meta = {}
        scalar_cols = {}
        for k, v in payload.items():
            k_lower = str(k).lower()
            if isinstance(v, dict) or k_lower in _METADATA_KEYS:
                if not any(tok in k_lower for tok in _SENSITIVE_KEY_TOKENS):
                    meta[k] = v
                continue
            if _is_sequence_like(v):
                seq_cols[k] = list(v) if not isinstance(v, pd.Series) else v.tolist()
            else:
                scalar_cols[k] = v

        # 4. Column-oriented dictionary
        if seq_cols:
            non_empty = {k: v for k, v in seq_cols.items() if len(v) > 0}
            if not non_empty:
                return pd.DataFrame(), meta
            preferred_len = None
            for pref in ("Date", "date", "Close", "close"):
                if pref in non_empty:
                    preferred_len = len(non_empty[pref])
                    break
            if preferred_len is None:
                preferred_len = max(len(v) for v in non_empty.values())
            aligned = {}
            dropped_cols = []
            for k, v in non_empty.items():
                if len(v) == preferred_len:
                    aligned[k] = v
                else:
                    dropped_cols.append(k)
            if dropped_cols:
                logger.warning(
                    "Dropped mismatched-length sequence columns %s (expected=%d)",
                    dropped_cols, preferred_len,
                )
            return pd.DataFrame(aligned), meta

        # 5. Single OHLCV record
        if scalar_cols:
            return pd.DataFrame([scalar_cols]), meta

        return pd.DataFrame(), meta

    return pd.DataFrame(), {"validation_error": f"Unsupported payload type: {type(payload).__name__}"}


def normalize_market_payload(
    payload,
    *,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize heterogeneous market payloads into a validated OHLCV DataFrame and metadata.

    Supports:
    1. Existing Pandas DataFrames (including Spark-to-Pandas results).
    2. Lists of OHLCV dictionaries.
    3. Dictionaries containing ``data``, ``records``, ``ohlcv`` or ``prices``.
    4. Column-oriented dictionaries with equally sized arrays or Series.
    5. Single OHLCV records.
    """
    payload_type = type(payload).__name__
    if isinstance(payload, dict):
        key_names = [
            k for k in payload.keys()
            if not any(tok in str(k).lower() for tok in _SENSITIVE_KEY_TOKENS)
        ]
        nested_types = {
            k: type(payload[k]).__name__ for k in key_names
        }
        source_name = payload.get("provider") or payload.get("source") or "hdfs_parquet"
        logger.info(
            "Normalizing market payload: type=%s keys=%s value_types=%s source=%s",
            payload_type, key_names, nested_types, source_name,
        )
    else:
        logger.info("Normalizing market payload: type=%s", payload_type)

    raw_frame, metadata = _extract_raw_frame(payload)
    if symbol and not metadata.get("symbol"):
        metadata["symbol"] = symbol
    metadata.setdefault("source", metadata.get("provider") or "hdfs_parquet")

    if raw_frame.empty:
        logger.info(
            "Market payload normalized to empty frame (symbol=%s, source=%s, error=%s)",
            metadata.get("symbol"), metadata.get("source"), metadata.get("validation_error"),
        )
        return pd.DataFrame(), metadata

    frame = _canonicalize_columns(raw_frame.copy())
    if "Symbol" not in frame.columns and metadata.get("symbol"):
        frame["Symbol"] = str(metadata["symbol"])
    elif "Symbol" in frame.columns and not metadata.get("symbol"):
        syms = [str(s) for s in frame["Symbol"].dropna().unique().tolist()]
        if len(syms) == 1:
            metadata["symbol"] = syms[0]
        if syms:
            metadata["symbols"] = syms

    missing = [col for col in _REQUIRED_OHLCV if col not in frame.columns]
    if missing:
        metadata["missing_columns"] = missing
        metadata["validation_error"] = f"Missing required OHLCV columns: {', '.join(missing)}"
        logger.warning(
            "Market payload missing required OHLCV columns %s (columns=%s)",
            missing, list(frame.columns),
        )
        return pd.DataFrame(), metadata

    # Parse dates safely, normalizing timezone-aware timestamps to tz-naive UTC
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Safely convert numeric columns
    for col in _NUMERIC_COLUMNS:
        if col in frame.columns:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
            frame[col] = pd.to_numeric(numeric, errors="coerce")

    if "Volume" in frame.columns:
        frame.loc[frame["Volume"] < 0, "Volume"] = pd.NA

    # Drop invalid Date / OHLC records
    initial_count = len(frame)
    valid_mask = frame["Date"].notna()
    for col in ("Open", "High", "Low", "Close"):
        valid_mask &= frame[col].notna() & (frame[col] > 0)

    dropped_invalid = int((~valid_mask).sum())
    frame = frame.loc[valid_mask].copy()

    # Validate OHLC relationships (High >= max(Open, Close, Low) and Low <= min(Open, Close, High))
    ohlc_valid = (
        (frame["High"] >= frame["Low"])
        & (frame["High"] >= frame["Open"])
        & (frame["High"] >= frame["Close"])
        & (frame["Low"] <= frame["Open"])
        & (frame["Low"] <= frame["Close"])
    )
    invalid_ohlc_count = int((~ohlc_valid).sum())
    if invalid_ohlc_count > 0:
        logger.warning("Removed %d record(s) violating OHLC relationships", invalid_ohlc_count)
        frame = frame.loc[ohlc_valid].copy()

    # Detect and remove duplicate timestamps (per symbol when multi-symbol)
    dup_subset = (
        ["Symbol", "Date"]
        if "Symbol" in frame.columns and frame["Symbol"].notna().any()
        else ["Date"]
    )
    duplicate_count = int(frame.duplicated(subset=dup_subset).sum())
    if duplicate_count > 0:
        logger.warning("Detected and removed %d duplicate record(s) on %s", duplicate_count, dup_subset)
        frame = frame.drop_duplicates(subset=dup_subset, keep="last").copy()

    sort_cols = (
        ["Symbol", "Date"]
        if "Symbol" in frame.columns and frame["Symbol"].nunique(dropna=True) > 1
        else ["Date"]
    )
    frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    metadata["record_count"] = len(frame)
    metadata["initial_count"] = initial_count
    metadata["dropped_invalid_count"] = dropped_invalid
    metadata["invalid_ohlc_count"] = invalid_ohlc_count
    metadata["duplicate_count"] = duplicate_count

    logger.info(
        "Normalized market DataFrame: symbol=%s source=%s records=%d columns=%s",
        metadata.get("symbol"), metadata.get("source"), len(frame), list(frame.columns),
    )
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5.2 end-to-end stock intelligence pipeline")
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument("--asset-type", default="stocks", choices=["stocks", "etfs"])
    parser.add_argument("--full-universe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--mode", default="local_historical",
                        choices=["local_historical", "live_incremental"])
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()

    if args.coverage_only:
        print(json.dumps(classify_coverage(discover_universe(args.asset_type)), indent=2))
        return

    pipeline = StockIntelligencePipeline()
    if args.full_universe:
        runs = pipeline.run_full_universe(
            asset_type=args.asset_type, batch_size=args.batch_size,
            max_symbols=args.max_symbols, mode=args.mode,
        )
        payload = [run.to_dict() for run in runs]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        payload = pipeline.run(
            symbols, mode=args.mode, asset_type=args.asset_type,
            force_ingest=args.force_ingest,
        ).to_dict()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

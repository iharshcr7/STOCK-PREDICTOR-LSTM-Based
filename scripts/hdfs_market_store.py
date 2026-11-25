"""
hdfs_market_store.py
====================
Phase 5.2 — HDFS persistence layer for raw market data, Spark Parquet output
and processing manifests.

All HDFS access goes through the existing Hadoop CLI (``hdfs dfs``), reusing
the project's HADOOP_HOME configuration. The CLI runner is injectable so unit
tests can mock it without touching a real cluster.

Paths (under HDFS_ROOT, default /stock_data):

    raw/stocks/<SYMBOL>/<YYYY-MM-DD>/data.csv
    raw/etfs/<SYMBOL>/<YYYY-MM-DD>/data.csv
    raw/live_updates/<TYPE>/<SYMBOL>/<YYYY-MM-DD>/data.csv
    features/daily/<RUN>/asset_type=<TYPE>/year=<YYYY>/*.snappy.parquet
    predictions/metadata/<RUN>/manifest.json

Note: the Windows ``hdfs.cmd`` wrapper splits arguments containing ``=``,
so raw and manifest paths deliberately avoid that character. Spark's own
Parquet partition directories still use Hive-style ``key=value`` names; those
are read through Spark (not the CLI), so they are unaffected.

Historical raw data is never overwritten: ingestion is a no-op when the
partition already exists, unless ``force=True`` (explicit refresh).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

HDFS_ROOT = os.environ.get("STOCK_HDFS_ROOT", "/stock_data").rstrip("/")
HDFS_NAMENODE = os.environ.get("STOCK_HDFS_NAMENODE", "hdfs://localhost:9000")

RAW_ROOT = f"{HDFS_ROOT}/raw"
RAW_STOCKS = f"{RAW_ROOT}/stocks"
RAW_ETFS = f"{RAW_ROOT}/etfs"
RAW_LIVE = f"{RAW_ROOT}/live_updates"
RAW_METADATA = f"{RAW_ROOT}/metadata"
FEATURES_ROOT = f"{HDFS_ROOT}/features/daily"
PREDICTION_METADATA = f"{HDFS_ROOT}/predictions/metadata"

HDFSRunResult = subprocess.CompletedProcess


def resolve_hdfs_bin(hdfs_bin: str = "hdfs") -> str:
    """Locate the Hadoop CLI, honouring HADOOP_HOME on Windows."""
    resolved = shutil.which(hdfs_bin)
    if resolved:
        return resolved
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        for candidate in (
            Path(hadoop_home) / "bin" / f"{hdfs_bin}.cmd",
            Path(hadoop_home) / "bin" / hdfs_bin,
        ):
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        "Hadoop CLI not found. Set HADOOP_HOME or add 'hdfs' to PATH."
    )


def run_hdfs(
    args: list[str],
    *,
    hdfs_bin: str = "hdfs",
    runner: Callable[[list[str]], HDFSRunResult] | None = None,
    timeout: int = 180,
) -> HDFSRunResult:
    """
    Execute ``hdfs dfs`` with a safe working directory.

    The project path contains parentheses (``Stock predictor(1)``), which
    breaks ``hdfs.cmd`` batch parsing on Windows when used as cwd. Staging all
    calls from the temp directory avoids that class of failure.
    """
    if runner is not None:
        return runner(args)
    binary = resolve_hdfs_bin(hdfs_bin)
    safe_cwd = os.environ.get("TEMP") or tempfile.gettempdir()
    logger.info("hdfs dfs %s", " ".join(args))
    return subprocess.run(
        [binary, "dfs", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=safe_cwd,
    )


def ensure_directories(
    *, hdfs_bin: str = "hdfs", runner=None
) -> dict[str, bool]:
    """Create the required HDFS directory tree; returns per-path success."""
    directories = [
        RAW_STOCKS, RAW_ETFS, RAW_LIVE, RAW_METADATA,
        FEATURES_ROOT, PREDICTION_METADATA,
    ]
    status = {}
    for path in directories:
        result = run_hdfs(["-mkdir", "-p", path], hdfs_bin=hdfs_bin, runner=runner)
        status[path] = result.returncode == 0
        if result.returncode != 0:
            logger.warning("Could not create HDFS directory %s: %s", path, result.stderr)
    return status


def path_exists(
    hdfs_path: str, *, hdfs_bin: str = "hdfs", runner=None
) -> bool:
    """
    Robust existence check.

    On Windows ``hdfs.cmd`` returns exit code 0 even for missing paths, so the
    CLI output must be inspected instead of relying on the return code.
    """
    result = run_hdfs(["-ls", hdfs_path], hdfs_bin=hdfs_bin, runner=runner)
    stderr = (result.stderr or "").lower()
    if "no such file" in stderr or "does not exist" in stderr:
        return False
    return hdfs_path in (result.stdout or "") or bool((result.stdout or "").strip())


def raw_symbol_path(
    symbol: str,
    asset_type: str = "stocks",
    ingest_date: str | None = None,
    *,
    live: bool = False,
) -> str:
    if asset_type not in {"stocks", "etfs"}:
        raise ValueError(
            f"Invalid asset_type {asset_type!r} for {symbol}; must be 'stocks' or 'etfs'. "
            "Resolve the ticker from the existing inventory before ingestion."
        )
    ingest_date = ingest_date or date.today().isoformat()
    root = RAW_LIVE if live else (RAW_ETFS if asset_type == "etfs" else RAW_STOCKS)
    if live:
        return f"{root}/{asset_type}/{symbol}/{ingest_date}/data.csv"
    return f"{root}/{symbol}/{ingest_date}/data.csv"


def features_path(run_id: str, asset_type: str = "stocks") -> str:
    return f"{FEATURES_ROOT}/{run_id}"


def manifest_path(run_id: str) -> str:
    return f"{PREDICTION_METADATA}/{run_id}/manifest.json"


def _stage_local(source: Path) -> Path:
    """Copy a local file to a short temp path to avoid Windows batch escaping issues."""
    safe_name = f"hdfs_stage_{abs(hash(str(source)))}_{source.name}"
    staged = Path(tempfile.gettempdir()) / safe_name
    shutil.copyfile(source, staged)
    return staged


def put_file(
    local_path: Path,
    hdfs_path: str,
    *,
    hdfs_bin: str = "hdfs",
    runner=None,
    overwrite: bool = True,
) -> bool:
    """Upload one local file to HDFS (idempotent for identical partitions)."""
    if not local_path.exists() or local_path.stat().st_size == 0:
        logger.warning("Refusing to upload missing/empty file: %s", local_path)
        return False
    staged = _stage_local(local_path)
    try:
        parent = hdfs_path.rsplit("/", 1)[0]
        run_hdfs(["-mkdir", "-p", parent], hdfs_bin=hdfs_bin, runner=runner)
        flag = "-f" if overwrite else "-n"
        result = run_hdfs(
            ["-put", flag, str(staged), hdfs_path], hdfs_bin=hdfs_bin, runner=runner
        )
        if result.returncode != 0:
            logger.warning("HDFS put failed for %s: %s", hdfs_path, result.stderr)
            return False
        return True
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def ingest_local_csv(
    symbol: str,
    local_path: Path,
    *,
    asset_type: str = "stocks",
    ingest_date: str | None = None,
    hdfs_bin: str = "hdfs",
    runner=None,
    force: bool = False,
) -> dict:
    """
    Ingest one local historical CSV into the raw HDFS zone.

    Returns a per-symbol status dict. Existing partitions are skipped unless
    ``force`` is set, preserving historical data and avoiding repeat uploads.
    """
    target = raw_symbol_path(symbol, asset_type, ingest_date)
    if not local_path.exists():
        return {
            "symbol": symbol, "status": "missing_local_file",
            "hdfs_path": None, "bytes": 0, "force": force,
        }
    if not force and path_exists(target, hdfs_bin=hdfs_bin, runner=runner):
        return {
            "symbol": symbol, "status": "already_ingested",
            "hdfs_path": target, "bytes": local_path.stat().st_size, "force": force,
        }
    ok = put_file(local_path, target, hdfs_bin=hdfs_bin, runner=runner)
    return {
        "symbol": symbol,
        "status": "ingested" if ok else "ingestion_failed",
        "hdfs_path": target if ok else None,
        "bytes": local_path.stat().st_size if ok else 0,
        "source_timestamp": datetime.fromtimestamp(
            local_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "force": force,
    }


def ingest_dataframe(
    df,
    symbol: str,
    *,
    asset_type: str = "stocks",
    ingest_date: str | None = None,
    hdfs_bin: str = "hdfs",
    runner=None,
    live: bool = False,
    force: bool = False,
) -> dict:
    """Persist a fetched DataFrame (incremental update) into raw HDFS."""
    if df is None or getattr(df, "empty", True):
        return {
            "symbol": symbol, "status": "empty_dataframe",
            "hdfs_path": None, "bytes": 0,
        }
    target = raw_symbol_path(symbol, asset_type, ingest_date, live=live)
    if not force and path_exists(target, hdfs_bin=hdfs_bin, runner=runner):
        return {
            "symbol": symbol, "status": "already_ingested",
            "hdfs_path": target, "bytes": 0,
        }
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / f"{symbol}.csv"
        df.to_csv(staged, index=False)
        ok = put_file(staged, target, hdfs_bin=hdfs_bin, runner=runner)
    return {
        "symbol": symbol,
        "status": "ingested" if ok else "ingestion_failed",
        "hdfs_path": target if ok else None,
        "bytes": 0,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_kind": "live_incremental" if live else "local_historical",
    }


def ingest_manifest_path(run_id: str) -> str:
    return f"{RAW_METADATA}/ingest_{run_id}.json"


def write_ingestion_manifest(run_id: str, statuses: list[dict], *, hdfs_bin: str = "hdfs", runner=None) -> bool:
    """Persist the per-symbol ingestion status list (authoritative raw index)."""
    target = ingest_manifest_path(run_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "ingestion_manifest.json"
        local.write_text(
            json.dumps(
                {"run_id": run_id, "statuses": statuses,
                 "written_at": datetime.now(timezone.utc).isoformat()},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return put_file(local, target, hdfs_bin=hdfs_bin, runner=runner, overwrite=True)


def read_ingestion_manifest(run_id: str, *, hdfs_bin: str = "hdfs", runner=None) -> dict:
    result = run_hdfs(["-cat", ingest_manifest_path(run_id)], hdfs_bin=hdfs_bin, runner=runner)
    if result.returncode != 0 or not (result.stdout or "").strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def write_manifest(
    run_id: str,
    payload: dict,
    *,
    hdfs_bin: str = "hdfs",
    runner=None,
) -> bool:
    """Write a run manifest (processing metrics/provenance) to HDFS."""
    target = manifest_path(run_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "manifest.json"
        local.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return put_file(local, target, hdfs_bin=hdfs_bin, runner=runner, overwrite=True)


def read_manifest(
    run_id: str, *, hdfs_bin: str = "hdfs", runner=None
) -> dict | None:
    """Read a run manifest back from HDFS (empty dict on failure)."""
    result = run_hdfs(
        ["-cat", manifest_path(run_id)], hdfs_bin=hdfs_bin, runner=runner
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Manifest at %s is not valid JSON", manifest_path(run_id))
        return None


def list_hdfs_paths(
    root: str, *, hdfs_bin: str = "hdfs", runner=None
) -> list[str]:
    result = run_hdfs(["-ls", "-R", root], hdfs_bin=hdfs_bin, runner=runner)
    if result.returncode != 0:
        return []
    paths = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith(("-", "d")) and len(parts) >= 8:
            paths.append(parts[-1])
    return paths


def hdfs_uri(path: str) -> str:
    return f"{HDFS_NAMENODE}{path}" if path.startswith("/") else path


__all__ = [
    "HDFS_ROOT", "RAW_STOCKS", "RAW_ETFS", "RAW_LIVE", "FEATURES_ROOT",
    "PREDICTION_METADATA", "resolve_hdfs_bin", "run_hdfs", "ensure_directories",
    "path_exists", "raw_symbol_path", "features_path", "manifest_path",
    "ingest_local_csv", "ingest_dataframe", "write_manifest", "read_manifest",
    "ingest_manifest_path", "write_ingestion_manifest", "read_ingestion_manifest",
    "list_hdfs_paths", "hdfs_uri",
]

"""
spark_env.py
============
Phase 5.2.1 — Windows-safe Java/Spark/Hadoop environment detection and startup.

Root causes handled here:

1. ``\\Stock was unexpected at this time.`` / ``JAVA_GATEWAY_EXITED``:
   PySpark launches the Java gateway through ``spark-submit.cmd``. On Windows,
   cmd.exe breaks when the *current working directory* contains parentheses
   (``Stock predictor(1)``) or spaces. The gateway is therefore launched with
   a safe cwd via ``launch_gateway(popen_kwargs={"cwd": ...})``; the project
   cwd itself is never changed, so relative project paths keep working.

2. Missing JAVA_HOME / SPARK_HOME / HADOOP_HOME / winutils:
   :func:`diagnose_environment` returns structured results instead of letting
   PySpark fail with an opaque gateway error.

3. Java compatibility: Spark 3.5.x supports Java 8/11/17. Java 11 is accepted;
   other versions are reported explicitly.

CLI::

    python scripts/spark_env.py --diagnose
    python scripts/spark_env.py --test-gateway
    python scripts/spark_env.py --test-session
    python scripts/spark_env.py --env-commands
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

logger = None  # set below after logging import
import logging  # noqa: E402
logger = logging.getLogger(__name__)

SUPPORTED_JAVA_MAJOR = (8, 11, 17)
SAFE_PROJECT_PATH = Path(r"D:\StockPredictor")

_GATEWAY_PATCHED = False


# ---------------------------------------------------------------------------
# Path / environment helpers
# ---------------------------------------------------------------------------

def safe_launch_directory() -> Path:
    """
    Return a directory whose path contains no spaces or parentheses.

    Preference order: the relocated project path (D:\\StockPredictor) when it
    exists, otherwise a short temp directory (C:\\Users\\...\\Temp\\opencode is
    itself safe on most machines, but we create a guaranteed-short fallback).
    """
    if SAFE_PROJECT_PATH.exists() and SAFE_PROJECT_PATH.is_dir():
        return SAFE_PROJECT_PATH
    short = Path(tempfile.gettempdir()) / "sparklaunch"
    try:
        short.mkdir(parents=True, exist_ok=True)
        return short
    except OSError:  # pragma: no cover - defensive
        return Path(tempfile.gettempdir())


def is_safe_path(path: str | Path) -> bool:
    text = str(path)
    return " " not in text and "(" not in text and ")" not in text


def setup_pyspark_environment() -> dict:
    """
    Apply run-time environment settings required by PySpark on Windows.

    This does not persist user/system variables; it makes the current process
    correct. :func:`persistent_env_commands` prints the commands to make it
    permanent.
    """
    applied = {}
    python_exe = sys.executable
    for key in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
        os.environ[key] = python_exe
        applied[key] = python_exe

    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if hadoop_home:
        conf_dir = os.environ.get("HADOOP_CONF_DIR", "")
        if not conf_dir:
            conf_dir = str(Path(hadoop_home) / "etc" / "hadoop")
            os.environ["HADOOP_CONF_DIR"] = conf_dir
        applied["HADOOP_CONF_DIR"] = os.environ.get("HADOOP_CONF_DIR", conf_dir)

    spark_home = os.environ.get("SPARK_HOME", "")
    if not spark_home:
        try:
            import pyspark
            spark_home = str(Path(pyspark.__file__).parent)
            os.environ["SPARK_HOME"] = spark_home
        except ImportError:
            pass
    if spark_home:
        applied["SPARK_HOME"] = spark_home

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    applied["SPARK_LOCAL_IP"] = os.environ["SPARK_LOCAL_IP"]
    return applied


def patch_gateway_cwd(force: bool = False) -> bool:
    """
    Inject a safe cwd into PySpark's Java gateway launch on Windows.

    Returns True when the patch is active.
    """
    global _GATEWAY_PATCHED
    if _GATEWAY_PATCHED and not force:
        return True
    if os.name != "nt":
        return False

    import pyspark.context as pyspark_context

    original = getattr(pyspark_context, "_stock_original_launch_gateway", None)
    if original is None:
        original = pyspark_context.launch_gateway

    safe_cwd = str(safe_launch_directory())

    def _gateway_with_safe_cwd(*args, **kwargs):
        kwargs.setdefault("popen_kwargs", {})
        kwargs["popen_kwargs"].setdefault("cwd", safe_cwd)
        return original(*args, **kwargs)

    pyspark_context.launch_gateway = _gateway_with_safe_cwd
    pyspark_context._stock_original_launch_gateway = original
    _GATEWAY_PATCHED = True
    logger.info("PySpark gateway cwd set to safe path: %s", safe_cwd)
    return True


def ensure_spark_ready() -> dict:
    """Prepare environment and gateway; returns the actions taken."""
    applied = setup_pyspark_environment()
    patched = patch_gateway_cwd()
    return {"environment": applied, "gateway_cwd_patched": patched,
            "safe_launch_directory": str(safe_launch_directory())}


# ---------------------------------------------------------------------------
# Validation / diagnostics
# ---------------------------------------------------------------------------

def _java_version() -> dict:
    java_home = os.environ.get("JAVA_HOME", "")
    java_exe = Path(java_home) / "bin" / "java.exe" if java_home else "java"
    result = {
        "java_home": java_home,
        "java_executable": str(java_exe),
        "found": False,
        "version": None,
        "major": None,
        "supported": False,
        "message": "",
    }
    try:
        proc = subprocess.run(
            [str(java_exe), "-version"], capture_output=True, text=True, timeout=30
        )
        output = (proc.stderr or "") + (proc.stdout or "")
        match = re.search(r'(?:java|openjdk) version "(\d+)(?:\.\d+)*"', output)
        result["found"] = True
        if match:
            version_text = match.group(0)
            parts = re.findall(r"\d+", version_text.split('"')[1])
            first = int(parts[0]) if parts else 0
            major = first if first != 1 else (int(parts[1]) if len(parts) > 1 else 1)
            result["version"] = version_text
            result["major"] = major
            result["supported"] = major in SUPPORTED_JAVA_MAJOR
            result["message"] = (
                f"Java {major} is supported by Spark 3.5.x"
                if result["supported"]
                else f"Java {major} is not supported; use Java 8, 11 or 17"
            )
        else:
            result["message"] = "Could not parse java -version output"
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        result["message"] = f"Java not usable: {exc}"
    return result


def diagnose_environment() -> dict:
    """Structured diagnostics for Java, Spark, Hadoop and gateway settings."""
    spark_home = os.environ.get("SPARK_HOME", "")
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    project_path = Path(__file__).resolve().parent.parent

    spark_submit = ""
    if spark_home:
        for candidate in (
            Path(spark_home) / "bin" / "spark-submit.cmd",
            Path(spark_home) / "bin" / "spark-submit",
        ):
            if candidate.exists():
                spark_submit = str(candidate)
                break
    winutils = str(Path(hadoop_home) / "bin" / "winutils.exe") if hadoop_home else ""
    hadoop_dll = str(Path(hadoop_home) / "bin" / "hadoop.dll") if hadoop_home else ""

    checks = {
        "java": _java_version(),
        "spark": {
            "spark_home": spark_home,
            "spark_submit": spark_submit,
            "found": bool(spark_submit),
            "pyspark_version": None,
        },
        "hadoop": {
            "hadoop_home": hadoop_home,
            "winutils": winutils,
            "winutils_found": bool(winutils and Path(winutils).exists()),
            "hadoop_dll": hadoop_dll,
            "hadoop_dll_found": bool(hadoop_dll and Path(hadoop_dll).exists()),
            "hadoop_conf_dir": os.environ.get("HADOOP_CONF_DIR", ""),
        },
        "project": {
            "path": str(project_path),
            "safe_path": is_safe_path(project_path),
            "relocation_target": str(SAFE_PROJECT_PATH),
            "relocation_exists": SAFE_PROJECT_PATH.exists(),
        },
        "gateway": {
            "safe_launch_directory": str(safe_launch_directory()),
            "patch_applied": bool(_GATEWAY_PATCHED),
        },
    }
    try:
        import pyspark
        checks["spark"]["pyspark_version"] = pyspark.__version__
    except ImportError:
        checks["spark"]["pyspark_version"] = None

    blockers = []
    if not checks["java"]["found"] or not checks["java"]["supported"]:
        blockers.append("Java is missing or unsupported")
    if not checks["spark"]["found"]:
        blockers.append("SPARK_HOME / spark-submit not found")
    if os.name == "nt" and not checks["hadoop"]["winutils_found"]:
        blockers.append("winutils.exe not found (HADOOP_HOME)")
    if os.name == "nt" and not checks["project"]["safe_path"]:
        blockers.append(
            "Project path contains spaces/parentheses; gateway uses a safe cwd "
            "but relocating to D:\\StockPredictor is recommended"
        )
    checks["blockers"] = blockers
    checks["ready"] = not any(
        b.startswith(("Java is", "SPARK_HOME", "winutils")) for b in blockers
    )
    return checks


def test_java_gateway() -> dict:
    """Start (and stop) a Java gateway independently of any SparkSession."""
    ensure_spark_ready()
    result = {"success": False, "error": None, "safe_cwd": str(safe_launch_directory())}
    gateway = None
    try:
        from pyspark.java_gateway import launch_gateway
        gateway = launch_gateway()
        result["success"] = True
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if gateway is not None:
            try:
                gateway.shutdown()
            except Exception:
                pass
    return result


def test_spark_session(app_name: str = "SparkEnvDiagnostic") -> dict:
    """Create and stop a standalone SparkSession as an integration check."""
    ensure_spark_ready()
    result = {"success": False, "error": None, "version": None, "ui_url": None}
    spark = None
    try:
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder.appName(app_name)
            .master("local[1]")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        result["success"] = True
        result["version"] = spark.version
        result["ui_url"] = getattr(spark.sparkContext, "uiWebUrl", None)
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass
    return result


def test_hdfs_csv_read(spark, hdfs_path: str) -> dict:
    """Read a small HDFS CSV through Spark (limit pushed down)."""
    result = {"success": False, "path": hdfs_path, "rows": 0, "error": None}
    try:
        frame = spark.read.option("header", True).option("inferSchema", True).csv(hdfs_path)
        result["rows"] = frame.limit(1).count()
        result["success"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def test_hdfs_parquet_roundtrip(spark, hdfs_root: str | None = None) -> dict:
    """Write a tiny DataFrame to HDFS Parquet and read it back."""
    root = hdfs_root or f"/tmp/stock_env_test_{os.getpid()}"
    result = {"success": False, "write_path": root, "rows": 0, "error": None}
    try:
        from pyspark.sql import Row
        frame = spark.createDataFrame([Row(symbol="TEST", value=1.0)])
        frame.write.mode("overwrite").option("compression", "snappy").parquet(root)
        result["rows"] = spark.read.parquet(root).count()
        result["success"] = result["rows"] == 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def persistent_env_commands() -> dict:
    """Return commands that make the validated environment permanent."""
    return {
        "powershell": [
            'setx JAVA_HOME "C:\\Java\\jdk-11"',
            'setx SPARK_HOME "D:\\spark\\spark-3.5.8-bin-hadoop3"',
            'setx HADOOP_HOME "D:\\Hadoop\\hadoop-3.3.6\\hadoop-3.3.6"',
            'setx HADOOP_CONF_DIR "%HADOOP_HOME%\\etc\\hadoop"',
            'setx PATH "$env:PATH;$env:JAVA_HOME\\bin;$env:SPARK_HOME\\bin;$env:HADOOP_HOME\\bin"',
        ],
        "relocation": (
            "Copy the project to D:\\StockPredictor and run all commands from "
            "there. The gateway already uses a safe cwd at runtime, but the "
            "relocation removes cmd.exe parsing risk entirely."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark/Java/Hadoop environment diagnostics")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--test-gateway", action="store_true")
    parser.add_argument("--test-session", action="store_true")
    parser.add_argument("--env-commands", action="store_true")
    parser.add_argument("--csv-path", default="")
    args = parser.parse_args()

    payload = {}
    if args.env_commands:
        payload = persistent_env_commands()
    elif args.test_gateway:
        payload = test_java_gateway()
    elif args.test_session:
        payload = test_spark_session()
    else:
        payload = diagnose_environment()
        if args.csv_path:
            spark = None
            try:
                from spark_processing import build_spark
                ensure_spark_ready()
                spark = build_spark("SparkEnvCsvCheck", use_hdfs=True)
                payload["csv_read"] = test_hdfs_csv_read(spark, args.csv_path)
                payload["parquet_roundtrip"] = test_hdfs_parquet_roundtrip(spark)
            finally:
                if spark is not None:
                    spark.stop()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

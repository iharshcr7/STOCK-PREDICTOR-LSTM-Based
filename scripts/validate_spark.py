"""
validate_spark.py
=================
Phase 5.2.2 — independent Spark validation before pipeline integration.

Two separate checks:

1. Local Spark: start local[2], create a small DataFrame, count it, write a
   temporary Parquet file, read it back, verify the count, stop the session.
2. HDFS Spark: start with HDFS enabled, read a small raw CSV, write a temporary
   HDFS Parquet dataset, read it back, verify, stop the session.

No pandas fallback is used: a failure is reported as a failure.

Usage::

    python scripts/validate_spark.py
    python scripts/validate_spark.py --csv-path /stock_data/raw/stocks/AAPL/2026-11-24/data.csv
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import spark_env


def _small_frame(spark):
    return spark.createDataFrame([(i, f"symbol_{i}") for i in range(5)], ["id", "symbol"])


def validate_local_spark() -> dict:
    """Local[2] Parquet round-trip, completely independent of HDFS."""
    result = {
        "stage": "local_spark", "success": False, "master": "local[2]",
        "rows_created": 0, "rows_read": 0, "parquet_path": None,
        "spark_version": None, "error": None,
    }
    spark = None
    try:
        spark_env.ensure_spark_ready()
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("Phase522LocalValidation")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .getOrCreate()
        )
        result["spark_version"] = spark.version
        frame = _small_frame(spark)
        result["rows_created"] = frame.count()

        path = str(Path(tempfile.gettempdir()) / f"phase522_parquet_{os.getpid()}")
        frame.write.mode("overwrite").option("compression", "snappy").parquet(path)
        result["parquet_path"] = path
        result["rows_read"] = spark.read.parquet(path).count()
        result["success"] = result["rows_created"] == result["rows_read"] == 5
    except Exception as exc:  # noqa: BLE001 - diagnostic surface
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass
    return result


def validate_hdfs_spark(csv_path: str) -> dict:
    """HDFS-backed read + Parquet round-trip using the project Spark config."""
    result = {
        "stage": "hdfs_spark", "success": False, "csv_path": csv_path,
        "csv_rows": 0, "parquet_rows": 0, "parquet_path": None,
        "spark_version": None, "error": None,
    }
    spark = None
    try:
        from spark_processing import build_spark

        spark = build_spark("Phase522HdfsValidation", use_hdfs=True)
        result["spark_version"] = spark.version

        frame = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)
        result["csv_rows"] = frame.limit(1).count()

        path = f"/tmp/phase522_parquet_{int(time.time())}"
        _small_frame(spark).write.mode("overwrite").option(
            "compression", "snappy"
        ).parquet(path)
        result["parquet_path"] = path
        result["parquet_rows"] = spark.read.parquet(path).count()
        result["success"] = result["csv_rows"] == 1 and result["parquet_rows"] == 5
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if spark is not None:
            try:
                spark.stop()
            except Exception:
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Spark independently")
    parser.add_argument(
        "--csv-path",
        default="/stock_data/raw/stocks/AAPL/2026-11-24/data.csv",
        help="Small HDFS CSV used for the HDFS read check",
    )
    parser.add_argument("--skip-hdfs", action="store_true")
    args = parser.parse_args()

    payload = {
        "java": spark_env._java_version(),
        "environment": spark_env.diagnose_environment(),
        "local": validate_local_spark(),
    }
    if not args.skip_hdfs:
        payload["hdfs"] = validate_hdfs_spark(args.csv_path)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

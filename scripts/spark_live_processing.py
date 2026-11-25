"""
spark_live_processing.py
========================
Production-ready Spark live data processing pipeline.

Automatically detects new files in HDFS, processes them with Spark,
generates features matching the training pipeline, and saves to Parquet.

Flow:
    Detect new file in HDFS → Read with Spark → Compute features →
    Validate schema → Save to Parquet → Update metadata

Usage:
    spark-submit scripts/spark_live_processing.py
    spark-submit scripts/spark_live_processing.py --symbol AAPL
    python scripts/spark_live_processing.py --local  # Local mode testing
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from model_config import get_expected_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HDFS_NAMENODE = "hdfs://localhost:9000"
HDFS_LIVE_DUMPS_PATH = "/stock_data/live_api_dumps/"
HDFS_LIVE_PROCESSED_PATH = "/stock_data/live_processed/"
METADATA_DIR = PROJECT_ROOT / "metadata"
LAST_PROCESSED_FILE = METADATA_DIR / "last_processed.txt"

# Spark UI keep-alive duration (seconds)
# Set to 0 to stop Spark immediately after processing
# Set to -1 to keep Spark running indefinitely (development mode)
SPARK_UI_KEEPALIVE_SECONDS = int(os.environ.get("SPARK_UI_KEEPALIVE_SECONDS", "-1"))  # Default: unlimited (for demo)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spark Session
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "StockLiveProcessing") -> SparkSession:
    """
    Create Spark session with HDFS configuration.
    
    Args:
        app_name: Spark application name
        
    Returns:
        Configured SparkSession
    """
    logger.info("Creating Spark session...")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .getOrCreate()
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    # Get Spark UI URL
    spark_ui_url = spark.sparkContext.uiWebUrl
    
    logger.info(f"Spark session created: {spark.version}")
    if spark_ui_url:
        logger.info(f"Spark UI available at: {spark_ui_url}")
    
    return spark


# ---------------------------------------------------------------------------
# File Detection
# ---------------------------------------------------------------------------

def get_latest_file_from_hdfs(hdfs_path: str) -> Optional[str]:
    """
    Get the latest file from HDFS directory.
    
    Args:
        hdfs_path: HDFS directory path
        
    Returns:
        Latest file path or None if directory is empty
    """
    import subprocess
    
    try:
        # List files in HDFS directory
        cmd = ["hdfs", "dfs", "-ls", hdfs_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )
        
        if result.returncode != 0:
            logger.warning(f"Failed to list HDFS directory: {hdfs_path}")
            return None
        
        # Parse output to find CSV files
        lines = result.stdout.strip().split('\n')
        csv_files = []
        
        for line in lines:
            if '.csv' in line:
                parts = line.split()
                if len(parts) >= 8:
                    # Extract filename (last part)
                    filename = parts[-1]
                    # Extract timestamp (modification time)
                    timestamp = f"{parts[5]} {parts[6]}"
                    csv_files.append((filename, timestamp))
        
        if not csv_files:
            logger.info("No CSV files found in HDFS directory")
            return None
        
        # Sort by timestamp and get latest
        csv_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = csv_files[0][0]
        
        logger.info(f"Latest file in HDFS: {latest_file}")
        return latest_file
        
    except Exception as e:
        logger.error(f"Error getting latest file from HDFS: {e}")
        return None


def get_last_processed_file() -> Optional[str]:
    """
    Get the last processed file from metadata.
    
    Returns:
        Last processed file path or None
    """
    try:
        if LAST_PROCESSED_FILE.exists():
            last_file = LAST_PROCESSED_FILE.read_text().strip()
            logger.info(f"Last processed file: {last_file}")
            return last_file
        else:
            logger.info("No last processed file found (first run)")
            return None
    except Exception as e:
        logger.error(f"Error reading last processed file: {e}")
        return None


def update_last_processed_file(file_path: str) -> None:
    """
    Update the last processed file metadata.
    
    Args:
        file_path: File path that was just processed
    """
    try:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        LAST_PROCESSED_FILE.write_text(file_path)
        logger.info(f"Updated last processed file: {file_path}")
    except Exception as e:
        logger.error(f"Error updating last processed file: {e}")


def detect_new_file() -> Optional[str]:
    """
    Detect if there's a new file to process.
    
    Returns:
        New file path if found, None otherwise
    """
    logger.info("Detecting new files in HDFS...")
    
    latest_file = get_latest_file_from_hdfs(HDFS_LIVE_DUMPS_PATH)
    if not latest_file:
        logger.info("No files found in HDFS")
        return None
    
    last_processed = get_last_processed_file()
    
    if last_processed is None or latest_file != last_processed:
        logger.info(f"New file detected: {latest_file}")
        return latest_file
    else:
        logger.info("No new files to process")
        return None


# ---------------------------------------------------------------------------
# Feature Engineering with Spark
# ---------------------------------------------------------------------------

def _ema_over_window(column_name: str, window: Window, alpha: float, min_periods: int):
    """
    Exact EWM (adjust=False) expressed with Spark higher-order functions.

    collect_list over the ordered window yields the chronological values;
    aggregate applies EMA_t = (1-alpha)*EMA_{t-1} + alpha*x_t starting from
    the first value. This is mathematically equivalent to pandas
    Series.ewm(alpha=..., adjust=False).mean(); a rolling average is never
    presented as an EMA. Rows before `min_periods` are null (pandas parity).
    """
    values = F.collect_list(F.col(column_name).cast("double")).over(
        window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    # Null/warmup values (e.g. early MACD) must not seed the recursion.
    clean = F.filter(values, lambda value: value.isNotNull())
    first = F.element_at(clean, 1).cast("double")
    tail = F.slice(clean, 2, F.size(clean) - 1)
    ema = F.aggregate(
        tail,
        first,
        lambda acc, value: ((1.0 - alpha) * acc.cast("double")) + (alpha * value.cast("double")),
    )
    return F.when(F.size(clean) >= min_periods, ema)


def _rolling_mean(column_name: str, window: Window, n: int):
    """Trailing n-row mean with pandas min_periods=n semantics."""
    w = window.rowsBetween(-(n - 1), 0)
    return F.when(
        F.count(column_name).over(w) == n,
        F.avg(column_name).over(w),
    )


def compute_spark_features(df: DataFrame) -> DataFrame:
    """
    Compute the canonical feature set with Spark, correctly isolated by symbol.

    All windows are partitionBy("Symbol").orderBy("Date") — no rolling value
    can combine different stocks. EMA12/MACD/RSI use true exponential/Wilder
    recursion, not trailing means. Duplicate (Symbol, Date) rows are removed
    before any window calculation.
    """
    logger.info("Computing features with Spark (per-symbol windows)...")

    if "Symbol" not in df.columns:
        logger.warning("Input has no Symbol column; assigning 'UNKNOWN'")
        df = df.withColumn("Symbol", F.lit("UNKNOWN"))

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df = df.withColumn(col, F.col(col).cast(DoubleType()))

    df = df.withColumn("Date", F.to_date("Date"))
    df = df.dropna(subset=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.dropDuplicates(["Symbol", "Date"])

    window_spec = Window.partitionBy("Symbol").orderBy("Date")

    # Trailing means (min_periods parity with the Pandas pipeline).
    df = df.withColumn("MA5", _rolling_mean("Close", window_spec, 5))
    df = df.withColumn("MA10", _rolling_mean("Close", window_spec, 10))
    df = df.withColumn("MA20", _rolling_mean("Close", window_spec, 20))

    prev_close = F.lag("Close", 1).over(window_spec)
    df = df.withColumn("Daily_Returns", ((F.col("Close") - prev_close) / prev_close) * 100.0)
    df = df.withColumn("Volatility", F.col("High") - F.col("Low"))
    df = df.withColumn("Price_Change_Pct",
                       ((F.col("Close") - F.col("Open")) / F.col("Open")) * 100.0)
    df = df.withColumn("Lag_1", prev_close)
    df = df.withColumn("Lag_3", F.lag("Close", 3).over(window_spec))

    # TRUE exponential moving averages (span -> alpha = 2/(span+1)).
    df = df.withColumn("EMA12", _ema_over_window("Close", window_spec, 2.0 / 13.0, 12))
    df = df.withColumn("_EMA26", _ema_over_window("Close", window_spec, 2.0 / 27.0, 26))
    df = df.withColumn("MACD", F.col("EMA12") - F.col("_EMA26"))
    df = df.withColumn("MACD_Signal", _ema_over_window("MACD", window_spec, 2.0 / 10.0, 9))

    # Wilder RSI: EWM with alpha = 1/14, min_periods = 14.
    change = F.col("Close") - F.lag("Close", 1).over(window_spec)
    df = df.withColumn("_price_change", change)
    df = df.withColumn("_gain", F.when(F.col("_price_change") > 0, F.col("_price_change")).otherwise(F.lit(0.0)))
    df = df.withColumn("_loss", F.when(F.col("_price_change") < 0, -F.col("_price_change")).otherwise(F.lit(0.0)))
    df = df.withColumn("_avg_gain", _ema_over_window("_gain", window_spec, 1.0 / 14.0, 14))
    df = df.withColumn("_avg_loss", _ema_over_window("_loss", window_spec, 1.0 / 14.0, 14))
    rs = F.col("_avg_gain") / F.when(F.col("_avg_loss") == 0, F.lit(None).cast("double")).otherwise(F.col("_avg_loss"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df = df.withColumn("RSI", F.when(
        F.col("_avg_loss") == 0,
        F.when(F.col("_avg_gain") > 0, F.lit(100.0)).otherwise(F.lit(50.0)),
    ).otherwise(rsi))

    # Bollinger width: trailing 20-row mean +/- 2 std.
    w20 = window_spec.rowsBetween(-19, 0)
    std20 = F.when(F.count("Close").over(w20) == 20, F.stddev("Close").over(w20))
    df = df.withColumn("BB_Width", ((4.0 * std20) / F.col("MA20")) * 100.0)

    # Momentum / volume / trend (all trailing, per symbol).
    prev5 = F.lag("Close", 5).over(window_spec)
    df = df.withColumn("Weekly_Momentum",
                       F.when(F.row_number().over(window_spec) >= 6,
                              ((F.col("Close") - prev5) / prev5) * 100.0))
    df = df.withColumn("Avg_5D_Volume_Trend",
                       F.col("Volume") / _rolling_mean("Volume", window_spec, 5))
    df = df.withColumn("Trend_Strength",
                       (F.abs(F.col("MA5") - F.col("MA20")) / F.col("MA20")) * 100.0)

    df = df.drop("_price_change", "_gain", "_loss", "_avg_gain", "_avg_loss",
                 "_EMA26", "MACD_Hist")
    logger.info("Features computed successfully")
    return df


def validate_spark_features(df: DataFrame) -> bool:
    """
    Validate that all required features are present.
    
    Args:
        df: Spark DataFrame with features
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    logger.info("Validating feature schema...")
    
    expected_features = get_expected_features()
    df_columns = df.columns
    
    missing_features = set(expected_features) - set(df_columns)
    
    if missing_features:
        raise ValueError(
            f"Missing features in Spark output: {sorted(missing_features)}\n"
            f"Expected: {expected_features}"
        )
    
    logger.info(f"Feature schema validated ({len(expected_features)} features)")
    return True


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_live_file(
    spark: SparkSession,
    file_path: str,
    symbol: Optional[str] = None
) -> bool:
    """
    Process a single live file with Spark.
    
    Args:
        spark: SparkSession
        file_path: HDFS path to CSV file (can be with or without hdfs:// prefix)
        symbol: Optional symbol filter
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("=" * 70)
        logger.info(f"Processing file: {file_path}")
        logger.info("=" * 70)
        
        # Step 1: Construct full HDFS path
        # If file_path doesn't start with hdfs://, add the namenode prefix
        if not file_path.startswith("hdfs://"):
            hdfs_file_path = f"{HDFS_NAMENODE}{file_path}"
        else:
            hdfs_file_path = file_path
        
        logger.info(f"Full HDFS path: {hdfs_file_path}")
        
        # Step 2: Read CSV from HDFS
        logger.info("Reading CSV from HDFS...")
        df = spark.read.csv(hdfs_file_path, header=True, inferSchema=True)
        
        row_count = df.count()
        logger.info(f"Read {row_count} rows from HDFS")
        
        if row_count == 0:
            logger.warning("File is empty, skipping")
            return False
        
        # Step 2: Filter by symbol if specified
        if symbol:
            df = df.filter(F.col("Symbol") == symbol)
            filtered_count = df.count()
            logger.info(f"Filtered to {filtered_count} rows for symbol {symbol}")
        
        # Step 3: Compute features
        df = compute_spark_features(df)
        
        # Step 4: Validate features
        validate_spark_features(df)
        
        # Step 5: Select only required features
        expected_features = get_expected_features()
        feature_columns = ["Symbol", "Date"] + expected_features
        df = df.select(*[c for c in feature_columns if c in df.columns])
        
        # Step 6: Drop rows with null values in critical features
        df = df.dropna(subset=expected_features)
        
        # Cache DataFrame so it appears in Spark UI Storage tab
        df.cache()
        
        final_count = df.count()
        logger.info(f"Final dataset: {final_count} rows with complete features (cached for Spark UI Storage)")
        
        if final_count == 0:
            logger.warning("No rows with complete features, skipping save")
            return False
        
        # Step 7: Save to Parquet in HDFS
        # Extract filename without extension for output naming
        filename = Path(file_path).stem
        output_path = f"{HDFS_NAMENODE}{HDFS_LIVE_PROCESSED_PATH}{filename}.parquet"
        
        logger.info(f"Saving processed data to: {output_path}")
        
        # Create output directory if needed
        import subprocess
        mkdir_cmd = ["hdfs", "dfs", "-mkdir", "-p", HDFS_LIVE_PROCESSED_PATH]
        subprocess.run(mkdir_cmd, capture_output=True, timeout=30, check=False, shell=True)
        
        # Save as Parquet (overwrite mode)
        df.write.mode("overwrite").parquet(output_path)
        
        logger.info(f"Processed data saved to Parquet: {output_path}")
        
        # Step 8: Update metadata
        update_last_processed_file(file_path)
        
        logger.info("=" * 70)
        logger.info("Processing completed successfully")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.exception(f"Error processing file {file_path}")
        return False


def _keep_spark_alive_background(spark_session, keepalive_seconds: int, ui_url: str):
    """
    Keep Spark session alive in a background thread so the UI remains accessible.
    After the keepalive period, the Spark session is stopped automatically.
    
    Args:
        spark_session: Active SparkSession to keep alive
        keepalive_seconds: How long to keep the session alive
        ui_url: Spark UI URL for logging
    """
    import time
    try:
        minutes = keepalive_seconds / 60
        logger.info(f"Spark UI background thread started — UI alive for {minutes:.1f} minutes at: {ui_url}")
        time.sleep(keepalive_seconds)
    except Exception as e:
        logger.warning(f"Spark UI keepalive thread error: {e}")
    finally:
        try:
            spark_session.stop()
            logger.info("Spark session stopped (keepalive expired)")
        except Exception:
            pass


def run_spark_live_processing(
    symbol: Optional[str] = None,
    force: bool = False
) -> Dict:
    """
    Main entry point for Spark live processing.
    
    Args:
        symbol: Optional symbol to filter
        force: Force processing even if no new file detected
        
    Returns:
        Dictionary with processing results including spark_ui_url
    """
    result = {
        "success": False,
        "new_file_detected": False,
        "file_processed": None,
        "rows_processed": 0,
        "error": None,
        "spark_ui_url": None,
        "timestamp": datetime.now().isoformat(),
    }
    
    spark = None
    spark_kept_alive = False  # Track if background thread owns the session
    
    try:
        # Step 1: Detect new file
        if not force:
            new_file = detect_new_file()
            if not new_file:
                logger.info("No new files to process")
                result["success"] = True
                result["message"] = "No new files to process"
                return result
            
            result["new_file_detected"] = True
            result["file_processed"] = new_file
        else:
            # Force mode: process latest file
            new_file = get_latest_file_from_hdfs(HDFS_LIVE_DUMPS_PATH)
            if not new_file:
                raise ValueError("No files found in HDFS")
            result["file_processed"] = new_file
        
        # Step 2: Create Spark session
        spark = create_spark_session()
        
        # Capture Spark UI URL
        if spark and spark.sparkContext.uiWebUrl:
            result["spark_ui_url"] = spark.sparkContext.uiWebUrl
            logger.info(f"Spark UI URL: {result['spark_ui_url']}")
        
        # Step 3: Process file
        success = process_live_file(spark, new_file, symbol)
        
        if success:
            result["success"] = True
            result["message"] = "Processing completed successfully"
            
            # Keep Spark UI alive in a BACKGROUND THREAD so results return immediately
            # while the user can still access localhost:4040
            if SPARK_UI_KEEPALIVE_SECONDS > 0 and result["spark_ui_url"]:
                import threading
                minutes = SPARK_UI_KEEPALIVE_SECONDS / 60
                logger.info("=" * 70)
                logger.info(f"Spark UI will stay alive for {minutes:.1f} minutes")
                logger.info(f"Access Spark UI at: {result['spark_ui_url']}")
                logger.info("Stages and jobs are visible in the UI now")
                logger.info("=" * 70)
                
                # Start background thread to keep Spark alive
                keepalive_thread = threading.Thread(
                    target=_keep_spark_alive_background,
                    args=(spark, SPARK_UI_KEEPALIVE_SECONDS, result["spark_ui_url"]),
                    daemon=True,
                    name="SparkUI-Keepalive",
                )
                keepalive_thread.start()
                spark_kept_alive = True  # Don't stop in finally block
                
            elif SPARK_UI_KEEPALIVE_SECONDS == -1:
                # Development mode: keep Spark running indefinitely
                logger.info("=" * 70)
                logger.info("SPARK UI DEVELOPMENT MODE")
                logger.info("=" * 70)
                logger.info(f"Spark UI available at: {result['spark_ui_url']}")
                logger.info("Spark session will NOT be stopped (development mode)")
                logger.info("=" * 70)
                spark_kept_alive = True  # Don't stop in finally block
        else:
            result["error"] = "Processing failed (see logs)"
        
        return result
        
    except Exception as e:
        logger.exception("Spark live processing failed")
        result["error"] = str(e)
        return result
        
    finally:
        # Only stop Spark if background thread is NOT keeping it alive
        if spark and not spark_kept_alive:
            spark.stop()
            logger.info("Spark session stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI entry point for Spark live processing.
    
    Examples:
        spark-submit scripts/spark_live_processing.py
        spark-submit scripts/spark_live_processing.py --symbol AAPL
        spark-submit scripts/spark_live_processing.py --force
        python scripts/spark_live_processing.py --local
    """
    parser = argparse.ArgumentParser(
        description="Spark live data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process new files (auto-detect)
  spark-submit scripts/spark_live_processing.py
  
  # Process specific symbol
  spark-submit scripts/spark_live_processing.py --symbol AAPL
  
  # Force processing of latest file
  spark-submit scripts/spark_live_processing.py --force
  
  # Local mode (for testing without spark-submit)
  python scripts/spark_live_processing.py --local
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        help="Filter by stock symbol (e.g., AAPL)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if no new file detected"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "summary"],
        default="summary",
        help="Output format (default: summary)"
    )
    
    args = parser.parse_args()
    
    # Run processing
    result = run_spark_live_processing(
        symbol=args.symbol,
        force=args.force
    )
    
    # Output results
    if args.output == "json":
        print("\n" + json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 70)
        print("  SPARK LIVE PROCESSING RESULT")
        print("=" * 70)
        print(f"  Success:           {result['success']}")
        print(f"  New File Detected: {result['new_file_detected']}")
        print(f"  File Processed:    {result.get('file_processed', 'None')}")
        if result.get('error'):
            print(f"  Error:             {result['error']}")
        print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()

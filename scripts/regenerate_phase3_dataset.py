"""
regenerate_phase3_dataset.py
============================
Regenerate the model-ready dataset with the CORRECTED Phase 2 pipeline and
write it to a NEW file (data/phase3_featured_data.csv).

Nothing else is touched: the stale data/final_featured_data.csv and every
model artifact remain exactly where they are.

Resource policy: the 5,884 raw stock CSVs (2.2 GB total) are opened ONE AT A
TIME. One symbol's frame is prepared, featured, targeted and appended to the
output CSV before the next file is read. Raw histories are never all resident
in RAM simultaneously.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from lstm_data_pipeline import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    UNKNOWN_TARGET,
    add_next_session_target,
    compute_technical_features,
    prepare_symbol_frame,
)
from symbol_mapper import (
    ETFS_DIR,
    STOCKS_DIR,
    discover_stocks_from_directory,
    get_supported_stocks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "phase3_featured_data.csv"
STATS_FILE = PROJECT_ROOT / "data" / "phase3_dataset_generation_stats.json"

# Documented Phase 2 exclusion policy (preprocessing.filter_low_data_stocks
# uses 100 raw rows; feature_engineering needs at least 22 valid rows; a
# 60-step sequence needs at least 61 finite feature rows).
MIN_RAW_ROWS = 100
MIN_VALID_ROWS = len(FEATURE_COLUMNS) + 1
MIN_FEATURE_ROWS = SEQUENCE_LENGTH + 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def regenerate(output: Path = OUTPUT_FILE, limit: int | None = None) -> dict:
    """Stream every discovered symbol through the canonical Phase 2 helpers."""
    output = Path(output)
    if output.resolve() == (PROJECT_ROOT / "data" / "final_featured_data.csv").resolve():
        raise RuntimeError("Refusing to overwrite the stale final_featured_data.csv")
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Remove it explicitly before regenerating."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    discovered = get_supported_stocks()
    etf_symbols = discover_stocks_from_directory(ETFS_DIR)
    if limit is not None:
        discovered = discovered[:limit]

    stats = {
        "generated_at": datetime.now().isoformat(),
        "source_directory": str(STOCKS_DIR),
        "symbols_discovered": len(discovered),
        "etf_files_present": len(etf_symbols),
        "etf_policy": (
            "ETFs excluded by the canonical Phase 2 universe "
            "(symbol_mapper.get_supported_stocks -> stocks only)"
        ),
        "exclusion_thresholds": {
            "min_raw_rows": MIN_RAW_ROWS,
            "min_valid_rows": MIN_VALID_ROWS,
            "min_feature_rows": MIN_FEATURE_ROWS,
        },
        "symbols_processed": 0,
        "excluded": [],
        "raw_rows_total": 0,
        "valid_rows_before_warmup": 0,
        "warmup_rows_dropped": 0,
        "output_rows": 0,
        "target_counts": {str(k): 0 for k in (UNKNOWN_TARGET, 0, 1, 2)},
        "sequence_blocks": 0,
        "duplicate_dates_in_raw": 0,
        "duplicate_dates_in_output": 0,
        "nonfinite_feature_cells": 0,
        "invalid_or_nonpositive_rows_dropped": 0,
        "output_file": str(output),
    }

    t0 = time.perf_counter()
    peak_rss = _rss_mb()
    first_write = True
    # One long-lived handle: avoids reopening a multi-GB file 5,884 times.
    handle = open(output, "w", encoding="utf-8", newline="")

    for i, symbol in enumerate(discovered, start=1):
        path = STOCKS_DIR / f"{symbol}.csv"
        if not path.is_file():
            stats["excluded"].append({
                "symbol": symbol, "reason": "raw_file_missing", "detail": str(path)
            })
            continue
        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # unreadable / corrupt file
            stats["excluded"].append({
                "symbol": symbol, "reason": "read_error", "detail": str(exc)[:200]
            })
            continue

        raw_rows = len(raw)
        stats["raw_rows_total"] += raw_rows
        raw_dates = pd.to_datetime(raw[DATE_COLUMN], errors="coerce")
        stats["duplicate_dates_in_raw"] += int(raw_dates.duplicated().sum())

        if raw_rows < MIN_RAW_ROWS:
            stats["excluded"].append({
                "symbol": symbol, "reason": "below_min_raw_rows",
                "detail": f"{raw_rows} < {MIN_RAW_ROWS}",
            })
            del raw, raw_dates
            continue

        frame = prepare_symbol_frame(raw, symbol=str(symbol))
        valid_rows = len(frame)
        stats["invalid_or_nonpositive_rows_dropped"] += raw_rows - valid_rows
        if valid_rows < MIN_VALID_ROWS:
            stats["excluded"].append({
                "symbol": symbol, "reason": "insufficient_valid_rows",
                "detail": f"{valid_rows} < {MIN_VALID_ROWS}",
            })
            del raw, raw_dates, frame
            continue

        frame = compute_technical_features(frame)
        frame = add_next_session_target(frame)
        stats["valid_rows_before_warmup"] += len(frame)
        before = len(frame)
        frame = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        stats["warmup_rows_dropped"] += before - len(frame)

        if len(frame) < MIN_FEATURE_ROWS:
            stats["excluded"].append({
                "symbol": symbol, "reason": "insufficient_feature_rows_for_sequence",
                "detail": f"{len(frame)} < {MIN_FEATURE_ROWS}",
            })
            del raw, raw_dates, frame
            continue

        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
        counts = frame[TARGET_COLUMN].value_counts().to_dict()
        for label in (UNKNOWN_TARGET, 0, 1, 2):
            stats["target_counts"][str(label)] += int(counts.get(label, 0))
        stats["output_rows"] += len(frame)
        stats["sequence_blocks"] += int(frame["Sequence_Block"].nunique())
        stats["duplicate_dates_in_output"] += int(frame[DATE_COLUMN].duplicated().sum())
        stats["nonfinite_feature_cells"] += int(
            (~np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64))).sum()
        )
        stats["symbols_processed"] += 1

        frame.to_csv(handle, header=first_write, index=False)
        first_write = False
        del raw, raw_dates, frame, counts

        peak_rss = max(peak_rss, _rss_mb())
        if i % 250 == 0 or i == len(discovered):
            logger.info(
                "progress %d/%d symbols | processed=%d | rows=%d | rss=%.0f MB | elapsed=%.0fs",
                i, len(discovered), stats["symbols_processed"],
                stats["output_rows"], peak_rss, time.perf_counter() - t0,
            )

    handle.close()
    stats["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    stats["peak_rss_mb"] = round(peak_rss, 1)
    stats["output_size_mb"] = round(output.stat().st_size / (1024 ** 2), 1)
    stats["feature_columns"] = list(FEATURE_COLUMNS)
    stats["sequence_length"] = SEQUENCE_LENGTH

    with open(STATS_FILE, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=str)

    logger.info("=" * 64)
    logger.info("REGENERATION COMPLETE")
    logger.info("Discovered=%d  Processed=%d  Excluded=%d",
                stats["symbols_discovered"], stats["symbols_processed"],
                len(stats["excluded"]))
    logger.info("Output rows=%d  size=%.1f MB  elapsed=%.1fs  peak RSS=%.0f MB",
                stats["output_rows"], stats["output_size_mb"],
                stats["elapsed_seconds"], peak_rss)
    logger.info("Targets: %s", stats["target_counts"])
    logger.info("Output: %s", output)
    logger.info("=" * 64)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N discovered symbols (smoke/regeneration check)",
    )
    args = parser.parse_args()
    regenerate(args.output, limit=args.limit)


if __name__ == "__main__":
    main()

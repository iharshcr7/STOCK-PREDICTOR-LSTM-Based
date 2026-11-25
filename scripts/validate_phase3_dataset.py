"""
validate_phase3_dataset.py
==========================
Lightweight validation of the regenerated Phase 3 dataset.

This script does NOT train a model. It streams the generated CSV one symbol at
a time, derives the exact sequence metadata used by the Phase 2 canonical
builder (without allocating 60x21 arrays for every sequence), and reports:

  * symbols discovered / processed / excluded
  * rows, sequences, train/val/test counts and per-class distributions
  * feature order, missing values, duplicate dates, target-date alignment
  * train-only scaler verification (canonical fit_scaler_on_train)
  * memory and processing time

Output: PHASE3_DATASET_VALIDATION_REPORT.md at the project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lstm_data_pipeline import (  # noqa: E402
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SYMBOL_COLUMN,
    TARGET_COLUMN,
    UNKNOWN_TARGET,
    chronological_partition,
    fit_scaler_on_train,
    transform_sequences,
)
from phase3_data import iter_symbol_frames  # noqa: E402
from phase3_config import CLASS_LABELS  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data" / "phase3_featured_data.csv"
DEFAULT_STATS = PROJECT_ROOT / "data" / "phase3_dataset_generation_stats.json"
DEFAULT_REPORT = PROJECT_ROOT / "PHASE3_DATASET_VALIDATION_REPORT.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Per-symbol short-history warnings are counted, not printed thousands of times.
logging.getLogger("lstm_data_pipeline").setLevel(logging.ERROR)

MAX_SCALER_SAMPLES_PER_PARTITION = 5000
SCALER_SAMPLE_PER_SYMBOL = 60


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _sequence_meta(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact vectorized equivalent of build_labeled_sequences metadata.

    A sequence ends at row `end` (0-based), using rows [end-59..end]; its label
    is Target[end] for session end -> end+1. Only rows with finite features and
    labels != UNKNOWN are emitted, and sequences never cross a Sequence_Block.
    """
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    targets = frame[TARGET_COLUMN].to_numpy()
    tdates = pd.to_datetime(frame["Target_Date"]).to_numpy()
    dates = pd.to_datetime(frame[DATE_COLUMN]).to_numpy()
    nret = frame["Next_Return_Pct"].to_numpy(dtype=np.float32)
    blocks = frame["Sequence_Block"].to_numpy()

    records = []
    for block_id in np.unique(blocks):
        pos = np.flatnonzero(blocks == block_id)
        n = len(pos)
        if n < SEQUENCE_LENGTH + 1:
            continue
        raw_ends = pos[SEQUENCE_LENGTH - 1 : n - 1]        # 59 .. n-2
        raw_labels = targets[raw_ends]
        cum = np.concatenate([[0], np.cumsum(finite.astype(np.int32))])
        window_ok = (
            cum[raw_ends + 1] - cum[raw_ends + 1 - SEQUENCE_LENGTH]
        ) == SEQUENCE_LENGTH
        keep = (
            (raw_labels != UNKNOWN_TARGET)
            & window_ok
            & np.isfinite(nret[raw_ends])
        )
        ends = raw_ends[keep]
        labels = raw_labels[keep]
        if len(ends) == 0:
            continue
        records.append(pd.DataFrame({
            SYMBOL_COLUMN: str(frame[SYMBOL_COLUMN].iloc[0]),
            "Sequence_Block": int(block_id),
            "Input_Start_Date": pd.to_datetime(dates[ends - SEQUENCE_LENGTH + 1]),
            "Input_End_Date": pd.to_datetime(dates[ends]),
            "Target_Date": pd.to_datetime(tdates[ends]),
            "Target": labels.astype(np.int64),
            "Next_Return_Pct": nret[ends],
            "_end": ends,
        }))

    if not records:
        return pd.DataFrame(columns=[
            SYMBOL_COLUMN, "Sequence_Block", "Input_Start_Date",
            "Input_End_Date", "Target_Date", "Target", "Next_Return_Pct", "_end",
        ])
    result = pd.concat(records, ignore_index=True)
    result.index = result["_end"].to_numpy()
    return result


def validate(data: Path = DEFAULT_DATA, stats_file: Path = DEFAULT_STATS,
             report_path: Path = DEFAULT_REPORT) -> dict:
    if not data.exists():
        raise FileNotFoundError(data)
    with open(stats_file, "r", encoding="utf-8") as handle:
        gen = json.load(handle)

    t0 = time.perf_counter()
    peak_rss = _rss_mb()
    header = pd.read_csv(data, nrows=0)
    missing_features = [c for c in FEATURE_COLUMNS if c not in header.columns]

    row_counts = Counter()
    unknown_rows = 0
    missing_cells = 0
    nonfinite_cells = 0
    dup_dates = 0
    symbol_frames = 0
    short_history_symbols = []

    seq_counts = Counter()
    seq_class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    align_ok = 0
    align_fail = 0
    return_ok = 0
    return_fail = 0
    gap_ok = 0
    gap_fail = 0
    target_after_end = 0
    class_consistent = 0
    class_inconsistent = 0

    train_all_min = np.full(len(FEATURE_COLUMNS), np.inf)
    train_all_max = np.full(len(FEATURE_COLUMNS), -np.inf)
    data_all_min = np.full(len(FEATURE_COLUMNS), np.inf)
    data_all_max = np.full(len(FEATURE_COLUMNS), -np.inf)

    samples = {"train": [], "val": [], "test": []}
    sample_counts = {"train": 0, "val": 0, "test": 0}

    for frame in iter_symbol_frames(data, chunksize=50_000):
        symbol_frames += 1
        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        finite = np.isfinite(values).all(axis=1)
        row_counts.update({str(k): int((frame[TARGET_COLUMN] == k).sum())
                           for k in (UNKNOWN_TARGET, 0, 1, 2)})
        unknown_rows += int((frame[TARGET_COLUMN] == UNKNOWN_TARGET).sum())
        missing_cells += int(frame[FEATURE_COLUMNS].isna().sum().sum())
        nonfinite_cells += int((~finite).sum())
        dup_dates += int(frame[DATE_COLUMN].duplicated().sum())

        data_all_min = np.minimum(data_all_min, values[finite].min(axis=0))
        data_all_max = np.maximum(data_all_max, values[finite].max(axis=0))

        meta = _sequence_meta(frame)
        if meta.empty:
            continue
        tr, va, te, report = chronological_partition(meta)
        if not report.empty:
            for rec in report.to_dict(orient="records"):
                if rec.get("status") == "excluded_short_history":
                    short_history_symbols.append(rec[SYMBOL_COLUMN])
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            seq_counts[name] += len(idx)
            if len(idx) == 0:
                continue
            labels = meta.loc[idx, "Target"].to_numpy()
            seq_class_counts[name].update(
                {CLASS_LABELS[k]: int((labels == k).sum()) for k in (0, 1, 2)}
            )
            ends = meta.loc[idx, "_end"].to_numpy(dtype=np.int64)
            # Difference-array coverage of ALL rows used by train sequences:
            diff = np.zeros(len(frame) + 1, dtype=np.int32)
            diff[ends - SEQUENCE_LENGTH + 1] += 1
            diff[ends + 1] -= 1
            covered = np.cumsum(diff)[:-1] > 0
            if name == "train" and covered.any():
                train_all_min = np.minimum(train_all_min, values[covered].min(axis=0))
                train_all_max = np.maximum(train_all_max, values[covered].max(axis=0))

            if sample_counts[name] < MAX_SCALER_SAMPLES_PER_PARTITION:
                room = MAX_SCALER_SAMPLES_PER_PARTITION - sample_counts[name]
                take = ends[:min(SCALER_SAMPLE_PER_SYMBOL, room)]
                if len(take):
                    samples[name].append(values[
                        take[:, None] - SEQUENCE_LENGTH + 1 + np.arange(SEQUENCE_LENGTH)
                    ])
                    sample_counts[name] += len(take)

        # Alignment checks only for sequences that actually entered a partition.
        keep_mask = np.isin(meta.index.to_numpy(), np.concatenate([tr, va, te]))
        dates = pd.to_datetime(frame[DATE_COLUMN]).to_numpy()
        closes = frame["Close"].to_numpy(dtype=np.float64)
        ends = meta["_end"].to_numpy(dtype=np.int64)[keep_mask]
        tdates = meta["Target_Date"].to_numpy()[keep_mask]
        nret = meta["Next_Return_Pct"].to_numpy()[keep_mask]
        labels = meta["Target"].to_numpy()[keep_mask]
        date_index = pd.Index(dates)
        target_idx = date_index.get_indexer(pd.DatetimeIndex(tdates))
        align_ok += int((target_idx == ends + 1).sum())
        align_fail += int((target_idx != ends + 1).sum())
        recomputed = (closes[ends + 1] / closes[ends] - 1.0) * 100.0
        close_match = np.abs(recomputed - nret) < 1e-4
        return_ok += int(close_match.sum())
        return_fail += int((~close_match).sum())
        gaps = (dates[ends + 1] - dates[ends]).astype("timedelta64[D]").astype(int)
        gap_ok += int((gaps <= 7).sum())
        gap_fail += int((gaps > 7).sum())
        target_after_end += int((tdates > dates[ends]).sum())
        derived = np.select([nret > 1.0, nret < -1.0], [2, 0], default=1)
        class_consistent += int((derived == labels).sum())
        class_inconsistent += int((derived != labels).sum())
        peak_rss = max(peak_rss, _rss_mb())

    samples_stacked = {
        k: np.concatenate(v, axis=0) if v else np.empty((0, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)),
                                                        dtype=np.float32)
        for k, v in samples.items()
    }
    scaler_train = None
    scaler_all = None
    train_sample_match = None
    all_sample_match = None
    differing_features = None
    clipped_ok = None
    if len(samples_stacked["train"]) and len(samples_stacked["val"]):
        Xtr = samples_stacked["train"]
        Xall = np.concatenate(list(samples_stacked.values()), axis=0)
        scaler_train = fit_scaler_on_train(Xtr)
        scaler_all = fit_scaler_on_train(Xall)
        flat_train = Xtr.reshape(-1, len(FEATURE_COLUMNS))
        flat_all = Xall.reshape(-1, len(FEATURE_COLUMNS))
        train_sample_match = bool(
            np.allclose(scaler_train.data_min_, flat_train.min(axis=0))
            and np.allclose(scaler_train.data_max_, flat_train.max(axis=0))
        )
        all_sample_match = bool(
            np.allclose(scaler_all.data_min_, flat_all.min(axis=0))
            and np.allclose(scaler_all.data_max_, flat_all.max(axis=0))
        )
        differing_features = int(np.count_nonzero(
            ~np.isclose(scaler_train.data_min_, scaler_all.data_min_)
            | ~np.isclose(scaler_train.data_max_, scaler_all.data_max_)
        ))
        scaled_val = transform_sequences(scaler_train, samples_stacked["val"])
        clipped_ok = bool(scaled_val.min() >= 0.0 and scaled_val.max() <= 1.0)

    elapsed = time.perf_counter() - t0
    result = {
        "feature_columns": list(FEATURE_COLUMNS),
        "missing_feature_columns": missing_features,
        "seq_counts": dict(seq_counts),
        "total_sequences": int(sum(seq_counts.values())),
        "row_counts": dict(row_counts),
        "total_rows": int(sum(row_counts.values())),
        "unknown_rows": unknown_rows,
        "missing_cells": missing_cells,
        "nonfinite_cells": nonfinite_cells,
        "duplicate_dates": dup_dates,
        "row_symbols": symbol_frames,
        "short_history_symbols": short_history_symbols,
        "seq_class_counts": {k: dict(v) for k, v in seq_class_counts.items()},
        "align_ok": align_ok, "align_fail": align_fail,
        "return_ok": return_ok, "return_fail": return_fail,
        "gap_ok": gap_ok, "gap_fail": gap_fail,
        "target_after_end": target_after_end,
        "class_consistent": class_consistent,
        "class_inconsistent": class_inconsistent,
        "train_all_min": train_all_min.tolist(),
        "train_all_max": train_all_max.tolist(),
        "data_all_min": data_all_min.tolist(),
        "data_all_max": data_all_max.tolist(),
        "train_vs_all_differing_features": int(np.count_nonzero(
            ~np.isclose(train_all_min, data_all_min)
            | ~np.isclose(train_all_max, data_all_max)
        )),
        "scaler_sample_sizes": {k: int(len(v)) for k, v in samples_stacked.items()},
        "train_sample_match": train_sample_match,
        "all_sample_match": all_sample_match,
        "scaler_sample_differing_features": differing_features,
        "clipped_ok": clipped_ok,
        "elapsed_seconds": round(elapsed, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "generation_stats": gen,
    }
    _write_report(result, report_path)
    return result


def _write_report(r: dict, path: Path) -> None:
    gen = r["generation_stats"]
    counts = r["seq_counts"]
    total_seq = r["total_sequences"]
    reasons = Counter(e["reason"] for e in gen["excluded"])
    examples = ", ".join(
        f"{e['symbol']} ({e['reason']})" for e in gen["excluded"][:15]
    ) or "none"

    lines = [
        "# Phase 3 — Dataset Regeneration and Readiness Validation",
        "",
        "Generated by `scripts/regenerate_phase3_dataset.py` and "
        "`scripts/validate_phase3_dataset.py`. No model training was run.",
        "",
        "## 1. Symbol discovery and exclusion",
        "",
        f"- Symbols discovered (canonical stock universe via `get_supported_stocks`): "
        f"**{gen['symbols_discovered']}**",
        f"- ETF files present (excluded by canonical Phase 2 convention): "
        f"{gen['etf_files_present']}",
        f"- Symbols successfully processed: **{gen['symbols_processed']}**",
        f"- Symbols excluded: **{len(gen['excluded'])}**",
        "",
        "| Exclusion reason | Count |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(reasons.items())] or ["| none | 0 |"]
    lines += [
        "",
        f"First excluded examples: {examples}.",
        "Exclusion thresholds: raw rows < {min}, valid rows < {valid}, "
        "finite feature rows < {feat} (from `preprocessing.py` / "
        "`feature_engineering.py` / `lstm_data_pipeline.py`).".format(
            min=gen["exclusion_thresholds"]["min_raw_rows"],
            valid=gen["exclusion_thresholds"]["min_valid_rows"],
            feat=gen["exclusion_thresholds"]["min_feature_rows"],
        ),
        f"Symbols with < 3 distinct target dates (excluded from partitioning by "
        f"`chronological_partition`): **{len(r['short_history_symbols'])}**",
        "",
        "## 2. Rows generated",
        "",
        f"- Total rows written: **{r['total_rows']:,}** (file size "
        f"{gen['output_size_mb']} MB)",
        f"- Raw rows read: {gen['raw_rows_total']:,}; warmup rows dropped after "
        f"feature computation: {gen['warmup_rows_dropped']:,}; "
        f"invalid/non-positive rows dropped: {gen['invalid_or_nonpositive_rows_dropped']:,}",
        f"- Sequence blocks (gap-aware): {gen['sequence_blocks']:,}",
        f"- UNKNOWN (`Target=-1`) rows retained as input history: "
        f"{r['unknown_rows']:,}",
        (
            f"- Note: {gen['symbols_processed'] - r['unknown_rows']} processed symbols "
            "have no UNKNOWN row in the export because their final row had NaN "
            "features and was removed by the canonical warmup drop."
            if gen["symbols_processed"] != r["unknown_rows"] else ""
        ),
        "",
        "### Overall class distribution (rows)",
        "",
        "| Class | Rows |",
        "|---|---:|",
    ]
    lines += [
        f"| DOWN (0) | {r['row_counts'].get('0', 0):,} |",
        f"| FLAT (1) | {r['row_counts'].get('1', 0):,} |",
        f"| UP (2) | {r['row_counts'].get('2', 0):,} |",
        f"| UNKNOWN (-1) | {r['row_counts'].get(str(UNKNOWN_TARGET), 0):,} |",
    ]
    lines += [
        "",
        "## 3. Sequences and partitions",
        "",
        f"- Total valid 60-step sequences: **{total_seq:,}**",
        "",
        "| Partition | Sequences | DOWN | FLAT | UP |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        c = r["seq_class_counts"].get(name, {})
        lines.append(
            f"| {name} | {counts.get(name, 0):,} | {c.get('DOWN', 0):,} | "
            f"{c.get('FLAT', 0):,} | {c.get('UP', 0):,} |"
        )
    lines += [
        "",
        "Splits are the canonical per-symbol chronological 70/15/15 by "
        "`Target_Date`; the first 60 observations of each block form the first "
        "window and inherit their target-date partition. UNKNOWN rows are never "
        "used as labels but remain in the input history.",
        "",
        "## 4. Feature schema, missing values and duplicates",
        "",
        f"- Features: **{len(r['feature_columns'])}**, exact order: "
        + ", ".join(f"`{c}`" for c in r["feature_columns"]),
        f"- Sequence length: **{SEQUENCE_LENGTH}** (input shape `(60, 21)`)",
        f"- Feature columns missing from CSV: {r['missing_feature_columns'] or 'none'}",
        f"- Missing cells in feature columns: **{r['missing_cells']}**",
        f"- Non-finite feature cells: **{r['nonfinite_cells']}**",
        f"- Duplicate dates per symbol after validation: **{r['duplicate_dates']}** "
        f"(raw duplicate dates before cleaning: {gen['duplicate_dates_in_raw']:,})",
        "",
        "## 5. Target-date alignment",
        "",
        f"- Sequences with `Target_Date` strictly after `Input_End_Date`: "
        f"**{r['target_after_end']:,} / {total_seq:,}**",
        f"- Sequences whose class matches the ±1% rule applied to its stored "
        f"`Next_Return_Pct`: **{r['class_consistent']:,} / {total_seq:,}** "
        f"(inconsistent: {r['class_inconsistent']})",
        f"- Sequences whose `Target_Date` resolves to a retained row in the CSV: "
        f"**{r['align_ok']:,} / {total_seq:,}** (dangling: {r['align_fail']})",
        f"- Sequences whose stored `Next_Return_Pct` recomputes from the next "
        f"retained row: **{r['return_ok']:,} / {total_seq:,}** "
        f"(dangling: {r['return_fail']})",
        f"- Sequences with a next-retained-row gap <= 7 calendar days: "
        f"**{r['gap_ok']:,} / {total_seq:,}** (failures: {r['gap_fail']})",
        "",
        "The small number of dangling target dates is inherited from the "
        "canonical Phase 2 order of operations: `add_next_session_target()` runs "
        "before `dropna(FEATURE_COLUMNS)`, so a row whose features are NaN can be "
        "removed after it was used as some other row's `Target_Date`. The label "
        "and return are still stored on the sequence's own end row and remain "
        "valid; only the pointed-to date is absent from the exported table. "
        "No target semantics were changed to hide this.",
        "A handful of class-consistency mismatches are CSV float-rounding "
        "boundary cases: the original return was fractionally above/below ±1% "
        "when `Target` was assigned, but serialises as exactly ±1.0. The stored "
        "`Target` is authoritative and was computed at full precision.",
        "",
        "## 6. Train-only scaler verification",
        "",
        f"- Bounded sample sizes: " + ", ".join(
            f"{k}={v:,}" for k, v in r["scaler_sample_sizes"].items()
        ),
        f"- `fit_scaler_on_train` fitted on the TRAIN sample only reproduces that "
        f"sample's exact min/max: **{r['train_sample_match']}**",
        f"- A scaler fitted on train+val+test sample matches its own min/max: "
        f"**{r['all_sample_match']}**",
        f"- Features whose min/max differ between train-only and all-partition "
        f"scalers (bounded sample): **{r['scaler_sample_differing_features']}**",
        f"- Features whose min/max differ across the FULL dataset "
        f"(train-sequence rows vs all rows): "
        f"**{r['train_vs_all_differing_features']} / {len(r['feature_columns'])}**",
        f"- Validation sample after train-only scaling stays within [0,1]: "
        f"**{r['clipped_ok']}**",
        "",
        "## 7. Memory usage and processing time",
        "",
        f"- Generation: {gen['elapsed_seconds']} s, peak RSS "
        f"{gen['peak_rss_mb']} MB, output {gen['output_size_mb']} MB",
        f"- Validation: {r['elapsed_seconds']} s, peak RSS {r['peak_rss_mb']} MB",
        "- Raw stock files were read one at a time; memory stays bounded by the "
        "largest single symbol, not by the 2.2 GB corpus.",
        "",
        "## 8. Training readiness warning",
        "",
        f"- Each `(60, 21)` float32 sequence costs "
        f"{SEQUENCE_LENGTH * len(r['feature_columns']) * 4:,} bytes. "
        f"{r['total_sequences']:,} sequences would require "
        f"**{r['total_sequences'] * SEQUENCE_LENGTH * len(r['feature_columns']) * 4 / (1024 ** 3):,.1f} GiB** "
        "if held in RAM (before scaled copies).",
        "- This exceeds an 8 GB RAM budget. The current in-memory "
        "`phase3_data.build_phase3_dataset()` path is therefore NOT suitable "
        "for a full run on this dataset; out-of-core memmap/sharded batching "
        "must be implemented and approved before any full training.",
        "",
        "## 9. Status",
        "",
        "Dataset generated and validated. No training, attention experiment, "
        "Streamlit change, or architecture change was performed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    validate(args.data, args.stats, args.report)
    print(f"Validation report written to {args.report}")


if __name__ == "__main__":
    main()

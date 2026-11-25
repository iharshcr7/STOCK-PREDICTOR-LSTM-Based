"""
benchmark_phase3_5_loader.py
============================
Bounded Phase 3.5 benchmark: 2-5 real symbols, limited batches, NO training.

The operation is intentionally explicit:

  1. Check free disk space.
  2. Stream the 10.62 GB featured CSV once and write a SMALL subset containing
     only the selected symbols (no full-copy, bounded memory).
  3. Calibrate the train-only scaler on that subset.
  4. Iterate a fixed number of train batches (shuffled) and val batches
     (chronological), measuring throughput/RAM/CPU/disk I/O.
  5. Run ONE forward pass through the existing baseline model (no weight
     updates) to confirm input compatibility.
  6. Extrapolate full-corpus epoch durations from the measured throughput.

Test partition is never touched. No model checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase3_5_loader import (  # noqa: E402
    calibrate_scaler,
    create_symbol_subset,
    make_tf_dataset,
)
from phase3_config import FEATURED_DATA_FILE  # noqa: E402

SOURCE = PROJECT_ROOT / "data" / "phase3_featured_data.csv"
SUBSET = PROJECT_ROOT / "data" / "phase3_5_benchmark_subset.csv"
RESULTS = PROJECT_ROOT / "data" / "phase3_5_benchmark_results.json"

FULL_TRAIN_SEQUENCES = 15_042_255
FULL_VAL_SEQUENCES = 3_223_890
BYTES_PER_SEQUENCE = 60 * 21 * 4


def _rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024 ** 2)


def _cpu_seconds(proc: psutil.Process) -> float:
    t = proc.cpu_times()
    return float(t.user + t.system)


def measure(dataset, n_batches: int, proc: psutil.Process) -> dict:
    """Iterate at most n_batches and return measured loader metrics."""
    peak = _rss_mb(proc)
    cpu0 = _cpu_seconds(proc)
    io0 = psutil.disk_io_counters()
    t0 = time.perf_counter()
    count = 0
    samples = 0
    shapes = None
    label_shapes = None
    for x, y in dataset:
        count += 1
        samples += int(x.shape[0])
        if shapes is None:
            shapes = list(x.shape)
            label_shapes = {
                "classification": list(y["classification"].shape),
                "regression": list(y["regression"].shape),
            }
        peak = max(peak, _rss_mb(proc))
        if count >= n_batches:
            break
    wall = time.perf_counter() - t0
    io1 = psutil.disk_io_counters()
    cpu1 = _cpu_seconds(proc)
    return {
        "batches": count,
        "samples": samples,
        "wall_seconds": round(wall, 3),
        "batches_per_second": round(count / wall, 3) if wall > 0 else None,
        "samples_per_second": round(samples / wall, 1) if wall > 0 else None,
        "peak_rss_mb": round(peak, 1),
        "process_cpu_percent": round((cpu1 - cpu0) / wall * 100, 1) if wall > 0 else None,
        "machine_cpu_percent": round(
            (cpu1 - cpu0) / (wall * (os.cpu_count() or 1)) * 100, 1
        ) if wall > 0 else None,
        "disk_read_mb": round((io1.read_bytes - io0.read_bytes) / (1024 ** 2), 1)
        if io0 and io1 else None,
        "batch_input_shape": shapes,
        "label_shapes": label_shapes,
    }


def run(symbols: list[str], batches: int, batch_size: int, seed: int,
        source: Path, subset_path: Path) -> dict:
    proc = psutil.Process()
    free_gb = shutil.disk_usage(source.parent).free / (1024 ** 3)
    print(f"[bench] source={source}")
    print(f"[bench] free disk space = {free_gb:.1f} GB")
    print(f"[bench] subset symbols = {symbols}")
    if free_gb < 5:
        raise RuntimeError("Need at least 5 GB free before creating a benchmark subset")

    t0 = time.perf_counter()
    if not subset_path.exists():
        print("[bench] streaming source ONCE to create a small symbol subset "
              "(no full CSV copy)...")
        subset_info = create_symbol_subset(source, subset_path, set(symbols))
    else:
        subset_info = {
            "destination": str(subset_path),
            "size_mb": round(subset_path.stat().st_size / (1024 ** 2), 2),
            "reused_existing": True,
        }
    subset_info["creation_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[bench] subset: {subset_info['size_mb']} MB "
          f"({subset_info['creation_seconds']} s)")

    print("[bench] calibrating train-only scaler on subset (pass 1)...")
    scaler, stats = calibrate_scaler(subset_path)
    print(f"[bench] subset sequence counts: {stats['sequence_counts']}")

    train_ds = make_tf_dataset(
        subset_path, "train", scaler, batch_size=batch_size,
        shuffle=True, seed=seed, reshuffle_each_iteration=False,
    )
    val_ds = make_tf_dataset(
        subset_path, "val", scaler, batch_size=batch_size, shuffle=False,
    )

    print(f"[bench] measuring {batches} train batches (batch_size={batch_size})...")
    train = measure(train_ds, batches, proc)
    print(f"[bench] train: {train['batches_per_second']} batches/s, "
          f"{train['samples_per_second']} samples/s, "
          f"peak RSS {train['peak_rss_mb']} MB")

    val_batches = max(1, batches // 4)
    print(f"[bench] measuring {val_batches} val batches (chronological, no shuffle)...")
    val = measure(val_ds, val_batches, proc)

    print("[bench] single forward pass (no training, no weight updates)...")
    from phase3_model import build_and_compile_model

    model = build_and_compile_model("baseline")
    x, _ = next(iter(train_ds))
    t1 = time.perf_counter()
    out = model(x, training=False)
    fwd_seconds = time.perf_counter() - t1
    forward = {
        "batch_input_shape": list(x.shape),
        "classification_output_shape": list(out["classification"].shape),
        "regression_output_shape": list(out["regression"].shape),
        "softmax_row_sum_max_error": float(
            np.abs(out["classification"].numpy().sum(axis=1) - 1.0).max()
        ),
        "forward_pass_seconds": round(fwd_seconds, 3),
        "training_performed": False,
    }
    print(f"[bench] forward pass OK: {forward['classification_output_shape']} / "
          f"{forward['regression_output_shape']}")

    rate = train["batches_per_second"]
    batches_per_epoch = int(np.ceil(FULL_TRAIN_SEQUENCES / batch_size))
    epoch_s = batches_per_epoch / rate
    val_batches_per_epoch = int(np.ceil(FULL_VAL_SEQUENCES / batch_size))
    val_s = val_batches_per_epoch / val["batches_per_second"]
    estimates = {
        "note": "ESTIMATES from a 3-symbol subset; not guarantees for the full corpus.",
        "full_train_sequences": FULL_TRAIN_SEQUENCES,
        "batch_size": batch_size,
        "train_batches_per_epoch": batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "estimated_train_seconds_per_epoch": round(epoch_s, 1),
        "estimated_val_seconds_per_epoch": round(val_s, 1),
        "estimated_seconds_5_epochs": round(5 * (epoch_s + val_s), 1),
        "estimated_seconds_20_epochs": round(20 * (epoch_s + val_s), 1),
        "estimated_hours_20_epochs": round(20 * (epoch_s + val_s) / 3600, 2),
        "sequence_bytes": BYTES_PER_SEQUENCE,
        "note_on_ram": (
            "Loader RAM stays bounded by one symbol's rows + shuffle buffer; "
            "add roughly 1 GB for TensorFlow/Keras runtime."
        ),
    }
    print(f"[bench] ESTIMATE: {batches_per_epoch:,} train batches/epoch, "
          f"~{epoch_s / 60:.1f} min/epoch, "
          f"~{estimates['estimated_hours_20_epochs']:.1f} h for 20 epochs")

    results = {
        "generated_at": datetime.now().isoformat(),
        "symbols": symbols,
        "source": str(source),
        "subset": subset_info,
        "seed": seed,
        "batch_size": batch_size,
        "train": train,
        "val": val,
        "forward_pass": forward,
        "estimates": estimates,
        "test_partition_used": False,
        "full_training_performed": False,
        "free_disk_gb_before": round(free_gb, 1),
    }
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[bench] results written to {RESULTS}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--subset", type=Path, default=SUBSET)
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,AMZN")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        batches=args.batches, batch_size=args.batch_size, seed=args.seed,
        source=args.source, subset_path=args.subset,
    )


if __name__ == "__main__":
    main()

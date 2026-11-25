"""
benchmark_phase3_6_training.py
==============================
Phase 3.6 bounded REAL training benchmark.

Executes 50-100 genuine training steps (forward + loss + backprop + Adam update)
on the existing three-symbol Phase 3.5 subset, using the unchanged Phase 3
baseline model and Phase 3.5 streaming loader.

Hard limits:
  * No test partition access.
  * No model/artifact saving of any kind.
  * No architecture, loss, or loss-weight changes.
  * Aborts if RSS exceeds --max-rss-gb (default 6 GB) on an 8 GB machine.
  * Fixed number of batches; no full epoch.

Outputs: data/phase3_6_training_benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase3_5_loader import calibrate_scaler, make_tf_dataset  # noqa: E402
from phase3_model import build_and_compile_model, set_seeds  # noqa: E402
from phase3_evaluation import evaluate_classification, evaluate_regression  # noqa: E402

SUBSET = PROJECT_ROOT / "data" / "phase3_5_benchmark_subset.csv"
RESULTS = PROJECT_ROOT / "data" / "phase3_6_training_benchmark_results.json"
FULL_TRAIN_SEQUENCES = 15_042_255
FULL_VAL_SEQUENCES = 3_223_890


def _rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024 ** 2)


def _cpu_seconds(proc: psutil.Process) -> float:
    t = proc.cpu_times()
    return float(t.user + t.system)


def run(n_batches: int, batch_size: int, seed: int, max_rss_gb: float,
        subset: Path) -> dict:
    set_seeds()
    proc = psutil.Process()
    max_rss_mb = max_rss_gb * 1024

    print(f"[bench36] subset={subset}")
    print(f"[bench36] {n_batches} REAL training batches, batch_size={batch_size}, "
          f"memory guard={max_rss_gb} GB, seed={seed}")
    print("[bench36] test partition is never accessed; no artifacts will be saved.")

    t0 = time.perf_counter()
    scaler, scaler_stats = calibrate_scaler(subset)
    model = build_and_compile_model("baseline")
    train_iter = iter(make_tf_dataset(
        subset, "train", scaler, batch_size=batch_size, shuffle=True,
        seed=seed, reshuffle_each_iteration=False,
    ))
    val_iter = iter(make_tf_dataset(
        subset, "val", scaler, batch_size=batch_size, shuffle=False,
    ))
    print(f"[bench36] subset sequences: {scaler_stats['sequence_counts']} "
          f"(test unused); model params={model.count_params():,}")

    peak_rss = _rss_mb(proc)
    load_times, step_times = [], []
    losses, cls_losses, reg_losses = [], [], []
    cpu0 = _cpu_seconds(proc)
    wall0 = time.perf_counter()
    last_x = None
    completed = 0

    try:
        for step in range(n_batches):
            rss = _rss_mb(proc)
            peak_rss = max(peak_rss, rss)
            if rss > max_rss_mb:
                raise MemoryError(
                    f"RSS {rss / 1024:.2f} GB exceeded guard {max_rss_gb} GB — aborting"
                )

            t_load0 = time.perf_counter()
            x, y = next(train_iter)
            t_load1 = time.perf_counter()
            metrics = model.train_on_batch(x, y, return_dict=True)
            t_step1 = time.perf_counter()

            load_times.append(t_load1 - t_load0)
            step_times.append(t_step1 - t_load1)
            losses.append(float(metrics["loss"]))
            cls_losses.append(float(metrics["classification_loss"]))
            reg_losses.append(float(metrics["regression_loss"]))
            last_x = x
            completed += 1
            if (step + 1) % 10 == 0:
                print(f"[bench36] step {step + 1:3d}/{n_batches} "
                      f"loss={losses[-1]:.4f} (cls={cls_losses[-1]:.4f}, "
                      f"reg={reg_losses[-1]:.4f}) rss={rss / 1024:.2f} GB")
    except StopIteration:
        print("[bench36] WARNING: train stream exhausted before target batch count")

    wall = time.perf_counter() - wall0
    cpu1 = _cpu_seconds(proc)
    total_load = float(np.sum(load_times))
    total_compute = float(np.sum(step_times))

    # Small chronological validation sample: no shuffle, no weight updates.
    val_x, val_y_cls, val_y_reg = [], [], []
    val_metrics = []
    val_load_times, val_test_times = [], []
    val_steps = min(20, max(1, n_batches // 4))
    for _ in range(val_steps):
        tv0 = time.perf_counter()
        xv, yv = next(val_iter)
        tv1 = time.perf_counter()
        m = model.test_on_batch(xv, yv, return_dict=True)
        tv2 = time.perf_counter()
        val_load_times.append(tv1 - tv0)
        val_test_times.append(tv2 - tv1)
        val_metrics.append(m)
        val_x.append(xv.numpy())
        val_y_cls.append(yv["classification"].numpy())
        val_y_reg.append(yv["regression"].numpy())

    Xv = np.concatenate(val_x, axis=0)
    yv_cls = np.concatenate(val_y_cls)
    yv_reg = np.concatenate(val_y_reg)
    out = model(Xv, training=False)
    cls_eval = evaluate_classification(yv_cls, out["classification"].numpy())
    reg_eval = evaluate_regression(yv_reg, out["regression"].numpy().reshape(-1))
    peak_rss = max(peak_rss, _rss_mb(proc))

    batches_per_second = completed / wall if wall > 0 else 0.0
    samples_per_second = completed * batch_size / wall if wall > 0 else 0.0
    val_batches_per_second = val_steps / (sum(val_load_times) + sum(val_test_times))
    train_batches_per_epoch = int(np.ceil(FULL_TRAIN_SEQUENCES / batch_size))
    val_batches_per_epoch = int(np.ceil(FULL_VAL_SEQUENCES / batch_size))
    epoch_s = train_batches_per_epoch / batches_per_second if batches_per_second else None
    val_epoch_s = val_batches_per_epoch / val_batches_per_second if val_batches_per_second else None

    results = {
        "generated_at": datetime.now().isoformat(),
        "stocks": sorted({"AAPL", "MSFT", "AMZN"}),
        "selection_rationale": (
            "Three liquid, long-history large caps from the already-created "
            "Phase 3.5 subset (different sectors); directly comparable to the "
            "loader-only benchmark. Not the full 5,764-symbol universe."
        ),
        "batch_size": batch_size,
        "seed": seed,
        "batches_executed": completed,
        "samples_trained": completed * batch_size,
        "optimizer_updates": completed,
        "training_performed": True,
        "test_partition_used": False,
        "artifacts_saved": False,
        "measured": {
            "wall_seconds": round(wall, 3),
            "training_batches_per_second": round(batches_per_second, 3),
            "training_samples_per_second": round(samples_per_second, 1),
            "total_data_loading_seconds": round(total_load, 3),
            "mean_data_loading_ms_per_batch": round(1000 * total_load / completed, 2),
            "total_forward_backward_seconds": round(total_compute, 3),
            "mean_forward_backward_ms_per_batch": round(1000 * total_compute / completed, 2),
            "data_loading_share_percent": round(100 * total_load / wall, 1),
            "compute_share_percent": round(100 * total_compute / wall, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "process_cpu_percent": round((cpu1 - cpu0) / wall * 100, 1),
            "machine_cpu_percent": round((cpu1 - cpu0) / (wall * (os.cpu_count() or 1)) * 100, 1),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "min_loss": min(losses),
            "loss_every_10": losses[::10],
            "classification_loss_first": cls_losses[0],
            "classification_loss_last": cls_losses[-1],
            "regression_loss_first": reg_losses[0],
            "regression_loss_last": reg_losses[-1],
            "validation_batches_evaluated": val_steps,
            "validation_samples_evaluated": int(len(Xv)),
            "validation_batches_per_second": round(val_batches_per_second, 3),
            "validation_mean_loss": round(
                float(np.mean([m["loss"] for m in val_metrics])), 4
            ),
            "validation_mean_accuracy": round(
                float(np.mean([m["classification_accuracy"] for m in val_metrics])), 4
            ),
            "validation_mean_mae": round(
                float(np.mean([m["regression_mae"] for m in val_metrics])), 4
            ),
            "validation_classification": {
                "accuracy": cls_eval["accuracy"],
                "macro_f1": cls_eval["macro_f1"],
                "confusion_matrix": cls_eval["confusion_matrix"],
            },
            "validation_regression": {
                "mae": reg_eval["mae"], "rmse": reg_eval["rmse"],
                "r2": reg_eval["r2"],
                "directional_agreement": reg_eval["directional_agreement"],
            },
            "input_shape": list(last_x.shape) if last_x is not None else None,
            "classification_label_shape": [batch_size],
            "regression_target_shape": [batch_size],
        },
        "estimates": {
            "note": "EXTRAPOLATED from a 3-symbol subset; not guarantees for full training.",
            "full_train_sequences": FULL_TRAIN_SEQUENCES,
            "train_batches_per_epoch": train_batches_per_epoch,
            "val_batches_per_epoch": val_batches_per_epoch,
            "estimated_train_seconds_per_epoch": round(epoch_s, 1),
            "estimated_val_seconds_per_epoch": round(val_epoch_s, 1),
            "estimated_seconds_5_epochs": round(5 * (epoch_s + val_epoch_s), 1),
            "estimated_seconds_20_epochs": round(20 * (epoch_s + val_epoch_s), 1),
            "estimated_hours_20_epochs": round(20 * (epoch_s + val_epoch_s) / 3600, 2),
        },
    }
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[bench36] DONE: {completed} updates, "
          f"{batches_per_second:.2f} batches/s, "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          f"peak RSS {peak_rss / 1024:.2f} GB")
    print(f"[bench36] ESTIMATE: ~{epoch_s / 3600:.2f} h/epoch, "
          f"~{results['estimates']['estimated_hours_20_epochs']:.1f} h for 20 epochs")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rss-gb", type=float, default=6.0)
    parser.add_argument("--subset", type=Path, default=SUBSET)
    args = parser.parse_args()
    run(args.batches, args.batch_size, args.seed, args.max_rss_gb, args.subset)


if __name__ == "__main__":
    main()

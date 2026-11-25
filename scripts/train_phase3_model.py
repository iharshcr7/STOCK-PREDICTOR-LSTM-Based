"""
train_phase3_model.py
=====================
Entry point for Phase 3 multi-task LSTM training.

IMPORTANT — this script does NOT start a full training run by default.

    python scripts/train_phase3_model.py --mode plan       # default: no training
    python scripts/train_phase3_model.py --mode smoke      # tiny synthetic sample
    python scripts/train_phase3_model.py --mode train --approve-full-training \
        --variant baseline

The ``--approve-full-training`` flag is the explicit approval gate. All output
paths are Phase-3-versioned; no legacy artifact is ever overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_data_pipeline import (
    SEQUENCE_LENGTH,
    add_next_session_target,
    build_labeled_sequences,
    chronological_partition,
    compute_technical_features,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from phase3_artifacts import save_artifacts
from phase3_config import (
    ARCHITECTURE,
    CLASS_LABELS,
    FEATURED_DATA_FILE,
    INPUT_SHAPE,
    LOSS_CONFIG,
    MODELS_DIR,
    PHASE3_VERSION,
    TRAINING_CONFIG,
    artifact_paths,
)
from phase3_data import build_phase3_dataset, estimate_memory, validate_featured_dataset
from phase3_evaluation import evaluate_model
from phase3_model import build_and_compile_model, model_signature, set_seeds


# ---------------------------------------------------------------------------
# Plan mode (no training, no data materialization)
# ---------------------------------------------------------------------------
def print_plan(variant: str) -> dict:
    set_seeds()
    model = build_and_compile_model(variant)
    signature = model_signature(model)

    n_rows = None
    n_symbols = None
    if FEATURED_DATA_FILE.exists():
        head = pd.read_csv(FEATURED_DATA_FILE, usecols=["Symbol"])
        n_rows, n_symbols = len(head), int(head["Symbol"].nunique())

    memory = estimate_memory(max(n_rows or 0, 0))
    print("\n" + "=" * 68)
    print(f"PHASE 3 TRAINING PLAN — {PHASE3_VERSION} / {variant}")
    print("=" * 68)
    print("Input shape          :", signature["input_shape"])
    print("Output shapes        :", signature["output_shapes"])
    print("Parameters           :", signature["total_params"])
    print("Losses               :", signature["loss"])
    print("Loss weights         :", signature["loss_weights"])
    print("Optimizer / LR       :", signature["optimizer"], "/", signature["learning_rate"])
    print("Batch size           :", TRAINING_CONFIG["batch_size"])
    print("Max epochs           :", TRAINING_CONFIG["max_epochs"])
    print("EarlyStopping        : patience", TRAINING_CONFIG["early_stopping_patience"],
          "monitor", TRAINING_CONFIG["monitor"])
    print("ReduceLROnPlateau    : patience", TRAINING_CONFIG["reduce_lr_patience"],
          "factor", TRAINING_CONFIG["reduce_lr_factor"], "min_lr", TRAINING_CONFIG["min_lr"])
    print("Seeds                : python/numpy/tensorflow =", 42)
    print("Split                :", TRAINING_CONFIG["train_frac"], "/",
          TRAINING_CONFIG["val_frac"], "/", TRAINING_CONFIG["test_frac"],
          "(per symbol, chronological, by target date)")
    print("Data rows / symbols  :", n_rows, "/", n_symbols)
    print("Memory estimate      :", memory)
    print("Artifacts            :", artifact_paths(variant)["model"], "(and versioned companions)")
    print("NO TRAINING WAS RUN. Test partition remains untouched.")
    print("=" * 68 + "\n")
    return signature


# ---------------------------------------------------------------------------
# Smoke test (synthetic data only, 2 epochs, nothing written to models/)
# ---------------------------------------------------------------------------
def synthetic_frames(seed_base: int = 0) -> list[pd.DataFrame]:
    """Three symbols with deliberate DOWN / FLAT / UP sessions and gaps."""
    frames = []
    for i, symbol in enumerate(("AAA", "BBB", "CCC")):
        rng = np.random.default_rng(seed_base + i)
        n = 110
        dates = pd.bdate_range("2023-01-02", periods=n)
        base = 100.0 * (i + 1)
        steps = rng.choice([-2.5, -0.2, 0.0, 0.2, 2.5], size=n,
                           p=[0.20, 0.15, 0.20, 0.15, 0.30])
        close = base + np.cumsum(steps)
        raw = pd.DataFrame({
            "Symbol": symbol, "Date": dates,
            "Open": close - 0.1, "High": close + 0.5, "Low": close - 0.5,
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
        })
        frames.append(add_next_session_target(
            compute_technical_features(prepare_symbol_frame(raw))
        ))
    return frames


def run_smoke_test(variant: str = "baseline") -> dict:
    """Tiny forward/backward pass on synthetic data. No full dataset, no test eval."""
    set_seeds()
    frames = synthetic_frames()
    X, y_cls, meta, y_reg = build_labeled_sequences(
        frames, label_mode="three_class", return_regression=True
    )
    tr, va, _te, _report = chronological_partition(meta)
    scaler = fit_scaler_on_train(X[tr])
    X_train = transform_sequences(scaler, X[tr])[:256]
    X_val = transform_sequences(scaler, X[va])[:128]
    y_train_cls, y_train_reg = y_cls[tr][:256], y_reg[tr][:256]
    y_val_cls, y_val_reg = y_cls[va][:128], y_reg[va][:128]

    model = build_and_compile_model(variant)
    print(f"[smoke] variant={variant} X_train={X_train.shape} X_val={X_val.shape} "
          f"classes={sorted(set(y_train_cls.tolist()))}")

    history = model.fit(
        x=X_train,
        y={"classification": y_train_cls, "regression": y_train_reg},
        validation_data=(
            X_val,
            {"classification": y_val_cls, "regression": y_val_reg},
        ),
        epochs=TRAINING_CONFIG["smoke_epochs"],
        batch_size=TRAINING_CONFIG["smoke_batch_size"],
        verbose=0,
    )
    metrics = evaluate_model(model, X_val, y_val_cls, y_val_reg)
    final_loss = float(history.history["loss"][-1])
    print(f"[smoke] final train loss={final_loss:.4f} "
          f"val macro-F1={metrics['classification']['macro_f1']:.4f} "
          f"val MAE={metrics['regression']['mae']:.4f} "
          f"(validation only — test set unused)")

    # Save/load round-trip into a temporary directory, never models/.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="phase3_smoke_") as tmp:
        path = Path(tmp) / "smoke.keras"
        model.save(str(path))
        from tensorflow import keras
        reloaded = keras.models.load_model(str(path))
        out = reloaded.predict(X_val[:4], verbose=0)
        assert out["classification"].shape == (4, len(CLASS_LABELS))
        assert out["regression"].shape == (4, 1)

    result = {
        "variant": variant,
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "epochs": TRAINING_CONFIG["smoke_epochs"],
        "final_train_loss": final_loss,
        "validation_metrics": metrics,
        "save_load_roundtrip": "ok",
    }
    print("[smoke] PASSED\n")
    return result


# ---------------------------------------------------------------------------
# Full training — explicit approval gate
# ---------------------------------------------------------------------------
def build_callbacks(variant: str):
    from tensorflow import keras

    paths = artifact_paths(variant)
    return [
        keras.callbacks.EarlyStopping(
            monitor=TRAINING_CONFIG["monitor"],
            patience=TRAINING_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=TRAINING_CONFIG["monitor"],
            factor=TRAINING_CONFIG["reduce_lr_factor"],
            patience=TRAINING_CONFIG["reduce_lr_patience"],
            min_lr=TRAINING_CONFIG["min_lr"],
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(paths["model"]),
            monitor=TRAINING_CONFIG["monitor"],
            save_best_only=True,
            verbose=1,
        ),
    ]


def run_full_training(variant: str, evaluate_test: bool = False) -> dict:
    validate_featured_dataset(FEATURED_DATA_FILE)
    set_seeds()
    data = build_phase3_dataset(FEATURED_DATA_FILE)
    model = build_and_compile_model(variant)

    print(f"[train] variant={variant} symbols={data['n_symbols_seen']} "
          f"train={len(data['train_idx'])} val={len(data['val_idx'])} "
          f"test={len(data['test_idx'])}")

    history = model.fit(
        x=data["X_train"],
        y={"classification": data["y_train_cls"], "regression": data["y_train_reg"]},
        validation_data=(
            data["X_val"],
            {"classification": data["y_val_cls"], "regression": data["y_val_reg"]},
        ),
        epochs=TRAINING_CONFIG["max_epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        callbacks=build_callbacks(variant),
        verbose=1,
    )

    val_metrics = evaluate_model(
        model, data["X_val"], data["y_val_cls"], data["y_val_reg"]
    )
    training_metadata = {
        "partition_sizes": {
            "train": len(data["train_idx"]),
            "val": len(data["val_idx"]),
            "test": len(data["test_idx"]),
        },
        "n_symbols_seen": data["n_symbols_seen"],
        "class_counts": data["class_counts"],
        "partition_report": data["partition_report"].to_dict(orient="records"),
        "excluded_symbols": data["excluded_symbols"],
        "test_metrics": None,
        "evaluate_test": evaluate_test,
    }
    if evaluate_test:
        training_metadata["test_metrics"] = evaluate_model(
            model, data["X_test"], data["y_test_cls"], data["y_test_reg"]
        )

    paths = save_artifacts(
        model=model,
        variant=variant,
        scaler=data["scaler"],
        feature_columns=_feature_columns(),
        history=history,
        validation_metrics=val_metrics,
        training_metadata=training_metadata,
        sequence_metadata=data["metadata"],
    )
    print(f"[train] saved artifacts: {paths}")
    return {"paths": paths, "validation_metrics": val_metrics}


def _feature_columns() -> list[str]:
    from phase3_config import N_FEATURES
    from lstm_data_pipeline import FEATURE_COLUMNS

    if len(FEATURE_COLUMNS) != N_FEATURES:
        raise ValueError("Feature schema length mismatch")
    return list(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "attention"], default="baseline")
    parser.add_argument("--mode", choices=["plan", "smoke", "train"], default="plan")
    parser.add_argument(
        "--approve-full-training", action="store_true",
        help="REQUIRED to run --mode train. Explicit approval only.",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate the held-out test set after training (default: off).",
    )
    args = parser.parse_args()

    if args.mode == "plan":
        print_plan(args.variant)
    elif args.mode == "smoke":
        run_smoke_test(args.variant)
    elif args.mode == "train":
        if not args.approve_full_training:
            raise SystemExit(
                "REFUSED: full training requires --approve-full-training. "
                "Run --mode plan first and obtain explicit approval."
            )
        run_full_training(args.variant, evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

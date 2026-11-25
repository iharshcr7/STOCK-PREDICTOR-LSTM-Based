"""
Phase 3 lightweight tests — synthetic data only. No full training, no real
dataset, no HDFS/MongoDB/Spark. Run:

    py -3.11 -m unittest tests.test_phase3_model -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from tensorflow import keras  # noqa: E402

from lstm_data_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    add_next_session_target,
    build_labeled_sequences,
    chronological_partition,
    compute_technical_features,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from model_config import FINAL_FEATURE_COLUMNS  # noqa: E402
from phase3_config import (  # noqa: E402
    CLASS_LABELS,
    CLASS_NAMES,
    INPUT_SHAPE,
    LEGACY_ARTIFACTS,
    LOSS_CONFIG,
    N_FEATURES,
    artifact_paths,
)
from phase3_data import estimate_memory  # noqa: E402
from phase3_model import (  # noqa: E402
    TemporalAttention,
    build_and_compile_model,
    build_multitask_lstm,
)


def synthetic_symbol(symbol: str, base: float, n: int = 110, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    steps = rng.choice([-2.5, -0.2, 0.0, 0.2, 2.5], size=n,
                       p=[0.2, 0.15, 0.2, 0.15, 0.3])
    close = base + np.cumsum(steps)
    raw = pd.DataFrame({
        "Symbol": symbol, "Date": dates,
        "Open": close - 0.1, "High": close + 0.5, "Low": close - 0.5,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })
    return add_next_session_target(
        compute_technical_features(prepare_symbol_frame(raw))
    )


class Phase3ModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline = build_and_compile_model("baseline")
        cls.attention = build_and_compile_model("attention")
        cls.rng = np.random.default_rng(2026)
        cls.X = cls.rng.random((9, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)

    # 1 ------------------------------------------------------------------
    def test_input_shape(self):
        for model in (self.baseline, self.attention):
            self.assertEqual(tuple(model.input_shape), (None, 60, 21))
        self.assertEqual(INPUT_SHAPE, (60, 21))
        self.assertEqual(SEQUENCE_LENGTH, 60)
        self.assertEqual(N_FEATURES, 21)

    # 2 ------------------------------------------------------------------
    def test_output_shapes(self):
        for model in (self.baseline, self.attention):
            shapes = {name: tuple(t.shape) for name, t in
                      zip(model.output_names, model.outputs)}
            self.assertEqual(shapes["classification"], (None, 3))
            self.assertEqual(shapes["regression"], (None, 1))
            self.assertEqual(set(shapes), {"classification", "regression"})

    # 3 ------------------------------------------------------------------
    def test_softmax_probabilities_sum_to_one(self):
        out = self.baseline.predict(self.X, verbose=0)
        proba = out["classification"]
        self.assertEqual(proba.shape, (9, 3))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(9), rtol=1e-5)
        self.assertTrue((proba >= 0).all() and (proba <= 1).all())

    # 4 ------------------------------------------------------------------
    def test_three_class_output_mapping(self):
        self.assertEqual(CLASS_LABELS, {0: "DOWN", 1: "FLAT", 2: "UP"})
        self.assertEqual(CLASS_NAMES, ["DOWN", "FLAT", "UP"])
        proba = self.attention.predict(self.X, verbose=0)["classification"]
        self.assertTrue(set(proba.argmax(axis=1).tolist()).issubset({0, 1, 2}))

        # Pipeline encodes the same three classes from a real synthetic frame.
        frame = pd.DataFrame({
            "Symbol": "MAP",
            "Date": pd.bdate_range("2023-01-02", periods=4),
            "Open": [100.0, 98.0, 98.2, 100.5],
            "High": [100.5, 98.5, 98.7, 101.0],
            "Low": [99.5, 97.5, 97.7, 100.0],
            "Close": [100.0, 98.0, 98.2, 100.5],
            "Volume": [1e6] * 4,
        })
        target = add_next_session_target(
            compute_technical_features(prepare_symbol_frame(frame))
        )
        self.assertEqual(target["Target"].tolist(), [0, 1, 2, -1])

    # 5 ------------------------------------------------------------------
    def test_regression_output_shape(self):
        out = self.baseline.predict(self.X, verbose=0)
        self.assertEqual(out["regression"].shape, (9, 1))
        self.assertTrue(np.isfinite(out["regression"]).all())

    # 6 ------------------------------------------------------------------
    def test_model_compilation(self):
        for model in (self.baseline, self.attention):
            self.assertIsNotNone(model.optimizer)
            self.assertIsNotNone(model.compiled_loss)
            self.assertEqual(set(model.loss.keys()),
                             {"classification", "regression"})
            self.assertAlmostEqual(
                float(model.optimizer.learning_rate.numpy()),
                LOSS_CONFIG["learning_rate"], places=8,
            )
        self.assertEqual(
            LOSS_CONFIG["loss_weights"],
            {"classification": 1.0, "regression": 0.25},
        )
        self.assertLess(LOSS_CONFIG["loss_weights"]["regression"],
                        LOSS_CONFIG["loss_weights"]["classification"])

    # 7 ------------------------------------------------------------------
    def test_batch_inference(self):
        batch = self.rng.random((7, 60, 21)).astype(np.float32)
        out = self.baseline.predict(batch, verbose=0)
        self.assertEqual(out["classification"].shape, (7, 3))
        self.assertEqual(out["regression"].shape, (7, 1))
        one = self.baseline.predict(batch[:1], verbose=0)
        np.testing.assert_allclose(
            out["regression"][:1], one["regression"], rtol=1e-5, atol=1e-5
        )

    # 8 ------------------------------------------------------------------
    def test_model_save_load(self):
        for variant, model in (("baseline", self.baseline),
                               ("attention", self.attention)):
            with tempfile.TemporaryDirectory(prefix="phase3_test_") as tmp:
                path = Path(tmp) / f"{variant}.keras"
                model.save(str(path))
                loaded = keras.models.load_model(str(path))
                expected = model.predict(self.X[:3], verbose=0)
                actual = loaded.predict(self.X[:3], verbose=0)
                for key in ("classification", "regression"):
                    np.testing.assert_allclose(
                        expected[key], actual[key], rtol=1e-5, atol=1e-5
                    )
                self.assertEqual(loaded.count_params(), model.count_params())

    # 9 ------------------------------------------------------------------
    def test_feature_schema_compatibility(self):
        self.assertEqual(FEATURE_COLUMNS, FINAL_FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 21)
        forbidden = {"Target", "Target_Date", "Next_Return_Pct", "Future_Return_Pct"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_COLUMNS))
        for variant in ("baseline", "attention"):
            paths = artifact_paths(variant)
            self.assertTrue(all(p.name not in LEGACY_ARTIFACTS
                                for p in paths.values()))
        memory = estimate_memory(1000)
        self.assertEqual(memory["n_sequences"], 1000)
        self.assertGreater(memory["bytes_per_sequence"], 0)

    # 10 -----------------------------------------------------------------
    def test_no_cross_symbol_sequence_mixing(self):
        aaa = synthetic_symbol("AAA", 100.0, seed=11)
        bbb = synthetic_symbol("BBB", 900.0, seed=12)
        X, y, meta, y_reg = build_labeled_sequences(
            [aaa, bbb], label_mode="three_class", return_regression=True
        )
        self.assertEqual(X.shape, (len(meta), 60, 21))
        self.assertEqual(len(y), len(meta))
        self.assertEqual(len(y_reg), len(meta))
        lookup = {"AAA": aaa, "BBB": bbb}
        close_idx = FEATURE_COLUMNS.index("Close")
        for i, row in enumerate(meta.itertuples(index=False)):
            frame = lookup[row.Symbol]
            end_close = frame.loc[frame["Date"] == row.Input_End_Date,
                                  "Close"].iloc[0]
            self.assertAlmostEqual(float(X[i, -1, close_idx]),
                                   float(end_close), places=3)
            start_date = frame.loc[frame["Date"] == row.Input_Start_Date, "Date"]
            self.assertFalse(start_date.empty)

    # 11 -----------------------------------------------------------------
    def test_regression_target_alignment(self):
        aaa = synthetic_symbol("AAA", 100.0, seed=21)
        bbb = synthetic_symbol("BBB", 500.0, seed=22)
        X, y, meta, y_reg = build_labeled_sequences(
            [aaa, bbb], label_mode="three_class", return_regression=True
        )
        lookup = {"AAA": aaa, "BBB": bbb}
        for i, row in enumerate(meta.itertuples(index=False)):
            frame = lookup[row.Symbol]
            match = frame.loc[frame["Date"] == row.Input_End_Date]
            self.assertEqual(len(match), 1)
            self.assertAlmostEqual(float(y_reg[i]),
                                   float(match["Next_Return_Pct"].iloc[0]), places=4)
            self.assertEqual(int(y[i]), int(match["Target"].iloc[0]))
            self.assertFalse(np.isnan(y_reg[i]))
        self.assertTrue((y_reg != 0).any())

    # 12 -----------------------------------------------------------------
    def test_partition_integrity_with_regression(self):
        frames = [synthetic_symbol("AAA", 100.0, seed=31),
                  synthetic_symbol("BBB", 500.0, seed=32)]
        X, y, meta, y_reg = build_labeled_sequences(
            frames, label_mode="three_class", return_regression=True
        )
        tr, va, te, _report = chronological_partition(meta)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(set(va) & set(te)), 0)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(va), 0)
        self.assertGreater(len(te), 0)
        scaler = fit_scaler_on_train(X[tr])
        np.testing.assert_allclose(
            scaler.data_min_, X[tr].reshape(-1, 21).min(axis=0), rtol=1e-5
        )
        scaled = transform_sequences(scaler, X[te])
        self.assertGreaterEqual(float(scaled.min()), 0.0)
        self.assertLessEqual(float(scaled.max()), 1.0)
        self.assertEqual(len(y_reg[tr]), len(tr))

    # 13 -----------------------------------------------------------------
    def test_attention_variant_parity(self):
        self.assertLess(self.attention.count_params(), 100_000)
        self.assertLess(self.baseline.count_params(), 100_000)
        layer = TemporalAttention(score_units=8)
        output = layer(np.zeros((5, 60, 64), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (5, 64))
        # Both builders honour a custom but identical interface.
        for variant in ("baseline", "attention"):
            m = build_multitask_lstm(variant=variant)
            self.assertEqual(tuple(m.input_shape), (None,) + INPUT_SHAPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)

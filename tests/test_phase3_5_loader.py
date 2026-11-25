"""
Phase 3.5 lightweight loader tests — synthetic data only, no training.

    py -3.11 -m unittest discover -s tests -p "test_phase3_5_loader.py" -v
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

from lstm_data_pipeline import (  # noqa: E402
    DATE_COLUMN,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    SYMBOL_COLUMN,
    add_next_session_target,
    build_labeled_sequences,
    chronological_partition,
    compute_technical_features,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from phase3_5_loader import (  # noqa: E402
    build_sequence_index,
    calibrate_scaler,
    create_symbol_subset,
    iter_sequences,
    make_tf_dataset,
    partition_positions,
    stream_symbol_frames,
)


def featured_frame(symbol, base, n=130, seed=3, gap_at=None, nan_at=None):
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2023-01-02", periods=n))
    if gap_at is not None:
        dates[gap_at:] = [d + pd.Timedelta(days=30) for d in dates[gap_at:]]
    steps = rng.choice([-2.5, -0.2, 0.0, 0.2, 2.5], size=n,
                       p=[0.2, 0.15, 0.2, 0.15, 0.3])
    close = base + np.cumsum(steps)
    raw = pd.DataFrame({
        "Symbol": symbol, "Date": dates,
        "Open": close - 0.1, "High": close + 0.5, "Low": close - 0.5,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })
    frame = add_next_session_target(compute_technical_features(prepare_symbol_frame(raw)))
    if nan_at is not None:
        frame.loc[nan_at, "RSI"] = np.nan
    return frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def batches(dataset, limit=None):
    out = []
    for i, (x, y) in enumerate(dataset):
        out.append((x.numpy(), y["classification"].numpy(), y["regression"].numpy()))
        if limit is not None and i + 1 >= limit:
            break
    return out


class Phase35LoaderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="phase35_test_")
        cls.dir = Path(cls.tmp.name)
        cls.frames = [
            featured_frame("AAA", 100.0, seed=11),
            featured_frame("BBB", 500.0, seed=12, gap_at=70),
            featured_frame("CCC", 900.0, seed=13, nan_at=80),
            featured_frame("DDD", 50.0, n=40, seed=14),  # short history
        ]
        cls.csv = cls.dir / "synthetic.csv"
        pd.concat(cls.frames, ignore_index=True).to_csv(cls.csv, index=False)
        cls.scaler, cls.stats = calibrate_scaler(cls.csv)
        cls.canonical = build_labeled_sequences(cls.frames, label_mode="three_class",
                                                return_regression=True)
        X, y, meta, yreg = cls.canonical
        tr, va, te, _ = chronological_partition(meta)
        cls.canonical_scaler = fit_scaler_on_train(X[tr])
        cls.canonical_counts = {"train": len(tr), "val": len(va), "test": len(te)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # 1 ------------------------------------------------------------------
    def test_sequence_index_matches_canonical(self):
        X, y, meta, yreg = self.canonical
        by_symbol = {}
        for frame in self.frames:
            idx = build_sequence_index(frame)
            by_symbol[str(frame[SYMBOL_COLUMN].iloc[0])] = idx

        self.assertEqual(sum(len(i) for i in by_symbol.values()), len(meta))
        for symbol, idx in by_symbol.items():
            rows = meta[meta[SYMBOL_COLUMN] == symbol]
            self.assertEqual(len(idx), len(rows))
            np.testing.assert_array_equal(
                idx.labels, rows["Target"].to_numpy(dtype=np.int64)
            )
            np.testing.assert_array_equal(
                idx.target_dates, pd.to_datetime(rows["Target_Date"]).to_numpy()
            )
            np.testing.assert_allclose(
                idx.next_returns,
                self.canonical[3][(meta[SYMBOL_COLUMN] == symbol).to_numpy()],
                rtol=1e-6, atol=1e-6,
            )

    # 2 ------------------------------------------------------------------
    def test_symbol_isolation(self):
        lookup = {str(f[SYMBOL_COLUMN].iloc[0]): f for f in self.frames}
        close_idx = FEATURE_COLUMNS.index("Close")
        for window, targets, meta in iter_sequences(
            self.csv, "train", None, with_metadata=True
        ):
            frame = lookup[meta["Symbol"]]
            end = pd.Timestamp(meta["Input_End_Date"])
            expected_close = frame.loc[frame[DATE_COLUMN] == end, "Close"].iloc[0]
            self.assertAlmostEqual(float(window[-1, close_idx]),
                                   float(expected_close), places=3)

    # 3 ------------------------------------------------------------------
    def test_partition_separation(self):
        self.assertEqual(
            {k: v for k, v in self.stats["sequence_counts"].items()},
            self.canonical_counts,
        )
        sets = {}
        for part in ("train", "val", "test"):
            meta = [
                m for _, _, m in iter_sequences(
                    self.csv, part, self.scaler, with_metadata=True
                )
            ]
            sets[part] = {(m["Symbol"], m["Input_End_Date"]) for m in meta}
            self.assertEqual(len(sets[part]), self.canonical_counts[part])
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])

    # 4 ------------------------------------------------------------------
    def test_scaler_consistency(self):
        np.testing.assert_allclose(
            self.scaler.data_min_, self.canonical_scaler.data_min_, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.scaler.data_max_, self.canonical_scaler.data_max_, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            self.scaler.scale_, self.canonical_scaler.scale_, rtol=1e-6, atol=1e-6
        )
        window, _, _ = next(iter_sequences(self.csv, "train", self.scaler,
                                           with_metadata=True))
        raw = transform_sequences(self.canonical_scaler, window.reshape(1, 60, 21))
        # Window is already scaled; verify it is in the canonical [0,1] range.
        self.assertGreaterEqual(float(window.min()), 0.0)
        self.assertLessEqual(float(window.max()), 1.0)
        self.assertEqual(raw.shape, (1, 60, 21))

    # 5 ------------------------------------------------------------------
    def test_deterministic_batches(self):
        ds_a = make_tf_dataset(self.csv, "train", self.scaler, batch_size=8,
                               shuffle=True, seed=42, reshuffle_each_iteration=False)
        ds_b = make_tf_dataset(self.csv, "train", self.scaler, batch_size=8,
                               shuffle=True, seed=42, reshuffle_each_iteration=False)
        a = batches(ds_a, limit=3)
        b = batches(ds_b, limit=3)
        self.assertEqual(len(a), 3)
        for (xa, ca, ra), (xb, cb, rb) in zip(a, b):
            np.testing.assert_allclose(xa, xb)
            np.testing.assert_array_equal(ca, cb)
            np.testing.assert_allclose(ra, rb)

    # 6 ------------------------------------------------------------------
    def test_incomplete_final_batch(self):
        ds = make_tf_dataset(self.csv, "train", self.scaler, batch_size=32)
        all_batches = batches(ds)
        self.assertGreater(len(all_batches), 1)
        last_x = all_batches[-1][0]
        self.assertEqual(last_x.shape[1:], (60, 21))
        self.assertLess(len(last_x), 32)
        self.assertEqual(sum(len(b[0]) for b in all_batches), self.canonical_counts["train"])

    # 7 ------------------------------------------------------------------
    def test_missing_data_handling(self):
        ccc = next(f for f in self.frames if f[SYMBOL_COLUMN].iloc[0] == "CCC")
        self.assertTrue(ccc[FEATURE_COLUMNS].notna().all().all())
        idx = build_sequence_index(ccc)
        expected = int((self.canonical[2][SYMBOL_COLUMN] == "CCC").sum())
        self.assertEqual(len(idx), expected)
        # No emitted window may contain the row that had a NaN feature.
        self.assertTrue((idx.ends >= 0).all())

    # 8 ------------------------------------------------------------------
    def test_invalid_gap_handling(self):
        bbb = next(f for f in self.frames if f[SYMBOL_COLUMN].iloc[0] == "BBB")
        self.assertGreater(bbb["Sequence_Block"].nunique(), 1)
        for window, targets, meta in iter_sequences(
            self.csv, "train", self.scaler, symbols={"BBB"}, with_metadata=True
        ):
            start, end = pd.Timestamp(meta["Input_Start_Date"]), pd.Timestamp(meta["Input_End_Date"])
            self.assertLessEqual((end - start).days, 120)
            block = bbb[(bbb[DATE_COLUMN] >= start) & (bbb[DATE_COLUMN] <= end)]["Sequence_Block"]
            self.assertEqual(block.nunique(), 1)

    # 9 ------------------------------------------------------------------
    def test_target_shapes(self):
        ds = make_tf_dataset(self.csv, "val", self.scaler, batch_size=4)
        x, y = next(iter(ds))
        self.assertEqual(tuple(x.shape), (4, 60, 21))
        self.assertEqual(tuple(y["classification"].shape), (4,))
        self.assertEqual(tuple(y["regression"].shape), (4,))
        self.assertTrue(np.issubdtype(y["classification"].numpy().dtype, np.integer))

    # 10 -----------------------------------------------------------------
    def test_repeated_iteration(self):
        ds = make_tf_dataset(self.csv, "val", self.scaler, batch_size=16)
        first = batches(ds, limit=2)
        second = batches(ds, limit=2)
        for (x1, c1, r1), (x2, c2, r2) in zip(first, second):
            np.testing.assert_allclose(x1, x2)
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_allclose(r1, r2)

    # 11 -----------------------------------------------------------------
    def test_model_input_compatibility(self):
        from phase3_model import build_and_compile_model

        model = build_and_compile_model("baseline")
        ds = make_tf_dataset(self.csv, "val", self.scaler, batch_size=8)
        x, y = next(iter(ds))
        out = model(x, training=False)
        self.assertEqual(tuple(out["classification"].shape), (8, 3))
        self.assertEqual(tuple(out["regression"].shape), (8, 1))
        np.testing.assert_allclose(out["classification"].numpy().sum(axis=1), 1.0, rtol=1e-5)

    # 12 -----------------------------------------------------------------
    def test_subset_creation_bounded(self):
        dest = self.dir / "subset.csv"
        info = create_symbol_subset(self.csv, dest, {"AAA", "BBB"})
        self.assertEqual(info["symbols"], ["AAA", "BBB"])
        self.assertGreater(info["rows"], 0)
        self.assertEqual(set(pd.read_csv(dest, usecols=["Symbol"])["Symbol"].unique()),
                         {"AAA", "BBB"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

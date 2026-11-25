"""
Phase 2 lightweight tests — synthetic data only. No training, no full dataset,
no HDFS/MongoDB, no large Spark jobs. Run:

    py -3.11 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lstm_data_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    build_labeled_sequences,
    chronological_partition,
    compute_technical_features,
    fit_scaler_on_train,
    prepare_symbol_frame,
    transform_sequences,
)
from fetch_live_data import compute_live_features  # noqa: E402
from predict_lstm import compute_features, build_prediction_sequence  # noqa: E402


def synthetic_frame(symbol="AAA", n=90, start="2023-01-02", base=100.0, seed=7):
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1.5, 0.0, 0.0, 1.5], size=n)  # explicit FLAT days
    close = base + np.cumsum(steps)
    return pd.DataFrame({
        "Symbol": symbol,
        "Date": dates,
        "Open": close - 0.2,
        "High": close + 0.8,
        "Low": close - 0.8,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, size=n).astype(float),
    })


class Phase2PipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a = prepare_symbol_frame(synthetic_frame("AAA", 120, base=100.0, seed=1))
        cls.b = prepare_symbol_frame(synthetic_frame("BBB", 120, base=500.0, seed=2))
        cls.both = pd.concat([cls.a, cls.b], ignore_index=True)
        cls.feat_a = compute_technical_features(cls.a)
        cls.feat_b = compute_technical_features(cls.b)

    # ------------------------------- A -------------------------------
    def test_a_multi_stock_isolation(self):
        X, y, meta = build_labeled_sequences([self.feat_a, self.feat_b],
                                              label_mode="three_class")
        self.assertGreater(len(meta), 0)
        self.assertTrue(set(meta["Symbol"]).issubset({"AAA", "BBB"}))
        close_idx = FEATURE_COLUMNS.index("Close")
        lookup = {"AAA": self.feat_a, "BBB": self.feat_b}
        for i, row in enumerate(meta.itertuples(index=False)):
            frame = lookup[row.Symbol]
            end_close = frame.loc[frame["Date"] == row.Input_End_Date, "Close"].iloc[0]
            self.assertAlmostEqual(float(X[i, -1, close_idx]), float(end_close), places=3)
            # last input date must belong to this symbol, never the other one
            self.assertTrue((frame["Date"] == row.Input_End_Date).any())

    # ------------------------------- B -------------------------------
    def test_b_chronological_sorting_per_symbol(self):
        shuffled = self.both.sample(frac=1.0, random_state=3)
        for symbol, frame in shuffled.groupby("Symbol"):
            prepared = prepare_symbol_frame(frame, symbol=symbol)
            self.assertTrue(prepared["Date"].is_monotonic_increasing)
            self.assertFalse(prepared["Date"].duplicated().any())
            self.assertTrue((prepared["Symbol"] == symbol).all())

    # ------------------------------- C -------------------------------
    def test_c_no_future_columns_in_features(self):
        forbidden = {"Target", "Target_Date", "Next_Return_Pct", "Future_Return_Pct"}
        self.assertTrue(forbidden.isdisjoint(FEATURE_COLUMNS))
        original = compute_technical_features(self.a)[FEATURE_COLUMNS].to_numpy(copy=True)
        changed = self.a.copy()
        changed.loc[changed.index[-1], "Close"] *= 1.10  # future observation only
        after = compute_technical_features(changed)[FEATURE_COLUMNS].to_numpy()
        np.testing.assert_allclose(original[:-1], after[:-1], rtol=1e-6, atol=1e-6)

    # ------------------------------- D -------------------------------
    def test_d_target_alignment_next_session(self):
        frame = synthetic_frame("TTT", n=100, base=100.0, seed=5)
        frame.loc[frame.index[-3], "Close"] = 100.0
        frame.loc[frame.index[-2], "Close"] = 102.0   # +2% -> UP
        frame.loc[frame.index[-1], "Close"] = 102.5   # +0.49% -> FLAT
        prepared = prepare_symbol_frame(frame)
        dated = compute_technical_features(prepared)
        from lstm_data_pipeline import add_next_session_target
        target = add_next_session_target(dated)

        d0, d1, d2 = prepared["Date"].iloc[-3:]
        self.assertEqual(int(target.loc[target["Date"] == d0, "Target"].iloc[0]), 2)
        self.assertEqual(target.loc[target["Date"] == d0, "Target_Date"].iloc[0], d1)
        self.assertEqual(int(target.loc[target["Date"] == d1, "Target"].iloc[0]), 1)
        self.assertEqual(int(target.loc[target["Date"] == d2, "Target"].iloc[0]), -1)

        X, y, meta = build_labeled_sequences([target], label_mode="three_class")
        match = meta.index[meta["Input_End_Date"] == d0].tolist()
        self.assertTrue(match)
        pos = match[0]
        self.assertEqual(meta.loc[pos, "Target_Date"], d1)    # predicted session t -> t+1
        self.assertEqual(int(meta.loc[pos, "Target"]), 2)
        self.assertEqual(int(y[pos]), 2)
        self.assertEqual(X.shape[1:], (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))

    # ------------------------------- E -------------------------------
    def test_e_flat_days_preserved(self):
        from lstm_data_pipeline import add_next_session_target
        target = add_next_session_target(self.feat_a)
        flat_days = int((target["Target"] == 1).sum())
        self.assertGreater(flat_days, 0)
        X, y, meta = build_labeled_sequences([target], label_mode="three_class")
        self.assertIn(1, set(y))                              # FLAT labels exist
        # pick a FLAT day inside the sequence-buildable range (after warmup)
        flat_date = target.loc[target["Target"] == 1, "Date"].iloc[-1]
        covered = ((meta["Input_Start_Date"] <= flat_date) &
                   (meta["Input_End_Date"] >= flat_date)).any()
        self.assertTrue(covered)                              # retained in input history

    # ------------------------------- F -------------------------------
    def test_f_split_integrity(self):
        from lstm_data_pipeline import add_next_session_target
        frames = [add_next_session_target(self.feat_a), add_next_session_target(self.feat_b)]
        X, y, meta = build_labeled_sequences(frames, label_mode="three_class")
        tr, va, te, report = chronological_partition(meta)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(set(va) & set(te)), 0)
        self.assertEqual(len(set(tr) & set(te)), 0)
        for symbol in meta["Symbol"].unique():
            sym_idx = meta.index[meta["Symbol"] == symbol].to_numpy()
            idx_tr = np.intersect1d(tr, sym_idx)
            idx_va = np.intersect1d(va, sym_idx)
            idx_te = np.intersect1d(te, sym_idx)
            self.assertGreater(pd.to_datetime(meta.loc[idx_va, "Target_Date"]).min(),
                               pd.to_datetime(meta.loc[idx_tr, "Target_Date"]).max())
            self.assertGreater(pd.to_datetime(meta.loc[idx_te, "Target_Date"]).min(),
                               pd.to_datetime(meta.loc[idx_va, "Target_Date"]).max())

    # ------------------------------- G -------------------------------
    def test_g_scaler_fit_on_train_only(self):
        from lstm_data_pipeline import add_next_session_target
        frames = [add_next_session_target(self.feat_a), add_next_session_target(self.feat_b)]
        X, y, meta = build_labeled_sequences(frames, label_mode="three_class")
        tr, va, te, _ = chronological_partition(meta)
        scaler = fit_scaler_on_train(X[tr])
        flat_train = X[tr].reshape(-1, len(FEATURE_COLUMNS))
        np.testing.assert_allclose(scaler.data_min_, flat_train.min(axis=0), rtol=1e-5)
        np.testing.assert_allclose(scaler.data_max_, flat_train.max(axis=0), rtol=1e-5)
        all_flat = X.reshape(-1, len(FEATURE_COLUMNS))
        self.assertLessEqual(scaler.data_min_[0], all_flat[:, 0].min() + 1e-6)
        scaled = transform_sequences(scaler, X[te])
        self.assertGreaterEqual(scaled.min(), 0.0)
        self.assertLessEqual(scaled.max(), 1.0)
        np.testing.assert_allclose(scaler.data_min_, flat_train.min(axis=0), rtol=1e-5)

    # ------------------------------- I -------------------------------
    def test_i_live_feature_consistency(self):
        frame = self.a.copy()
        expected = compute_technical_features(frame)[FEATURE_COLUMNS]
        live = compute_live_features(frame, symbol="AAA")[FEATURE_COLUMNS]
        cli = compute_features(frame, symbol="AAA")[FEATURE_COLUMNS]
        np.testing.assert_allclose(expected.to_numpy(), live.to_numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(expected.to_numpy(), cli.to_numpy(), rtol=1e-6, atol=1e-6)
        with self.assertRaises(ValueError):
            build_prediction_sequence(frame.iloc[:10], object(), symbol="AAA")


class SparkIntegrityTests(unittest.TestCase):
    """Tiny local Spark job; verifies window isolation and exact EMA."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        # The shell has PYSPARK_PYTHON pinned to 3.10; PySpark workers must
        # match this interpreter exactly (no cross-minor-version workers).
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        from pyspark.sql import SparkSession
        cls.spark = (SparkSession.builder.master("local[1]")
                     .appName("Phase2PipelineTest")
                     .config("spark.ui.enabled", "false")
                     .config("spark.sql.shuffle.partitions", "1")
                     .getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")
        from spark_live_processing import compute_spark_features
        data = []
        for symbol, base in (("AAA", 100.0), ("BBB", 500.0)):
            frame = synthetic_frame(symbol, 40, base=base, seed=11)
            for r in frame.itertuples(index=False):
                data.append((r.Symbol, r.Date.strftime("%Y-%m-%d"), r.Open,
                             r.High, r.Low, r.Close, r.Volume))
        cls.pdf = pd.DataFrame(data, columns=[
            "Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
        cls.result = compute_spark_features(
            cls.spark.createDataFrame(cls.pdf)
        ).orderBy("Symbol", "Date").collect()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    # ------------------------------- H -------------------------------
    def test_h_spark_windows_and_ema(self):
        rows = [r.asDict() for r in self.result]
        for symbol, base in (("AAA", 100.0), ("BBB", 500.0)):
            group = [r for r in rows if r["Symbol"] == symbol]
            self.assertEqual(len(group), 40)
            self.assertIsNone(group[0]["MA5"])                 # min_periods parity
            expected_ma5 = np.mean([r["Close"] for r in group[:5]])
            self.assertAlmostEqual(group[4]["MA5"], expected_ma5, places=4)

            closes = pd.Series([r["Close"] for r in group])
            pandas_ema = closes.ewm(span=12, adjust=False).mean()
            self.assertAlmostEqual(group[11]["EMA12"], float(pandas_ema.iloc[11]), places=4)

        # Rolling values of two symbols with different price levels stay separate.
        aaa = [r for r in rows if r["Symbol"] == "AAA"][20]["MA20"]
        bbb = [r for r in rows if r["Symbol"] == "BBB"][20]["MA20"]
        self.assertLess(aaa, 200.0)
        self.assertGreater(bbb, 400.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

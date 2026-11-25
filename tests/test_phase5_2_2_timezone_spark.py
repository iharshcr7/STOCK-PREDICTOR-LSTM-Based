"""
Phase 5.2.2 tests — mixed timezone normalization and Spark gateway validation.

External systems are mocked; the real trained LSTM is used for inference where
needed. Run:

    py -3.11 -m unittest discover -s tests -p "test_phase5_2_2_timezone_spark.py" -v
"""

import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import live_prediction_pipeline as lpp  # noqa: E402
import spark_env  # noqa: E402
import validate_spark  # noqa: E402
import stock_intelligence_pipeline as sip  # noqa: E402


def ohlcv_row(date_value, seed=1, price=150.0):
    rng = np.random.default_rng(seed)
    return {
        "Date": date_value,
        "Open": price - 0.2, "High": price + 0.7, "Low": price - 0.7,
        "Close": price, "Volume": float(rng.integers(500_000, 1_500_000)),
    }


def frame(rows):
    return pd.DataFrame(rows)


class Phase522TimezoneSparkTests(unittest.TestCase):

    # 1 ------------------------------------------------------------------
    def test_naive_timestamp_session_date(self):
        merged, report = self._merge([], ["2026-11-24"])
        self.assertEqual(report["latest_date"], "2026-11-24")
        self.assertEqual(merged["Date"].iloc[0].date().isoformat(), "2026-11-24")

    # 2 ------------------------------------------------------------------
    def test_utc_timestamp_session_date(self):
        # UTC midnight maps to the previous US/Eastern evening -> previous session.
        merged, report = self._merge([], ["2026-11-24T00:00:00Z"])
        self.assertEqual(report["latest_date"], "2026-11-23")

    # 3 ------------------------------------------------------------------
    def test_us_timezone_timestamp_session_date(self):
        merged, report = self._merge(
            [], ["2026-11-24T00:00:00-05:00", "2026-11-25T00:00:00-05:00"]
        )
        self.assertEqual(report["sessions"], ["2026-11-24", "2026-11-25"])

    # 4 ------------------------------------------------------------------
    def test_mixed_timezone_timestamps(self):
        with patch.object(
            lpp, "load_local_warmup",
            return_value=frame([ohlcv_row("2026-11-20", 1)]),
        ):
            provider = frame([
                ohlcv_row("2026-11-21T00:00:00Z", 2),      # -> 2026-11-20 session
                ohlcv_row("2026-11-23T00:00:00-05:00", 3), # -> 2026-11-23
                ohlcv_row("2026-11-24T00:00:00-05:00", 4), # -> 2026-11-24
            ])
            merged, report = lpp.merge_warmup_history(
                provider, "AAPL", asset_type="stocks", min_rows=2
            )
        self.assertFalse(merged.empty)
        self.assertEqual(report["sessions"], ["2026-11-20", "2026-11-23", "2026-11-24"])
        self.assertTrue(report["sufficient"])
        self.assertEqual(merged["Date"].dt.tz, None)

    # 5 ------------------------------------------------------------------
    def test_invalid_dates_removed(self):
        merged, report = self._merge([], ["not-a-date", "2026-11-24"])
        self.assertEqual(report["invalid_dates_dropped"], 1)
        self.assertEqual(report["sessions"], ["2026-11-24"])

    # 6 ------------------------------------------------------------------
    def test_duplicate_sessions_deduplicated(self):
        merged, report = self._merge(["2026-11-24"], ["2026-11-24"])
        self.assertEqual(report["duplicate_sessions_removed"], 1)
        self.assertEqual(report["merged_rows"], 1)

    # 7 ------------------------------------------------------------------
    def test_warmup_sufficiency(self):
        merged, report = self._merge(["2026-11-20", "2026-11-21"], ["2026-11-24"], min_rows=2)
        self.assertTrue(report["sufficient"])
        merged, report = self._merge(["2026-11-20"], [], min_rows=5)
        self.assertFalse(report["sufficient"])

    # 8 ------------------------------------------------------------------
    def test_ohlcv_preserved(self):
        merged, _ = self._merge([], ["2026-11-24"])
        for column in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(column, merged.columns)
            self.assertTrue(merged[column].notna().all())

    # 9 ------------------------------------------------------------------
    def test_java_and_spark_environment_validation(self):
        report = spark_env.diagnose_environment()
        self.assertIn(report["java"]["major"], (8, 11, 17))
        self.assertTrue(report["java"]["supported"])
        self.assertTrue(report["spark"]["found"])
        self.assertTrue(report["hadoop"]["winutils_found"])
        self.assertFalse(report["project"]["safe_path"])

    # 10 -----------------------------------------------------------------
    def test_spark_build_local_does_not_require_hadoop(self):
        from spark_processing import build_spark
        with patch("spark_env.diagnose_environment", return_value={
            "java": {"found": True, "supported": True, "major": 11, "version": "11"},
            "spark": {"found": True, "pyspark_version": "3.5.8"},
            "hadoop": {"hadoop_home": "", "winutils_found": False},
            "project": {"safe_path": False},
        }), patch("spark_env.ensure_spark_ready") as ready, \
             patch("spark_processing.SparkSession") as session:
            builder = MagicMock()
            session.builder = builder
            builder.appName.return_value = builder
            builder.master.return_value = builder
            builder.config.return_value = builder
            built_spark = MagicMock()
            built_spark.version = "3.5.8"
            builder.getOrCreate.return_value = built_spark
            spark = build_spark("LocalTest", use_hdfs=False)
        self.assertIs(spark, built_spark)
        ready.assert_called_once()

    # 11 -----------------------------------------------------------------
    def test_spark_build_hdfs_requires_hadoop(self):
        from spark_processing import build_spark
        with patch("spark_env.diagnose_environment", return_value={
            "java": {"found": True, "supported": True, "major": 11, "version": "11"},
            "spark": {"found": True, "pyspark_version": "3.5.8"},
            "hadoop": {"hadoop_home": "", "winutils_found": False},
            "project": {"safe_path": False},
        }), patch("spark_env.ensure_spark_ready"):
            with self.assertRaises(RuntimeError) as ctx:
                build_spark("HdfsTest", use_hdfs=True)
        self.assertIn("HADOOP_HOME", str(ctx.exception))

    # 12 -----------------------------------------------------------------
    def test_independent_local_spark_validation_mocked(self):
        spark = MagicMock()
        spark.version = "3.5.8"
        df = MagicMock()
        df.count.return_value = 5
        spark.createDataFrame.return_value = df
        spark.read.parquet.return_value = df
        with patch("spark_env.ensure_spark_ready"), \
             patch("pyspark.sql.SparkSession") as session:
            builder = MagicMock()
            session.builder = builder
            builder.appName.return_value = builder
            builder.master.return_value = builder
            builder.config.return_value = builder
            builder.getOrCreate.return_value = spark
            result = validate_spark.validate_local_spark()
        self.assertTrue(result["success"])
        self.assertEqual(result["rows_created"], 5)
        self.assertEqual(result["rows_read"], 5)

    # 13 -----------------------------------------------------------------
    def test_hdfs_read_write_mocked(self):
        spark = MagicMock()
        spark.version = "3.5.8"
        csv_df = MagicMock()
        csv_df.limit.return_value.count.return_value = 1
        spark.read.option.return_value.option.return_value.csv.return_value = csv_df
        small_df = MagicMock()
        small_df.write.mode.return_value.option.return_value = small_df
        spark.createDataFrame.return_value = small_df
        spark.read.parquet.return_value.count.return_value = 5
        with patch("spark_processing.build_spark", return_value=spark):
            result = validate_spark.validate_hdfs_spark("/stock_data/raw/stocks/AAPL/x/data.csv")
        self.assertTrue(result["success"])
        self.assertEqual(result["csv_rows"], 1)
        self.assertEqual(result["parquet_rows"], 5)

    # 14 -----------------------------------------------------------------
    def test_integration_pipeline_mocked(self):
        pipeline = sip.StockIntelligencePipeline(
            hdfs_runner=self._hdfs_runner(),
            spark_factory=lambda: "spark",
            spark_processor=lambda s, p, r, a: {
                "success": True, "input_records": 3, "output_records": 3,
                "parquet_path": "/stock_data/features/daily/mock", "feature_count": 21,
            },
            feature_reader=self._feature_reader,
            mongo_collection=self._collection(),
        )
        run = pipeline.run(["AAPL", "MSFT", "NVDA"])
        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.predictions), 3)
        self.assertEqual(run.input_records, 3)
        self.assertEqual(run.output_records, 3)
        self.assertEqual(run.spark["feature_count"], 21)

    # 15 -----------------------------------------------------------------
    def test_no_training_and_model_unchanged(self):
        import hashlib
        import lstm_inference as inference
        digest = hashlib.sha256(inference.MODEL_FILE.read_bytes()).hexdigest()
        self.assertEqual(digest, inference.validate_artifacts().model_sha256)
        source = (ROOT / "scripts" / "live_prediction_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("model.fit(", source)
        self.assertNotIn(".fit_transform(", source)

    # 16 -----------------------------------------------------------------
    def test_existing_phase4_and_phase5_tests(self):
        import unittest as _unittest
        from live_prediction_pipeline import MarketDataFetcher
        names = ["test_phase4_lstm_inference", "test_phase5_live_pipeline",
                 "test_phase5_1_dashboard", "test_phase5_2_architecture",
                 "test_phase5_2_1_pipeline_fixes"]
        for name in names:
            with self.subTest(suite=name):
                MarketDataFetcher.reset_runtime_state()
                suite = _unittest.defaultTestLoader.loadTestsFromName(name)
                self.assertGreater(suite.countTestCases(), 0)
                outcome = _unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
                self.assertTrue(outcome.wasSuccessful(), str(outcome.errors) + str(outcome.failures))

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _merge(local_dates, provider_dates, min_rows=1):
        local = frame([ohlcv_row(d, i) for i, d in enumerate(local_dates)])
        provider = frame([ohlcv_row(d, i + 10) for i, d in enumerate(provider_dates)])
        with patch.object(lpp, "load_local_warmup", return_value=local):
            return lpp.merge_warmup_history(
                provider, "AAPL", asset_type="stocks", min_rows=min_rows
            )

    @staticmethod
    def _hdfs_runner():
        files = {}
        def runner(args):
            cmd = args[0]
            if cmd == "-mkdir":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd == "-ls":
                path = args[-1]
                if path in files:
                    return SimpleNamespace(returncode=0, stdout=f"-rw-r--r-- x {path}\n", stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr=f"ls: `{path}': No such file or directory\n")
            if cmd == "-put":
                files[args[-1]] = Path(args[-2]).read_text(encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd == "-cat":
                return SimpleNamespace(returncode=0, stdout=files.get(args[-1], ""), stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return runner

    @staticmethod
    def _feature_reader(spark, run_id, asset_type, symbol, limit=300):
        dates = pd.bdate_range(end="2026-11-24", periods=150)
        close = 150 + np.cumsum(np.random.default_rng(7).normal(0.05, 1.0, 150))
        frame_ = pd.DataFrame({
            "Date": dates, "Open": close - 0.2, "High": close + 0.6,
            "Low": close - 0.6, "Close": close, "Volume": 1_000_000.0,
        })
        from lstm_data_pipeline import compute_technical_features
        return compute_technical_features(frame_)

    @staticmethod
    def _collection():
        class Collection:
            def __init__(self):
                self.docs = []
            def create_index(self, *a, **k):
                pass
            def find_one(self, q, projection=None):
                for d in self.docs:
                    if all(d.get(k) == v for k, v in q.items()):
                        return {"_id": 1}
                return None
            def insert_one(self, doc):
                stored = dict(doc)
                stored["_id"] = len(self.docs) + 1
                self.docs.append(stored)
                class R:
                    inserted_id = stored["_id"]
                return R()
            def update_one(self, *a, **k):
                return SimpleNamespace(matched_count=0)
        return Collection()


if __name__ == "__main__":
    unittest.main()

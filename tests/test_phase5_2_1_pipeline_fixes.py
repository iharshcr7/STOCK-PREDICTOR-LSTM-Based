"""
Phase 5.2.1 tests — Windows Spark, HDFS path, state and warm-up fixes.

External systems (Java gateway, Spark, HDFS, APIs) are mocked; inference uses
the real trained checkpoint. No training occurs.

    py -3.11 -m unittest discover -s tests -p "test_phase5_2_1_pipeline_fixes.py" -v
"""

import ast
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

import hdfs_market_store as hdfs  # noqa: E402
import lstm_inference as inference  # noqa: E402
import scaler_compat  # noqa: E402
import spark_env  # noqa: E402
import stock_intelligence_pipeline as sip  # noqa: E402
from lstm_data_pipeline import FEATURE_COLUMNS, compute_technical_features  # noqa: E402


def ohlcv(symbol="AAPL", days=150, end="2026-11-24", base=150.0, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days)
    close = base + np.cumsum(rng.normal(0.05, 1.0, days))
    return pd.DataFrame({
        "Symbol": symbol, "Date": dates,
        "Open": close - 0.2, "High": close + 0.6, "Low": close - 0.6,
        "Close": close, "Adj Close": close * 0.99,
        "Volume": rng.integers(500_000, 2_000_000, days).astype(float),
    })


def processed(symbol="AAPL", days=150):
    return compute_technical_features(ohlcv(symbol, days))[
        ["Symbol", "Date", "Adj Close", *FEATURE_COLUMNS]
    ]


class FakeHdfs:
    def __init__(self):
        self.files = {}
        self.calls = []

    def runner(self, args):
        self.calls.append(list(args))
        cmd = args[0]
        if cmd == "-mkdir":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == "-ls":
            path = args[-1]
            if path in self.files:
                return SimpleNamespace(returncode=0, stdout=f"-rw-r--r-- x {path}\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr=f"ls: `{path}': No such file or directory\n")
        if cmd == "-put":
            self.files[args[-1]] = Path(args[-2]).read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == "-cat":
            return SimpleNamespace(returncode=0, stdout=self.files.get(args[-1], ""), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class DummyExpr:
    def __getattr__(self, name):
        return self

    def __call__(self, *a, **k):
        return self

    def __gt__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __and__(self, other):
        return self

    def __or__(self, other):
        return self


def spark_expr_patches():
    names = ("input_file_name", "regexp_extract", "upper", "col", "length",
             "when", "lit", "year", "to_date")
    return [patch(f"pyspark.sql.functions.{n}", return_value=DummyExpr()) for n in names]


class FakeDF:
    def __init__(self, columns, log, rows=100):
        self.columns = list(columns)
        self.log = log
        self.rows = rows
        self.write = FakeWriter(log)

    def withColumn(self, name, value):
        if name not in self.columns:
            self.columns.append(name)
        self.log.setdefault("with_columns", []).append(name)
        return self

    def filter(self, expr):
        return self

    def dropDuplicates(self, cols):
        self.log["deduplicated"] = cols
        return self

    def dropna(self, subset):
        return self

    def select(self, *cols):
        self.columns = list(cols)
        self.log["selected"] = list(cols)
        return self

    def count(self):
        return self.rows

    def orderBy(self, *a):
        return self

    def limit(self, n):
        return self

    def toPandas(self):
        return processed("AAPL")

    def createDataFrame(self, rows):
        return self


class FakeWriter:
    def __init__(self, log):
        self.log = log

    def mode(self, value):
        self.log["write_mode"] = value
        return self

    def partitionBy(self, *cols):
        self.log["partitions"] = list(cols)
        return self

    def option(self, key, value):
        return self

    def parquet(self, path):
        self.log["parquet_path"] = path
        return self


class FakeSpark:
    def __init__(self, df):
        self.df = df
        self.read = self
        self.version = "3.5.8"
        self.sparkContext = SimpleNamespace(uiWebUrl="http://localhost:4040")
        self.log = df.log

    def option(self, *a):
        return self

    def csv(self, paths):
        self.log["csv_paths"] = list(paths)
        return self.df

    def parquet(self, path):
        self.log["read_parquet"] = path
        return self.df

    def createDataFrame(self, rows):
        return self.df


class FakeCollection:
    def __init__(self):
        self.docs = []
        self.indexes = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return {"_id": 1}
        return None

    def insert_one(self, doc):
        stored = dict(doc)
        stored["_id"] = len(self.docs) + 1
        self.docs.append(stored)

        class Result:
            inserted_id = stored["_id"]
        return Result()

    def update_one(self, query, update):
        return SimpleNamespace(matched_count=0)

    def find(self, query=None, projection=None):
        class Cursor:
            def __init__(self, rows):
                self.rows = rows
            def sort(self, *a, **k):
                return self
            def limit(self, n):
                self.rows = self.rows[:n]
                return self
            def __iter__(self):
                return iter(self.rows)
        return Cursor([])


def make_pipeline(hdfs_fake=None, reader=None, collection=None):
    log = {}
    df = FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log)
    return sip.StockIntelligencePipeline(
        hdfs_runner=(hdfs_fake or FakeHdfs()).runner,
        spark_factory=lambda: FakeSpark(df),
        spark_processor=lambda s, paths, run_id, asset_type: {
            "success": True, "input_records": 100, "output_records": 90,
            "parquet_path": "/stock_data/features/daily/x", "parquet_uri": "hdfs://localhost:9000/x",
        },
        feature_reader=reader or (lambda s, r, a, sym, limit=300: processed(sym)),
        mongo_collection=collection or FakeCollection(),
    )


class Phase521PipelineFixTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    # 1 ------------------------------------------------------------------
    def test_windows_spark_environment_detection(self):
        report = spark_env.diagnose_environment()
        self.assertIn("java", report)
        self.assertIn("spark", report)
        self.assertIn("hadoop", report)
        self.assertTrue(report["spark"]["found"])
        self.assertIn(report["java"]["major"], (8, 11, 17))
        self.assertTrue(report["hadoop"]["winutils_found"])
        self.assertFalse(report["project"]["safe_path"])
        self.assertIn("gateway", report)

    # 2 ------------------------------------------------------------------
    def test_java_gateway_startup(self):
        gateway = MagicMock()
        with patch("pyspark.java_gateway.launch_gateway", return_value=gateway) as launch:
            result = spark_env.test_java_gateway()
        self.assertTrue(result["success"])
        launch.assert_called_once()
        gateway.shutdown.assert_called_once()

    # 3 ------------------------------------------------------------------
    def test_spark_session_creation(self):
        with patch("spark_processing.build_spark", return_value="SPARK") as build:
            sip._SPARK_CACHE.clear()
            with patch.object(spark_env, "ensure_spark_ready") as ready:
                spark = sip.get_spark("UnitTestApp")
        self.assertEqual(spark, "SPARK")
        build.assert_called_once()
        ready.assert_called_once()

    # 4 ------------------------------------------------------------------
    def test_hdfs_csv_reading(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Close"], log, rows=1))
        result = spark_env.test_hdfs_csv_read(spark, "/stock_data/raw/stocks/AAPL/x/data.csv")
        self.assertTrue(result["success"])
        self.assertEqual(result["rows"], 1)

    # 5 ------------------------------------------------------------------
    def test_hdfs_parquet_writing_and_reading(self):
        log = {}
        spark = FakeSpark(FakeDF(["symbol", "value"], log, rows=1))
        result = spark_env.test_hdfs_parquet_roundtrip(spark, "/tmp/unit_parquet")
        self.assertTrue(result["success"])
        self.assertEqual(result["rows"], 1)
        self.assertEqual(log["write_mode"], "overwrite")

    # 6 ------------------------------------------------------------------
    def test_stock_and_etf_classification(self):
        self.assertEqual(sip.symbol_asset_type("AAPL"), "stocks")
        self.assertEqual(sip.symbol_asset_type("MSFT"), "stocks")
        self.assertEqual(sip.symbol_asset_type("NVDA"), "stocks")
        self.assertEqual(sip.symbol_asset_type("AMZN"), "stocks")
        self.assertEqual(sip.symbol_asset_type("GOOGL"), "stocks")
        self.assertEqual(sip.symbol_asset_type("AAAU"), "etfs")
        etfs = {s.symbol for s in sip.discover_universe("etfs")}
        self.assertNotIn("AAPL", etfs)
        resolved = sip.resolve_symbol_info("AAPL", asset_type_hint="etfs")
        self.assertEqual(resolved.asset_type, "stocks")

    # 7 ------------------------------------------------------------------
    def test_hdfs_ingestion_path_consistency(self):
        fake = FakeHdfs()
        status = hdfs.ingest_dataframe(
            ohlcv("AAPL").drop(columns=["Symbol"]), "AAPL",
            asset_type="stocks", runner=fake.runner, live=True, force=True,
        )
        self.assertEqual(status["status"], "ingested")
        self.assertIn("/raw/live_updates/stocks/AAPL/", status["hdfs_path"])
        self.assertNotIn("/etfs/", status["hdfs_path"])
        with self.assertRaises(ValueError):
            hdfs.raw_symbol_path("AAPL", asset_type="unknown")

    # 8 ------------------------------------------------------------------
    def test_symbol_column_propagation(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log))
        with patch("pyspark.sql.functions.input_file_name", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.regexp_extract", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.upper", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.col", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.length", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.when", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.lit", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.year", return_value=DummyExpr()), \
             patch("pyspark.sql.functions.to_date", return_value=DummyExpr()), \
             patch("spark_live_processing.compute_spark_features", side_effect=lambda df: df), \
             patch("spark_live_processing.validate_spark_features", return_value=True):
            sip.process_raw_with_spark(
                spark,
                ["/stock_data/raw/stocks/AAPL/2026-11-24/data.csv"],
                "run-symbols", "stocks",
            )
        self.assertIn("Symbol", log["with_columns"])
        self.assertIn("asset_type", log["with_columns"])
        self.assertTrue(all(c in log["selected"] for c in FEATURE_COLUMNS))

    # 9 ------------------------------------------------------------------
    def test_spark_input_discovery_from_manifest(self):
        fake = FakeHdfs()
        path = "/stock_data/raw/stocks/AAPL/2026-11-24/data.csv"
        fake.files[path] = "Date,Open,High,Low,Close,Volume\n2026-11-24,1,2,0.5,1.5,100\n"
        statuses = [{"symbol": "AAPL", "status": "ingested", "hdfs_path": path, "asset_type": "stocks"}]
        hdfs.write_ingestion_manifest("run-discover", statuses, runner=fake.runner)
        captured = {}
        def processor(spark, paths, run_id, asset_type):
            captured["paths"] = paths
            return {"success": True, "input_records": 1, "output_records": 1,
                    "parquet_path": "/x", "feature_count": 21}
        pipeline = make_pipeline(fake)
        pipeline.spark_processor = processor
        run = sip.PipelineRun("run-discover", "local_historical", "stocks")
        pipeline.run_spark(run, [{"symbol": "WRONG", "status": "ingested", "hdfs_path": "/wrong"}])
        self.assertEqual(captured["paths"], [path])

    # 10 -----------------------------------------------------------------
    def test_session_state_key_separation(self):
        source = self.app_source
        self.assertIn('key="p52_run_button"', source)
        self.assertIn("st.session_state['p52_run_result']", source)
        self.assertNotIn('key="p52_run"', source)
        self.assertNotIn("st.session_state['p52_run']", source)
        self.assertIn("normalize_run_result", source)
        tree = ast.parse(source)
        self.assertIsInstance(tree, ast.Module)

    # 11 -----------------------------------------------------------------
    def test_failed_run_result_normalization(self):
        for bad in (True, False, None, "error", 42):
            result = sip.normalize_run_result(bad)
            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "unknown")
            self.assertIsInstance(result["errors"], list)
            self.assertIsInstance(result["failed_symbols"], dict)
        partial = sip.normalize_run_result({
            "status": "failed", "errors": ["stage failed"],
            "failed_symbols": {"AAPL": "api_failure"},
        })
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(partial["errors"], ["stage failed"])
        self.assertEqual(partial["predictions"], [])

    # 12 -----------------------------------------------------------------
    def test_partial_pipeline_failures(self):
        def reader(spark, run_id, asset_type, symbol, limit=300):
            return pd.DataFrame() if symbol == "MSFT" else processed(symbol)
        pipeline = make_pipeline(reader=reader)
        run = pipeline.run(["AAPL", "MSFT"], mode="local_historical")
        self.assertEqual(run.status, "partial")
        self.assertIn("MSFT", run.incompatible_symbols)
        self.assertEqual(len(run.predictions), 1)
        self.assertIsInstance(run.to_dict()["ingestion"], dict)

    # 13 -----------------------------------------------------------------
    def test_feature_warmup_sufficiency(self):
        local = ohlcv("AAPL", days=400)  # load_local_warmup adds Symbol
        provider = ohlcv("AAPL", days=100, end="2026-11-25", seed=9).drop(columns=["Symbol"])
        with patch("live_prediction_pipeline.load_local_warmup", return_value=local):
            from live_prediction_pipeline import merge_warmup_history
            merged, report = merge_warmup_history(provider, "AAPL", asset_type="stocks")
        self.assertTrue(report["sufficient"])
        self.assertGreaterEqual(report["merged_rows"], 400)
        self.assertEqual(merged["Symbol"].iloc[0], "AAPL")
        self.assertEqual(merged["Date"].is_monotonic_increasing, True)
        self.assertEqual(merged["Date"].duplicated().sum(), 0)

    # 14 -----------------------------------------------------------------
    def test_original_scaler_transform_only(self):
        with patch("sklearn.preprocessing.MinMaxScaler.fit", side_effect=AssertionError("fit")), \
             patch("sklearn.preprocessing.MinMaxScaler.fit_transform", side_effect=AssertionError("fit_transform")):
            report = scaler_compat.check_scaler()
            self.assertTrue(report["transform_consistent"])
            result = inference.predict_from_features("AAPL", processed("AAPL"))
        self.assertTrue(result["ok"], result.get("errors"))

    # 15 -----------------------------------------------------------------
    def test_existing_lstm_inference(self):
        result = inference.predict_from_features("AAPL", processed("AAPL"))
        self.assertTrue(result["ok"])
        self.assertIn(result["prediction"], ("UP", "DOWN"))
        self.assertEqual(result["model_sha256"], inference.validate_artifacts().model_sha256)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 100.0, places=2)

    # 16 -----------------------------------------------------------------
    def test_existing_phase4_and_phase5_tests(self):
        import unittest as _unittest
        from live_prediction_pipeline import MarketDataFetcher

        names = ["test_phase4_lstm_inference", "test_phase5_live_pipeline",
                 "test_phase5_1_dashboard", "test_phase5_2_architecture"]
        for name in names:
            with self.subTest(suite=name):
                # Provider cache is process-wide by design; clear it so an
                # earlier suite cannot make a later suite observe cache hits.
                MarketDataFetcher.reset_runtime_state()
                suite = _unittest.defaultTestLoader.loadTestsFromName(name)
                self.assertGreater(suite.countTestCases(), 0)
                outcome = _unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
                self.assertTrue(outcome.wasSuccessful(), str(outcome.errors) + str(outcome.failures))


if __name__ == "__main__":
    unittest.main()

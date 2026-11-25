"""
Phase 5.2 tests — HDFS -> Spark -> Parquet -> existing LSTM -> MongoDB.

Network calls, HDFS CLI and Spark jobs are mocked. Inference uses the real
trained checkpoint only (no training). Run:

    py -3.11 -m unittest discover -s tests -p "test_phase5_2_architecture.py" -v
"""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hdfs_market_store as hdfs  # noqa: E402
import lstm_inference as inference  # noqa: E402
import stock_intelligence_pipeline as sip  # noqa: E402
from lstm_data_pipeline import (  # noqa: E402
    FEATURE_COLUMNS, SEQUENCE_LENGTH, compute_technical_features,
)
from model_config import FINAL_FEATURE_COLUMNS, read_model_metadata  # noqa: E402


def synthetic_ohlcv(symbol="AAPL", days=140, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-11-24"), periods=days)
    close = 150 + np.cumsum(rng.normal(0.05, 1.1, days))
    return pd.DataFrame({
        "Symbol": symbol, "Date": dates,
        "Open": close - 0.2, "High": close + 0.7, "Low": close - 0.7,
        "Close": close, "Adj Close": close * 0.99,
        "Volume": rng.integers(500_000, 2_000_000, days).astype(float),
    })


def processed_frame(symbol="AAPL"):
    return compute_technical_features(synthetic_ohlcv(symbol))[
        ["Symbol", "Date", "Adj Close", *FEATURE_COLUMNS]
    ]


class FakeHdfs:
    """In-memory HDFS CLI stand-in."""

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
                return SimpleNamespace(returncode=0, stdout=f"-rw-r--r-- 0 0 1 x {path}\n", stderr="")
            return SimpleNamespace(
                returncode=0, stdout="",
                stderr=f"ls: `{path}': No such file or directory\n",
            )
        if cmd == "-put":
            staged, target = args[-2], args[-1]
            self.files[target] = Path(staged).read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd == "-cat":
            return SimpleNamespace(
                returncode=0, stdout=self.files.get(args[-1], ""), stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class FakeWriter:
    def __init__(self, log):
        self.log = log

    def mode(self, value):
        self.log["mode"] = value
        return self

    def partitionBy(self, *cols):
        self.log["partition_by"] = list(cols)
        return self

    def option(self, key, value):
        self.log["option"] = (key, value)
        return self

    def parquet(self, path):
        self.log["parquet_path"] = path
        return self


class FakeDF:
    """Minimal chainable Spark DataFrame used to verify the processing stages."""

    def __init__(self, columns, log, count=100):
        self.columns = list(columns)
        self.log = log
        self._count = count
        self.write = FakeWriter(log)

    def withColumn(self, name, value):
        if name not in self.columns:
            self.columns.append(name)
        self.log.setdefault("with_columns", []).append(name)
        return self

    def filter(self, expr):
        self.log["filtered"] = True
        return self

    def dropDuplicates(self, cols):
        self.log["deduplicated"] = cols
        return self

    def dropna(self, subset):
        self.log["dropna"] = list(subset)
        return self

    def select(self, *cols):
        self.columns = list(cols)
        self.log["selected"] = list(cols)
        return self

    def count(self):
        return self._count

    def orderBy(self, col):
        return self

    def limit(self, n):
        self.log["limit"] = n
        return self

    def toPandas(self):
        return processed_frame("AAPL")


class FakeSpark:
    def __init__(self, df):
        self.df = df
        self.read = self
        self.sparkContext = SimpleNamespace(uiWebUrl="http://localhost:4040")
        self.version = "3.5.8"
        self.log = df.log

    def option(self, key, value):
        return self

    def csv(self, paths):
        self.log["csv_paths"] = list(paths)
        return self.df

    def parquet(self, path):
        self.log["read_parquet_path"] = path
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
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                doc.setdefault("meta", {}).update(
                    {k.split(".", 1)[-1]: v for k, v in update["$set"].items()}
                )
                return SimpleNamespace(matched_count=1)
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
        return Cursor([dict(d) for d in self.docs])


class DummyExpr:
    """Chainable stand-in for a PySpark Column expression."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __gt__(self, other):
        return self

    def __lt__(self, other):
        return self

    def __and__(self, other):
        return self

    def __or__(self, other):
        return self

    def __eq__(self, other):
        return self

    def __ne__(self, other):
        return self


def spark_function_patches():
    """Allow Spark expression construction without an active SparkContext."""
    dummy = DummyExpr()
    names = (
        "input_file_name", "regexp_extract", "upper", "col", "length",
        "when", "lit", "year", "to_date",
    )
    return [
        patch(f"pyspark.sql.functions.{name}", return_value=dummy)
        for name in names
    ]


def make_pipeline(hdfs_fake, spark=None, processor=None, reader=None, collection=None):
    log = {}
    df = FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log)
    return sip.StockIntelligencePipeline(
        hdfs_runner=hdfs_fake.runner,
        spark_factory=lambda: spark or FakeSpark(df),
        spark_processor=processor or (lambda s, p, r, a: {
            "success": True, "input_records": 100, "output_records": 90,
            "parquet_path": "/stock_data/features/daily/x", "feature_count": 21,
        }),
        feature_reader=reader or (lambda s, r, a, sym, limit=300: processed_frame(sym)),
        mongo_collection=collection or FakeCollection(),
    )


class Phase52ArchitectureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = inference.validate_artifacts()
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    # 1 ------------------------------------------------------------------
    def test_full_stock_universe_discovery(self):
        universe = sip.discover_universe("all")
        coverage = sip.classify_coverage(universe)
        self.assertEqual(coverage["stocks"], 5884)
        self.assertEqual(coverage["etfs"], 2165)
        self.assertEqual(coverage["discovered"], 8049)
        self.assertGreater(coverage["with_local_data"], 7000)

    # 2 ------------------------------------------------------------------
    def test_raw_hdfs_ingestion(self):
        fake = FakeHdfs()
        status = hdfs.ingest_local_csv(
            "AAPL", ROOT / "data" / "stock_market_dataset" / "stocks" / "AAPL.csv",
            runner=fake.runner,
        )
        self.assertEqual(status["status"], "ingested")
        self.assertIn("AAPL", status["hdfs_path"])
        self.assertTrue(any(c[0] == "-put" for c in fake.calls))

    # 3 ------------------------------------------------------------------
    def test_hdfs_data_persistence_and_manifest(self):
        fake = FakeHdfs()
        hdfs.ensure_directories(runner=fake.runner)
        ok = hdfs.write_manifest("run-test", {"status": "completed", "n": 7}, runner=fake.runner)
        self.assertTrue(ok)
        manifest = hdfs.read_manifest("run-test", runner=fake.runner)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["n"], 7)

    # 4 ------------------------------------------------------------------
    def test_spark_raw_data_reading(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log))
        with ExitStack() as stack:
            for p in spark_function_patches():
                stack.enter_context(p)
            stack.enter_context(patch("spark_live_processing.compute_spark_features", side_effect=lambda df: df))
            stack.enter_context(patch("spark_live_processing.validate_spark_features", return_value=True))
            result = sip.process_raw_with_spark(
                spark, ["/stock_data/raw/stocks/AAPL/2026-11-24/data.csv"], "run-x", "stocks"
            )
        self.assertTrue(result["success"])
        self.assertTrue(log["csv_paths"][0].startswith("hdfs://localhost:9000"))
        self.assertTrue(log["filtered"])

    # 5 ------------------------------------------------------------------
    def test_spark_preprocessing_invocation(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log))
        with ExitStack() as stack:
            for p in spark_function_patches():
                stack.enter_context(p)
            compute = stack.enter_context(
                patch("spark_live_processing.compute_spark_features", side_effect=lambda df: df)
            )
            validate = stack.enter_context(
                patch("spark_live_processing.validate_spark_features", return_value=True)
            )
            sip.process_raw_with_spark(
                spark, ["/stock_data/raw/stocks/AAPL/2026-11-24/data.csv"], "run-y", "stocks"
            )
        compute.assert_called_once()
        validate.assert_called_once()

    # 6 ------------------------------------------------------------------
    def test_feature_ordering(self):
        self.assertEqual(read_model_metadata()["feature_columns"], FINAL_FEATURE_COLUMNS)
        self.assertEqual(FINAL_FEATURE_COLUMNS, list(FEATURE_COLUMNS))
        self.assertEqual(len(FEATURE_COLUMNS), 21)

    # 7 ------------------------------------------------------------------
    def test_per_symbol_rolling_calculations(self):
        source = (ROOT / "scripts" / "spark_live_processing.py").read_text(encoding="utf-8")
        self.assertIn('Window.partitionBy("Symbol").orderBy("Date")', source)
        self.assertIn("_ema_over_window", source)

    # 8 ------------------------------------------------------------------
    def test_parquet_generation(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log))
        with ExitStack() as stack:
            for p in spark_function_patches():
                stack.enter_context(p)
            stack.enter_context(patch("spark_live_processing.compute_spark_features", side_effect=lambda df: df))
            stack.enter_context(patch("spark_live_processing.validate_spark_features", return_value=True))
            result = sip.process_raw_with_spark(
                spark, ["/stock_data/raw/stocks/AAPL/2026-11-24/data.csv"], "run-z", "stocks"
            )
        self.assertEqual(log["mode"], "overwrite")
        self.assertEqual(log["partition_by"], ["asset_type", "year"])
        self.assertEqual(log["option"], ("compression", "snappy"))
        self.assertEqual(log["parquet_path"], result["parquet_uri"])

    # 9 ------------------------------------------------------------------
    def test_hdfs_parquet_reading(self):
        log = {}
        spark = FakeSpark(FakeDF(["Date", "Open", "High", "Low", "Close", "Volume"], log))
        with ExitStack() as stack:
            for p in spark_function_patches():
                stack.enter_context(p)
            frame = sip.read_processed_features(spark, "run-z", "stocks", "AAPL")
        self.assertIn("read_parquet_path", log)
        self.assertEqual(log["limit"], sip.DEFAULT_MAX_ROWS)
        self.assertTrue({"Date", "Close", *FEATURE_COLUMNS}.issubset(frame.columns))

    # 10 -----------------------------------------------------------------
    def test_lstm_inference_from_processed_features(self):
        result = inference.predict_from_features(
            "AAPL", processed_frame("AAPL"), source="hdfs_parquet", run_id="run-test"
        )
        self.assertTrue(result["ok"], result.get("errors"))
        self.assertIn(result["prediction"], ("UP", "DOWN"))
        self.assertEqual(result["sequence_length"], SEQUENCE_LENGTH)
        self.assertEqual(result["n_features"], 21)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 100.0, places=2)

    # 11 -----------------------------------------------------------------
    def test_scaler_transform_only(self):
        from sklearn.preprocessing import MinMaxScaler
        with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("fit called")), \
             patch.object(MinMaxScaler, "fit_transform", side_effect=AssertionError("fit_transform called")):
            result = inference.predict_from_features("AAPL", processed_frame("AAPL"))
        self.assertTrue(result["ok"], result.get("errors"))

    # 12 -----------------------------------------------------------------
    def test_mongodb_prediction_storage(self):
        collection = FakeCollection()
        pipeline = make_pipeline(FakeHdfs(), collection=collection)
        run = sip.PipelineRun(run_id="run-mongo", mode="local_historical", asset_type="stocks")
        prediction = inference.predict_from_features("AAPL", processed_frame("AAPL"))
        prediction["run_id"] = run.run_id
        prediction["hdfs_parquet_reference"] = "/stock_data/features/daily/run-mongo"
        mongo_id = pipeline.persist_prediction(run, prediction)
        self.assertTrue(mongo_id)
        record = collection.docs[0]
        self.assertEqual(record["model"], "LSTM")
        self.assertEqual(record["meta"]["processing_run_id"], "run-mongo")
        self.assertIn("hdfs_parquet_reference", record["meta"])

    # 13 -----------------------------------------------------------------
    def test_duplicate_prevention(self):
        from mongo_store import save_lstm_prediction
        collection = FakeCollection()
        record = {
            "symbol": "AAPL", "prediction": "UP", "confidence": 55.0,
            "timestamp": "2026-11-24T00:00:00+00:00", "source": "hdfs_parquet",
            "model": "LSTM", "dedup_key": "same-key", "meta": {},
        }
        self.assertEqual(save_lstm_prediction(record, collection), save_lstm_prediction(record, collection))
        self.assertEqual(len(collection.docs), 1)

    # 14 -----------------------------------------------------------------
    def test_multi_stock_partial_failures(self):
        fake = FakeHdfs()
        def reader(spark, run_id, asset_type, symbol, limit=300):
            if symbol == "MSFT":
                return pd.DataFrame()
            return processed_frame(symbol)
        collection = FakeCollection()
        pipeline = make_pipeline(fake, reader=reader, collection=collection)
        run = pipeline.run(["AAPL", "MSFT"])
        self.assertEqual(run.status, "partial")
        self.assertIn("MSFT", run.incompatible_symbols)
        self.assertEqual(len(run.predictions), 1)

    # 15 -----------------------------------------------------------------
    def test_ohlcv_visualization_data(self):
        payload = sip.build_market_payload(processed_frame("AAPL"), "AAPL")
        for key in ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume",
                    "MA20", "RSI", "MACD", "BB_Upper", "Daily_Returns", "SMA50",
                    "SMA200", "latest_metrics"):
            self.assertIn(key, payload)
        self.assertIn("daily_change_pct", payload["latest_metrics"])
        self.assertEqual(payload["symbol"], "AAPL")

    # 16 -----------------------------------------------------------------
    def test_historical_and_live_data_distinction(self):
        fake = FakeHdfs()
        pipeline = make_pipeline(fake)
        status = pipeline.ingest_live.__wrapped__ if hasattr(pipeline.ingest_live, "__wrapped__") else None
        # Local ingestion keeps historical provenance; live ingestion is labelled.
        info = sip.SymbolInfo(symbol="AAPL", asset_type="stocks",
                              local_path=str(ROOT / "data" / "stock_market_dataset" / "stocks" / "AAPL.csv"))
        local_status = pipeline.ingest_local([info], run_id="r")[0]
        self.assertEqual(local_status["status"], "ingested")
        live = hdfs.ingest_dataframe(
            synthetic_ohlcv("AAPL").drop(columns=["Symbol"]),
            "AAPL", runner=fake.runner, live=True, force=True
        )
        self.assertEqual(live["data_kind"], "live_incremental")
        prediction = inference.predict_from_features("AAPL", processed_frame("AAPL"))
        self.assertEqual(prediction["source"], "hdfs_parquet")

    # 17 -----------------------------------------------------------------
    def test_pipeline_recovery(self):
        fake = FakeHdfs()
        manifest = {
            "mode": "local_historical", "asset_type": "stocks",
            "failed_symbols": {"AAPL": "ingestion_failed"},
            "incompatible_symbols": {"MSFT": "insufficient history"},
        }
        hdfs.write_manifest("run-recover", manifest, runner=fake.runner)
        pipeline = make_pipeline(fake)
        with patch.object(pipeline, "run") as rerun:
            rerun.return_value = sip.PipelineRun("r2", "local_historical", "stocks")
            result = pipeline.resume("run-recover")
        self.assertIsNotNone(result)
        args = rerun.call_args[0][0]
        self.assertCountEqual(args, ["AAPL", "MSFT"])

    # 18 -----------------------------------------------------------------
    def test_incremental_updates(self):
        fake = FakeHdfs()
        info = sip.SymbolInfo(symbol="AAPL", asset_type="stocks",
                              local_path=str(ROOT / "data" / "stock_market_dataset" / "stocks" / "AAPL.csv"))
        pipeline = make_pipeline(fake)
        first = pipeline.ingest_local([info], run_id="r")[0]
        second = pipeline.ingest_local([info], run_id="r")[0]
        forced = pipeline.ingest_local([info], run_id="r", force=True)[0]
        self.assertEqual(first["status"], "ingested")
        self.assertEqual(second["status"], "already_ingested")
        self.assertEqual(forced["status"], "ingested")

    # 19 -----------------------------------------------------------------
    def test_streamlit_integration(self):
        source = self.app_source
        self.assertIn('"Distributed Pipeline"', source)
        for token in (
            "StockIntelligencePipeline", "build_market_payload",
            "render_market_analytics", "Run HDFS+Spark pipeline",
            "candlestick", "SMA200", "MACD", "Bollinger", "daily returns",
            "HDFS parquet", "Full-universe coverage",
        ):
            self.assertIn(token, source)
        tree = ast.parse(source)
        self.assertIsInstance(tree, ast.Module)
        self.assertNotIn("use_container_width", source)

    # 20 -----------------------------------------------------------------
    def test_no_training_methods_invoked(self):
        module_source = (ROOT / "scripts" / "stock_intelligence_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(module_source)
        forbidden = {"fit", "fit_transform", "train", "compile", "build_lstm_model"}
        called = {
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertFalse(forbidden & called, f"training call found: {forbidden & called}")

        from sklearn.preprocessing import MinMaxScaler
        from tensorflow import keras
        with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("fit called")), \
             patch.object(MinMaxScaler, "fit_transform", side_effect=AssertionError("fit called")), \
             patch.object(keras.Model, "fit", side_effect=AssertionError("model.fit called")):
            result = inference.predict_from_features("AAPL", processed_frame("AAPL"))
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()

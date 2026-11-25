"""
Phase 5 tests — live data pipeline and orchestration.

All network calls are mocked; no API quota is consumed and no training occurs.
The real trained checkpoint is loaded read-only for the inference tests.

    py -3.11 -m unittest discover -s tests -p "test_phase5_live_pipeline.py" -v
"""

import ast
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import live_prediction_pipeline as p5  # noqa: E402
import lstm_inference as inference  # noqa: E402
import predict_lstm as legacy  # noqa: E402
from lstm_data_pipeline import FEATURE_COLUMNS, SEQUENCE_LENGTH  # noqa: E402
from model_config import FINAL_FEATURE_COLUMNS, read_model_metadata  # noqa: E402
from mongo_store import save_lstm_prediction  # noqa: E402

FIXED_NOW = datetime(2026, 11, 24, 17, 0, tzinfo=ZoneInfo("America/New_York"))  # after close
FIXED_DATE = pd.Timestamp("2026-11-24")


def synthetic_history(symbol="TEST", days=130, end=FIXED_DATE, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days + 3)[-days:]
    steps = rng.normal(0.05, 1.2, size=days)
    close = 150.0 + np.cumsum(steps)
    return pd.DataFrame({
        "Symbol": symbol,
        "Date": dates,
        "Open": close - 0.2,
        "High": close + np.abs(steps) + 0.5,
        "Low": close - np.abs(steps) - 0.5,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, size=days).astype(float),
    })


class FakeFetcher:
    """Deterministic in-memory replacement for MarketDataFetcher."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def fetch(self, symbol, force_refresh=False):
        self.calls.append((symbol, force_refresh))
        value = self.responses.get(symbol)
        if isinstance(value, Exception):
            raise value
        df = value if value is not None else synthetic_history(symbol)
        return p5.MarketData(symbol=symbol, df=df, provider="yfinance")


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, key, direction=1):
        field = key[0] if isinstance(key, (list, tuple)) else key
        self.rows.sort(key=lambda r: r.get(field, ""), reverse=int(direction) < 0)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeCollection:
    def __init__(self, fail_insert=False):
        self.docs = []
        self.indexes = []
        self.fail_insert = fail_insert

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    @staticmethod
    def _project(doc, projection):
        if projection is None:
            return dict(doc)
        included = [k for k, v in projection.items() if v == 1]
        excluded = {k for k, v in projection.items() if v == 0}
        if included:
            return {k: v for k, v in doc.items() if k in included}
        return {k: v for k, v in doc.items() if k not in excluded}

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return self._project(doc, projection)
        return None

    def insert_one(self, doc):
        from pymongo.errors import ServerSelectionTimeoutError
        if self.fail_insert:
            raise ServerSelectionTimeoutError("mock outage")
        for keys, kwargs in self.indexes:
            if kwargs.get("unique") and "dedup_key" in (
                [keys] if isinstance(keys, str) else keys
            ):
                if any(d.get("dedup_key") == doc.get("dedup_key") for d in self.docs):
                    return None
        stored = dict(doc)
        stored["_id"] = len(self.docs) + 1
        self.docs.append(stored)

        class Result:
            inserted_id = stored["_id"]
        return Result()

    def find(self, query=None, projection=None):
        rows = []
        for doc in self.docs:
            if query and not all(doc.get(k) == v for k, v in query.items()):
                continue
            rows.append(self._project(doc, projection))
        return FakeCursor(rows)


def make_pipeline(fetcher=None, collection=None, max_age=4, **kwargs):
    return p5.LivePredictionPipeline(
        data_fetcher=fetcher or FakeFetcher(),
        collection=collection or FakeCollection(),
        freshness_max_days=max_age,
        min_request_interval=0,
        **kwargs,
    )


class Phase5LivePipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = inference.validate_artifacts()

    # 1 ------------------------------------------------------------------
    def test_historical_data_retrieval_and_cache(self):
        frame = synthetic_history("AAPL")
        loader_calls = []

        def loader(sym, provider):
            loader_calls.append((sym, provider))
            return frame.copy(), provider

        fetcher = p5.MarketDataFetcher(
            source="yfinance", min_request_interval=0, cache_ttl_seconds=60,
            history_loader=loader,
        )
        first = fetcher.fetch("AAPL")
        second = fetcher.fetch("AAPL")
        forced = fetcher.fetch("AAPL", force_refresh=True)

        self.assertGreaterEqual(len(first.df), SEQUENCE_LENGTH + 30)
        self.assertEqual(first.provider, "yfinance")
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertFalse(forced.from_cache)
        self.assertEqual(len(loader_calls), 2, "cache should prevent a third API call")

    # 2 ------------------------------------------------------------------
    def test_live_provider_integration_with_mocked_responses(self):
        frame = synthetic_history("AAPL")
        # Primary provider is yfinance; Alpha Vantage is not called at all
        # when Yahoo already returns usable data.
        with patch("fetch_live_data.fetch_alpha_vantage", return_value=frame) as av, \
             patch("fetch_live_data.fetch_yfinance", return_value=frame) as yf:
            fetcher = p5.MarketDataFetcher(
                source="auto", api_key="TEST_KEY", min_request_interval=0,
                history_loader=None,
            )
            result = fetcher.fetch("AAPL")
            self.assertEqual(result.provider, "yfinance")
            yf.assert_called_once_with("AAPL", period="1y")
            av.assert_not_called()

        # When Yahoo hits a rate limit, the free-tier compact AV endpoint is used.
        with patch(
            "fetch_live_data.fetch_alpha_vantage", return_value=frame
        ) as av, patch(
            "fetch_live_data.fetch_yfinance",
            side_effect=RuntimeError("rate limit exceeded"),
        ) as yf:
            fetcher = p5.MarketDataFetcher(
                source="auto", api_key="TEST_KEY", min_request_interval=0,
                history_loader=None,
            )
            result = fetcher.fetch("MSFT")
            self.assertEqual(result.provider, "alpha_vantage")
            self.assertTrue(result.rate_limited)
            av.assert_called_once_with("MSFT", "TEST_KEY", outputsize="compact")
            yf.assert_called_once()

    # 3 ------------------------------------------------------------------
    def test_invalid_ticker_handling(self):
        self.assertEqual(p5.validate_ticker(" aapl "), "AAPL")
        for bad in ("", "AAPL;DROP", "123", "TWO WORDS", "TOOLONGTICKER123"):
            with self.subTest(ticker=bad):
                with self.assertRaises(p5.InvalidTickerError):
                    p5.validate_ticker(bad)
        result = make_pipeline().run_single("BAD SYMBOL")
        self.assertEqual(result.status, p5.STATUS_INVALID_TICKER)
        self.assertFalse(result.success)

    # 4 ------------------------------------------------------------------
    def test_insufficient_history_is_rejected(self):
        short = synthetic_history("SHORT", days=20)
        result = make_pipeline(FakeFetcher({"SHORT": short})).run_single("SHORT")
        self.assertEqual(result.status, p5.STATUS_INCOMPATIBLE)
        self.assertTrue(any("Need at least 60" in e for e in result.errors))
        self.assertIsNone(result.prediction)

    # 5 ------------------------------------------------------------------
    def test_missing_ohlcv_values_are_rejected(self):
        broken = synthetic_history("BROKEN")
        broken["Close"] = np.nan
        result = make_pipeline(FakeFetcher({"BROKEN": broken})).run_single("BROKEN")
        self.assertEqual(result.status, p5.STATUS_INCOMPATIBLE)
        self.assertFalse(result.success)

    # 6 ------------------------------------------------------------------
    def test_feature_ordering_consistency(self):
        metadata = read_model_metadata()
        self.assertEqual(metadata["feature_columns"], FINAL_FEATURE_COLUMNS)
        self.assertEqual(FINAL_FEATURE_COLUMNS, list(FEATURE_COLUMNS))
        result = make_pipeline().run_single("AAPL")
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.feature_config, self.report.feature_config_version)

    # 7 ------------------------------------------------------------------
    def test_scaler_transform_only(self):
        from sklearn.preprocessing import MinMaxScaler

        original_transform = MinMaxScaler.transform
        transform_calls = []

        def spy(self, X):
            transform_calls.append(np.asarray(X).shape)
            return original_transform(self, X)

        with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("fit called")), \
             patch.object(MinMaxScaler, "fit_transform", side_effect=AssertionError("fit_transform called")), \
             patch.object(MinMaxScaler, "transform", spy):
            result = make_pipeline().run_single("AAPL", persist=False)

        self.assertTrue(result.success, result.errors)
        self.assertTrue(transform_calls)

    # 8 ------------------------------------------------------------------
    def test_correct_60_by_21_sequence(self):
        _, scaler, _ = inference.load_bundle()
        X = legacy.build_prediction_sequence(synthetic_history("SEQ"), scaler, symbol="SEQ")
        self.assertEqual(X.shape, (1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue((X >= 0).all() and (X <= 1).all())

    # 9 ------------------------------------------------------------------
    def test_existing_lstm_inference_integration(self):
        result = make_pipeline().run_single("AAPL")
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.model_sha256, self.report.model_sha256)
        self.assertIn(result.prediction, ("UP", "DOWN"))
        self.assertAlmostEqual(sum(result.probabilities.values()), 100.0, places=2)
        self.assertAlmostEqual(result.probabilities["UP"] / 100.0, result.raw_output, places=4)
        self.assertTrue(result.prediction_timestamp)

    # 10 -----------------------------------------------------------------
    def test_multi_stock_processing(self):
        fetcher = FakeFetcher()
        pipeline = make_pipeline(fetcher)
        results = pipeline.run_many(
            ["AAPL", "BAD SYMBOL", "MSFT", "NVDA"], max_workers=2
        )
        self.assertEqual([r.symbol for r in results], ["AAPL", "BAD SYMBOL", "MSFT", "NVDA"])
        self.assertEqual(results[1].status, p5.STATUS_INVALID_TICKER)
        self.assertTrue(all(r.success for r in (results[0], results[2], results[3])))
        self.assertEqual(len(fetcher.calls), 3)

    # 11 -----------------------------------------------------------------
    def test_duplicate_prediction_prevention(self):
        collection = FakeCollection()
        pipeline = make_pipeline(collection=collection)
        first = pipeline.run_single("DUP")
        second = pipeline.run_single("DUP")
        self.assertTrue(first.success and second.success)
        self.assertEqual(first.mongo_id, second.mongo_id)
        self.assertEqual(len(collection.docs), 1)
        record = collection.docs[0]
        self.assertEqual(record["model"], "LSTM")
        self.assertIn("provider", record["meta"])
        self.assertIn("data_freshness", record["meta"])
        self.assertIn("feature_config_version", record["meta"])

    # 12 -----------------------------------------------------------------
    def test_mongodb_failure_handling_and_retry(self):
        broken = FakeCollection(fail_insert=True)
        pipeline = make_pipeline(collection=broken)
        result = pipeline.run_single("MONGO")
        self.assertTrue(result.success)
        self.assertFalse(result.persisted)
        self.assertEqual(pipeline.monitor.mongo_failures, 1)
        self.assertEqual(len(pipeline._pending_records), 1)

        working = FakeCollection()
        pipeline.collection = working
        self.assertEqual(pipeline.retry_pending(), 1)
        self.assertEqual(len(working.docs), 1)
        self.assertEqual(pipeline._pending_records, [])

    # 13 -----------------------------------------------------------------
    def test_api_rate_limit_handling(self):
        def loader(symbol, provider):
            if provider == "yfinance":
                raise RuntimeError("API rate limit exceeded on free tier")
            return synthetic_history(symbol), provider

        fetcher = p5.MarketDataFetcher(
            source="auto", api_key="KEY", min_request_interval=0, history_loader=loader
        )
        pipeline = make_pipeline(fetcher)
        result = pipeline.run_single("RATE")
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.provider, "alpha_vantage")
        self.assertTrue(result.freshness.get("rate_limited"))
        self.assertEqual(pipeline.monitor.rate_limit_errors, 1)

    # 14 -----------------------------------------------------------------
    def test_stale_data_detection(self):
        old = synthetic_history("STALE", end=pd.Timestamp("2026-11-01"))
        pipeline = make_pipeline(FakeFetcher({"STALE": old}))
        result = pipeline.run_single("STALE", now=FIXED_NOW)
        self.assertEqual(result.status, p5.STATUS_STALE)
        self.assertEqual(result.freshness["status"], p5.FRESH_STALE)
        self.assertGreater(result.freshness["age_days"], 4)
        self.assertTrue(any("stale" in w.lower() for w in result.warnings))
        self.assertTrue(result.success, "stale data must still be clearly labelled")

    # 15 -----------------------------------------------------------------
    def test_market_hours_handling(self):
        weekend = p5.market_session_state(datetime(2026, 11, 28, 12, 0, tzinfo=ZoneInfo("America/New_York")))
        opening = p5.market_session_state(datetime(2026, 11, 24, 10, 0, tzinfo=ZoneInfo("America/New_York")))
        closed = p5.market_session_state(FIXED_NOW)
        self.assertEqual(weekend["phase"], "weekend")
        self.assertEqual(opening["phase"], "open")
        self.assertTrue(opening["is_intraday"])
        self.assertEqual(closed["phase"], "after_close")

        intraday_now = datetime(2026, 11, 24, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        frame = synthetic_history("INTRA")
        freshness = p5.classify_freshness(frame, now=intraday_now)
        self.assertTrue(freshness.has_incomplete_bar)
        cleaned, dropped = p5.drop_incomplete_bars(frame, freshness)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(cleaned), len(frame) - 1)

        pipeline = make_pipeline(FakeFetcher({"INTRA": frame}))
        result = pipeline.run_single("INTRA", now=intraday_now)
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.dropped_incomplete_bars, 1)
        self.assertEqual(result.freshness["status"], p5.FRESH_FRESH)

    # 16 -----------------------------------------------------------------
    def test_pipeline_recovery_after_failure(self):
        class FlakyFetcher(FakeFetcher):
            def fetch(self, symbol, force_refresh=False):
                if not self.calls:
                    self.calls.append((symbol, force_refresh))
                    raise p5.MarketDataError("temporary provider outage")
                return super().fetch(symbol, force_refresh)

        pipeline = make_pipeline(FlakyFetcher())
        failed = pipeline.run_single("RECOVER")
        self.assertEqual(failed.status, p5.STATUS_API_FAILURE)
        recovered = pipeline.run_single("RECOVER")
        self.assertTrue(recovered.success, recovered.errors)
        self.assertEqual(pipeline.monitor.api_failures, 1)

    # 17 -----------------------------------------------------------------
    def test_streamlit_compatibility(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        required = [
            "LivePredictionPipeline", "p5_run", "p5_refresh", "p5_batch",
            "p5_monitor", "Pipeline monitoring", "Filter by stock",
            "Model prediction (not investment advice)", "Multi-Stock Inference",
            "Recent LSTM Records (MongoDB)",
        ]
        for token in required:
            self.assertIn(token, source, f"dashboard hook missing: {token}")
        called = {
            n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertIn("LivePredictionPipeline", called)
        self.assertIn("save_lstm_prediction", called)
        self.assertIn("predict_live_lstm", called)

    # 18 -----------------------------------------------------------------
    def test_no_training_methods_are_called(self):
        module_path = ROOT / "scripts" / "live_prediction_pipeline.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        forbidden = {"fit", "fit_transform", "train", "build_lstm_model"}
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
            result = make_pipeline().run_single("NOTRAIN", persist=False)
        self.assertTrue(result.success, result.errors)

    # 19 -----------------------------------------------------------------
    def test_bounded_scheduler(self):
        pipeline = make_pipeline()
        runs = pipeline.run_scheduled(
            ["AAPL"], interval_seconds=0, max_iterations=2,
            market_hours_only=False, now=FIXED_NOW,
        )
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(r[0].success for r in runs))

    # 20 -----------------------------------------------------------------
    def test_request_throttling_and_unknown_symbol_warning(self):
        calls = []

        def loader(sym, provider):
            calls.append(time.monotonic())
            return synthetic_history(sym), provider

        fetcher = p5.MarketDataFetcher(
            source="yfinance", min_request_interval=0.12,
            cache_ttl_seconds=0, history_loader=loader,
        )
        fetcher.fetch("A", force_refresh=True)
        fetcher.fetch("B", force_refresh=True)
        self.assertGreaterEqual(calls[1] - calls[0], 0.08)

        compat = p5.check_symbol_compatibility("ZZZZ")
        self.assertTrue(compat.warnings)
        self.assertIn("not in the bundled", compat.warnings[0])

    # 21 -----------------------------------------------------------------
    def test_model_artifacts_unchanged(self):
        import hashlib
        digest = hashlib.sha256(inference.MODEL_FILE.read_bytes()).hexdigest()
        self.assertEqual(digest, self.report.model_sha256)


if __name__ == "__main__":
    unittest.main()

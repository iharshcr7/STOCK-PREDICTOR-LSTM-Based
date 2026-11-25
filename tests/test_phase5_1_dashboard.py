"""
Phase 5.1 regression tests — provider fixes, result handling, dashboard.

All market data is mocked; no API quota is consumed and no training occurs.

    py -3.11 -m unittest discover -s tests -p "test_phase5_1_dashboard.py" -v
"""

import ast
import hashlib
import io
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import live_prediction_pipeline as p5  # noqa: E402
import lstm_inference as inference  # noqa: E402
import fetch_live_data  # noqa: E402

NOW = datetime(2026, 11, 24, 17, 0, tzinfo=ZoneInfo("America/New_York"))
END = pd.Timestamp("2026-11-24")


def frame(symbol, base=150.0, days=130, end=END, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=days + 3)[-days:]
    close = base + np.cumsum(rng.normal(0.05, 1.0, days))
    return pd.DataFrame({
        "Symbol": symbol,
        "Date": dates,
        "Open": close - 0.2, "High": close + 0.6, "Low": close - 0.6,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, days).astype(float),
    })


class Fetcher:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def fetch(self, symbol, force_refresh=False):
        self.calls.append((symbol, force_refresh))
        value = self.responses.get(symbol)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, p5.MarketData):
            return value
        df = value if value is not None else frame(symbol)
        return p5.MarketData(symbol=symbol, df=df, provider="yfinance")


class Collection:
    def __init__(self):
        self.docs = []
        self.indexes = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return {"_id": len(self.docs)}
        return None

    def insert_one(self, doc):
        stored = dict(doc)
        stored["_id"] = len(self.docs) + 1
        self.docs.append(stored)

        class Result:
            inserted_id = stored["_id"]
        return Result()

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


def pipeline(fetcher=None, collection=None):
    return p5.LivePredictionPipeline(
        data_fetcher=fetcher or Fetcher(),
        collection=collection or Collection(),
        min_request_interval=0,
    )


class Phase51DashboardTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = inference.validate_artifacts()
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    # 1 ------------------------------------------------------------------
    def test_boolean_iterable_crash_regression(self):
        for bad in (True, False, None, {}, "", 42, object()):
            with self.subTest(value=bad):
                self.assertEqual(p5.coerce_results(bad), [])
        self.assertIsInstance(p5.coerce_results([True, None]), list)
        self.assertEqual(p5.coerce_results([True, None]), [])
        # The widget key must no longer be reused as a result key.
        self.assertIn("p5_batch_btn", self.app_source)
        self.assertIn("st.session_state['p5_batch_results']", self.app_source)
        self.assertNotIn("st.session_state['p5_batch'] =", self.app_source)

    # 2 ------------------------------------------------------------------
    def test_successful_single_stock_prediction(self):
        result = pipeline().run_single("AAPL", now=NOW)
        self.assertTrue(result.success, result.errors)
        self.assertIn(result.prediction, ("UP", "DOWN"))
        self.assertTrue(result.chart_data.get("Close"))
        self.assertEqual(len(result.chart_data["Date"]), len(result.chart_data["Close"]))

    # 3 ------------------------------------------------------------------
    def test_failed_single_stock_prediction(self):
        failing = Fetcher({"FAIL": p5.MarketDataError("provider unavailable")})
        result = pipeline(failing).run_single("FAIL")
        self.assertFalse(result.success)
        self.assertEqual(result.status, p5.STATUS_API_FAILURE)
        self.assertTrue(result.errors)
        self.assertFalse(result.chart_data)

    # 4 ------------------------------------------------------------------
    def test_partial_success_multi_stock(self):
        responses = {
            "GOOD": frame("GOOD"),
            "BAD": p5.MarketDataError("no usable data"),
        }
        results = pipeline(Fetcher(responses)).run_many(["GOOD", "BAD", "BAD TICKER"])
        summary = p5.summarize_results(results)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertTrue(summary["partial_success"])

    # 5 ------------------------------------------------------------------
    def test_empty_api_responses(self):
        empty = p5.MarketData(symbol="EMPTY", df=pd.DataFrame(), provider="yfinance")
        result = pipeline(Fetcher({"EMPTY": empty})).run_single("EMPTY")
        self.assertFalse(result.success)
        self.assertEqual(result.status, p5.STATUS_INCOMPATIBLE)

        def loader(sym, provider):
            return pd.DataFrame(), provider
        fetcher = p5.MarketDataFetcher(source="yfinance", min_request_interval=0, history_loader=loader)
        with self.assertRaises(p5.MarketDataError) as ctx:
            fetcher.fetch("EMPTY2")
        self.assertIn("empty response", str(ctx.exception))

    # 6 ------------------------------------------------------------------
    def test_invalid_and_delisted_symbols(self):
        self.assertEqual(p5.normalize_ticker_candidates(" $ADVM "), ["ADVM"])
        with self.assertRaises(p5.InvalidTickerError):
            p5.validate_ticker("$ADVM")

        def loader(sym, provider):
            raise RuntimeError(f"{sym}: No data found, symbol may be delisted")
        fetcher = p5.MarketDataFetcher(source="yfinance", min_request_interval=0, history_loader=loader)
        with self.assertRaises(p5.MarketDataError) as ctx:
            fetcher.fetch("$ADVM")
        self.assertIn("delisted", str(ctx.exception))
        result = pipeline(fetcher).run_single("$ADVM")
        self.assertFalse(result.success)
        self.assertTrue(result.errors)

    # 7 ------------------------------------------------------------------
    def test_alpha_vantage_free_tier_handling(self):
        premium = ValueError("outputsize=full is a premium feature")
        self.assertTrue(p5._is_premium(premium))
        self.assertFalse(p5._is_rate_limit(premium))

        def loader(sym, provider):
            if provider == "alpha_vantage":
                raise premium
            if provider == "yfinance":
                raise RuntimeError("rate limit exceeded")
            return frame(sym), provider
        fetcher = p5.MarketDataFetcher(source="auto", api_key="KEY", min_request_interval=0, history_loader=loader)
        result = pipeline(fetcher).run_single("AVFREE")
        self.assertTrue(result.success, result.errors)
        self.assertEqual(result.provider, "local")
        self.assertEqual(result.data_vintage, "local_historical")
        self.assertTrue(any("premium" in w.lower() for w in result.warnings))
        self.assertTrue(any("NOT live" in w for w in result.warnings))

        with patch("fetch_live_data.requests.get") as get:
            get.return_value.json.return_value = {
                "Information": "outputsize=full is a premium feature"
            }
            with self.assertRaises(ValueError) as ctx:
                fetch_live_data.fetch_alpha_vantage("AAPL", "KEY", outputsize="full")
            self.assertIn("premium restriction", str(ctx.exception))

    # 8 ------------------------------------------------------------------
    def test_yfinance_fallback(self):
        with patch("fetch_live_data.fetch_yfinance", side_effect=RuntimeError("rate limit")), \
             patch("fetch_live_data.fetch_alpha_vantage", return_value=frame("YF")) as av:
            fetcher = p5.MarketDataFetcher(source="auto", api_key="KEY", min_request_interval=0)
            result = fetcher.fetch("YF")
            self.assertEqual(result.provider, "alpha_vantage")
            av.assert_called_once_with("YF", "KEY", outputsize="compact")

    # 9 ------------------------------------------------------------------
    def test_local_historical_data_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "LOCAL.csv").write_text(
                frame("LOCAL").to_csv(index=False), encoding="utf-8"
            )
            with patch.object(p5, "LOCAL_STOCK_DIR", directory):
                df, provider = p5.MarketDataFetcher._load_local_history("LOCAL")
            self.assertEqual(provider, "local")
            self.assertLessEqual(len(df), 400)
            self.assertTrue({"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns))

    # 10 -----------------------------------------------------------------
    def test_stale_data_warnings(self):
        old = frame("STALE", end=pd.Timestamp("2026-11-01"))
        result = pipeline(Fetcher({"STALE": old})).run_single("STALE", now=NOW)
        self.assertEqual(result.status, p5.STATUS_STALE)
        self.assertEqual(result.freshness["status"], p5.FRESH_STALE)
        self.assertTrue(any("stale" in w.lower() for w in result.warnings))

    # 11 -----------------------------------------------------------------
    def test_multi_stock_result_normalization(self):
        good = pipeline().run_single("AAPL", now=NOW)
        payload = [good, {"symbol": "X", "status": "success", "success": True}, True, None]
        normalized = p5.coerce_results(payload)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0].symbol, "AAPL")
        self.assertEqual(normalized[1].symbol, "X")
        self.assertEqual(p5.summarize_results(normalized)["successful"], 2)

    # 12 -----------------------------------------------------------------
    def test_individual_chart_data_separation(self):
        responses = {"AAPL": frame("AAPL", 150.0), "MSFT": frame("MSFT", 420.0, seed=99)}
        results = pipeline(Fetcher(responses)).run_many(["AAPL", "MSFT"], now=NOW)
        self.assertTrue(all(r.success for r in results))
        aapl, msft = results
        self.assertNotEqual(aapl.chart_data["Close"], msft.chart_data["Close"])
        self.assertGreater(min(msft.chart_data["Close"]), max(aapl.chart_data["Close"]))
        for token in ("render_symbol_analysis", "go.Candlestick", "st.tabs", "p5_prob_"):
            self.assertIn(token, self.app_source)

    # 13 -----------------------------------------------------------------
    def test_mongodb_deduplication(self):
        col = Collection()
        pl = pipeline(collection=col)
        first = pl.run_single("DUP", now=NOW)
        second = pl.run_single("DUP", now=NOW)
        self.assertEqual(first.mongo_id, second.mongo_id)
        self.assertEqual(len(col.docs), 1)

    # 14 -----------------------------------------------------------------
    def test_streamlit_rerun_behavior(self):
        self.assertIn("width=\"stretch\"", self.app_source)
        self.assertNotIn("use_container_width", self.app_source)
        self.assertIn("coerce_results(", self.app_source)
        self.assertIn("st.session_state.get('p5_last_result')", self.app_source)
        tree = ast.parse(self.app_source)
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                for kw in node.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                        keys.add(kw.value.value)
        self.assertTrue({"p5_run_btn", "p5_refresh_btn", "p5_batch_btn"}.issubset(keys))
        self.assertFalse({"p5_run", "p5_refresh", "p5_batch"}.intersection(keys))

    # 15 -----------------------------------------------------------------
    def test_no_repeated_api_requests_unnecessarily(self):
        calls = []

        def loader(sym, provider):
            calls.append((sym, provider))
            return frame(sym), provider

        fetcher = p5.MarketDataFetcher(
            source="yfinance", min_request_interval=0, cache_ttl_seconds=300,
            history_loader=loader,
        )
        fetcher.fetch("CACHED")
        fetcher.fetch("CACHED")
        self.assertEqual(len(calls), 1)
        fetcher.fetch("CACHED", force_refresh=True)
        self.assertEqual(len(calls), 2)

    # 16 -----------------------------------------------------------------
    def test_no_training_or_model_modification(self):
        digest = hashlib.sha256(inference.MODEL_FILE.read_bytes()).hexdigest()
        self.assertEqual(digest, self.report.model_sha256)
        tree = ast.parse((ROOT / "scripts" / "live_prediction_pipeline.py").read_text(encoding="utf-8"))
        forbidden = {"fit", "fit_transform", "train", "build_lstm_model"}
        called = {
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertFalse(forbidden & called)

        from sklearn.preprocessing import MinMaxScaler
        from tensorflow import keras
        with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("fit called")), \
             patch.object(MinMaxScaler, "fit_transform", side_effect=AssertionError("fit_transform called")), \
             patch.object(keras.Model, "fit", side_effect=AssertionError("model.fit called")):
            result = pipeline().run_single("NOTRAIN", now=NOW, persist=False)
        self.assertTrue(result.success, result.errors)

    # 17 -----------------------------------------------------------------
    def test_existing_phase4_tests_still_pass(self):
        suite = unittest.defaultTestLoader.loadTestsFromName("test_phase4_lstm_inference")
        self.assertGreater(suite.countTestCases(), 0)
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        self.assertTrue(result.wasSuccessful(), str(result.errors) + str(result.failures))


    # 18 -----------------------------------------------------------------
    def test_dashboard_apptest_batch_no_crash(self):
        """Run the real Streamlit script with a mocked batch pipeline."""
        from types import SimpleNamespace
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:  # pragma: no cover - older Streamlit
            self.skipTest("streamlit.testing.AppTest unavailable")

        def make_result(symbol, ok=True):
            return p5.PipelineResult(
                symbol=symbol, status="success" if ok else "api_failure", success=ok,
                prediction="UP" if ok else None, confidence=56.1 if ok else 0.0,
                probabilities={"UP": 56.1, "DOWN": 43.9} if ok else {},
                raw_output=0.561 if ok else 0.0,
                provider="yfinance", data_vintage="delayed_eod",
                freshness={"status": "fresh", "age_days": 0},
                market_data_timestamp="2026-11-24", model_sha256="abc123",
                chart_data=p5.build_chart_data(frame(symbol, 150 if symbol == "AAPL" else 420))
                if ok else {},
            )

        class FakePipeline:
            def __init__(self, *args, **kwargs):
                self.monitor = SimpleNamespace(snapshot=lambda: {
                    "api_failures": 1, "rate_limit_errors": 0,
                    "validation_failures": 0, "mongo_failures": 0,
                    "last_successful_fetch": {"AAPL": "now"},
                    "last_successful_inference": {"AAPL": "now"},
                    "latest_market_data_timestamp": "2026-11-24",
                    "per_symbol_status": {},
                })

            def run_many(self, symbols, **kwargs):
                return [make_result(s, ok=(s != "BAD")) for s in symbols]

            def run_single(self, symbol, **kwargs):
                return make_result(symbol)

        with patch.object(p5, "LivePredictionPipeline", FakePipeline):
            app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
            self.assertEqual([str(e.value) for e in app_test.exception], [])
            app_test.session_state["p5_ticker_input"] = "AAPL, MSFT, BAD"
            app_test.button(key="p5_batch_btn").click().run()

        self.assertEqual(
            [str(e.value) for e in app_test.exception], [],
            "dashboard raised an application-level exception during batch inference",
        )
        self.assertEqual(len(app_test.tabs), 3, "expected Overview + 2 successful symbols")


if __name__ == "__main__":
    unittest.main()

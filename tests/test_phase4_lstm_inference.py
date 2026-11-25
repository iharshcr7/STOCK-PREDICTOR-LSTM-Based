"""
Phase 4 tests — existing-model inference integration only.

No training, retraining, fine-tuning or model construction is performed here.
The real trained checkpoint (models/lstm_model.keras) and its fitted scaler are
loaded read-only. Run with:

    py -3.11 -m unittest tests.test_phase4_lstm_inference -v
"""

import ast
import json
import os
import sys
import unittest
from pathlib import Path

# Force CPU execution before TensorFlow is imported by the pipeline modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import predict_lstm as legacy  # noqa: E402
import lstm_inference as inference  # noqa: E402
from lstm_data_pipeline import FEATURE_COLUMNS, SEQUENCE_LENGTH  # noqa: E402
from model_config import FINAL_FEATURE_COLUMNS, read_model_metadata  # noqa: E402
from mongo_store import save_lstm_prediction  # noqa: E402


def synthetic_history(symbol="TEST", days=130, seed=2024):
    """Small deterministic OHLCV frame — enough for a 60-session sequence."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    steps = rng.normal(0.05, 1.2, size=days)
    close = 150.0 + np.cumsum(steps)
    return pd.DataFrame({
        "Symbol": symbol,
        "Date": dates,
        "Open": close - 0.25,
        "High": close + np.abs(steps) + 0.5,
        "Low": close - np.abs(steps) - 0.5,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, size=days).astype(float),
    })


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
    """In-memory stand-in for a MongoDB collection (no server required)."""

    def __init__(self):
        self.docs = []
        self.indexes = []

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
        for keys, kwargs in self.indexes:
            if kwargs.get("unique") and "dedup_key" in (
                [keys] if isinstance(keys, str) else keys
            ):
                if any(d.get("dedup_key") == doc.get("dedup_key") for d in self.docs):
                    from pymongo.errors import DuplicateKeyError
                    raise DuplicateKeyError("duplicate dedup_key")
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


class Phase4InferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metadata = read_model_metadata()
        cls.report = inference.validate_artifacts()
        cls.df = synthetic_history()

    # 1 ------------------------------------------------------------------
    def test_existing_model_loads(self):
        self.assertTrue(legacy.is_lstm_available())
        model, scaler, report = inference.load_bundle()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(tuple(model.input_shape[1:]), (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
        self.assertEqual(tuple(model.output_shape[1:]), (1,))
        self.assertEqual(report.model_sha256, inference.validate_artifacts().model_sha256)

    # 2 ------------------------------------------------------------------
    def test_correct_scaler_is_loaded(self):
        scaler = legacy.load_lstm_scaler()
        self.assertEqual(type(scaler).__name__, "MinMaxScaler")
        self.assertEqual(int(scaler.n_features_in_), len(FEATURE_COLUMNS))
        self.assertEqual(tuple(scaler.feature_range), (0, 1))
        self.assertEqual(self.report.scaler_type, "MinMaxScaler")
        self.assertEqual(self.report.scaler_n_features, len(FEATURE_COLUMNS))
        # Cached loader must return the same fitted object, never a new scaler.
        self.assertIs(scaler, legacy.load_lstm_scaler())

    # 3 ------------------------------------------------------------------
    def test_feature_count_and_ordering_match(self):
        expected = self.metadata["feature_columns"]
        self.assertEqual(len(expected), 21)
        self.assertEqual(expected, FINAL_FEATURE_COLUMNS)
        self.assertEqual(expected, list(FEATURE_COLUMNS))
        self.assertEqual(expected, self.report.feature_columns)

    # 4 ------------------------------------------------------------------
    def test_sequence_length_matches_original(self):
        model, _, _ = inference.load_bundle()
        self.assertEqual(int(self.metadata["sequence_length"]), 60)
        self.assertEqual(self.report.sequence_length, SEQUENCE_LENGTH)
        self.assertEqual(model.input_shape[1], SEQUENCE_LENGTH)

    # 5 ------------------------------------------------------------------
    def test_input_tensor_shape_is_correct(self):
        _, scaler, _ = inference.load_bundle()
        X = legacy.build_prediction_sequence(self.df, scaler, symbol="TEST")
        self.assertEqual(X.shape, (1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue((X >= 0.0).all() and (X <= 1.0).all())

    # 6 ------------------------------------------------------------------
    def test_model_output_is_interpreted_correctly(self):
        result = inference.predict_from_frame("TEST", self.df, source="local")
        self.assertTrue(result["ok"], result.get("errors"))
        self.assertIn(result["prediction"], ("UP", "DOWN"))
        probs = result["probabilities"]
        self.assertAlmostEqual(probs["UP"] + probs["DOWN"], 100.0, places=2)
        expected = "UP" if result["raw_output"] >= self.metadata["prediction_config"]["threshold"] else "DOWN"
        self.assertEqual(result["prediction"], expected)
        self.assertAlmostEqual(result["confidence"], max(probs.values()), places=2)
        # Capabilities reflect the ACTUAL model, not the aspirational objectives.
        self.assertTrue(result["capabilities"]["direction_binary"])
        self.assertFalse(result["capabilities"]["three_class"])
        self.assertFalse(result["capabilities"]["next_session_return"])
        self.assertIsNone(result["predicted_next_return_pct"])
        self.assertIn("binary sigmoid", result["next_return_status"])

    # 7 ------------------------------------------------------------------
    def test_missing_or_incompatible_data_is_handled_safely(self):
        cases = {
            "none": None,
            "empty": pd.DataFrame(),
            "no_columns": pd.DataFrame({"Close": [1, 2, 3]}),
            "short_history": synthetic_history(days=20),
        }
        for label, frame in cases.items():
            with self.subTest(case=label):
                result = inference.predict_from_frame("BAD", frame)
                self.assertFalse(result["ok"])
                self.assertTrue(result["errors"])
                self.assertIsNone(result["prediction"])
                self.assertIsNone(result["predicted_next_return_pct"])

    # 7b -----------------------------------------------------------------
    def test_historical_gaps_are_reported_not_silent(self):
        moderate_gap = synthetic_history(symbol="GAP")
        moderate_gap.loc[moderate_gap.index[-1], "Date"] += pd.Timedelta(days=12)
        result = inference.predict_from_frame("GAP", moderate_gap)
        self.assertTrue(result["ok"], result.get("errors"))
        self.assertTrue(any("historical gap" in w for w in result["warnings"]))

        severe_gap = synthetic_history(symbol="BADGAP")
        severe_gap.loc[severe_gap.index[-1], "Date"] += pd.Timedelta(days=45)
        blocked = inference.predict_from_frame("BADGAP", severe_gap)
        self.assertFalse(blocked["ok"])
        self.assertTrue(any("data gap" in e for e in blocked["errors"]))

    # 8 ------------------------------------------------------------------
    def test_inference_runs_on_cpu(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1")
        result = inference.predict_from_frame("CPU", synthetic_history(symbol="CPU"))
        self.assertTrue(result["ok"], result.get("errors"))
        self.assertIsInstance(result["raw_output"], float)

    # 9 ------------------------------------------------------------------
    def test_mongodb_storage_without_duplicates(self):
        collection = FakeCollection()
        result = inference.predict_from_frame("DUP", synthetic_history(symbol="DUP"))
        record = inference.to_storage_record(result)

        first_id = save_lstm_prediction(record, collection=collection)
        second_id = save_lstm_prediction(record, collection=collection)

        self.assertEqual(first_id, second_id)
        self.assertEqual(len(collection.docs), 1)
        self.assertTrue(any(k == "dedup_key" for k, _ in collection.indexes))
        stored = collection.docs[0]
        self.assertEqual(stored["model"], "LSTM")
        self.assertEqual(stored["symbol"], "DUP")
        self.assertIsNone(stored["meta"]["predicted_next_return_pct"])
        self.assertEqual(stored["meta"]["sequence_length"], 60)

    # 10 -----------------------------------------------------------------
    def test_streamlit_integration_preserves_existing_functionality(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(app_source)  # fails loudly on a broken integration

        required_existing = [
            "get_historical_for_chart",
            "compute_live_features",
            "create_lstm_forecast_chart",
            "create_lstm_backtest_chart",
            "create_lstm_metric_chart",
            "run_lstm_backtest",
        ]
        for name in required_existing:
            self.assertIn(name, app_source, f"existing dashboard feature removed: {name}")

        required_phase4 = [
            "predict_live_lstm",
            "infer_symbols_with_existing_lstm",
            "lstm_storage_record",
            "fetch_lstm_predictions",
            "Multi-Stock Inference",
            "Next-session return",
        ]
        for name in required_phase4:
            self.assertIn(name, app_source, f"Phase 4 UI hook missing: {name}")

        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("save_lstm_prediction", called)
        self.assertIn("predict_live_lstm", called)
        self.assertIn("infer_symbols_with_existing_lstm", called)

    # 11 -----------------------------------------------------------------
    def test_discovery_reports_artifact_details(self):
        report = self.report.to_dict()
        self.assertEqual(report["model_format"], ".keras")
        self.assertEqual(report["input_shape"], [None, 60, 21])
        self.assertEqual(report["output_shape"], [None, 1])
        self.assertEqual(report["output_activation"], "sigmoid")
        self.assertEqual(report["label_mapping"], {"0": "DOWN", "1": "UP"})
        self.assertEqual(report["threshold"], 0.5)
        self.assertTrue(report["model_sha256"])
        self.assertFalse(report["capabilities"]["three_class"])
        self.assertFalse(report["capabilities"]["next_session_return"])
        self.assertIn("keras_version", json.dumps(report))


if __name__ == "__main__":
    unittest.main()

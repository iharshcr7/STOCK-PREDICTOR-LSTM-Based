"""
tests/test_phase5_2_3_market_payload.py
======================================
Phase 5.2.3 — Market analytics payload normalization & OHLCV validation tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import stock_intelligence_pipeline as sip  # noqa: E402


def _sample_df(rows: int = 220, symbol: str = "AAPL") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    base = np.linspace(100.0, 150.0, rows)
    return pd.DataFrame({
        "Date": dates,
        "Symbol": [symbol] * rows,
        "Open": base - 1.0,
        "High": base + 2.5,
        "Low": base - 2.5,
        "Close": base,
        "Volume": np.full(rows, 1_000_000.0),
        "MA20": pd.Series(base).rolling(20, min_periods=20).mean(),
        "RSI": np.full(rows, 55.0),
        "MACD": np.full(rows, 0.5),
        "MACD_Signal": np.full(rows, 0.3),
        "BB_Width": np.full(rows, 4.0),
        "Daily_Returns": np.full(rows, 0.2),
    })


class Phase523MarketPayloadTests(unittest.TestCase):
    def test_dataframe_payload(self):
        df = _sample_df(30, symbol="MSFT")
        frame, meta = sip.normalize_market_payload(df)
        self.assertEqual(len(frame), 30)
        self.assertEqual(meta["symbol"], "MSFT")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["Date"]))
        self.assertTrue(frame["Date"].is_monotonic_increasing)

    def test_list_of_dicts_payload(self):
        records = [
            {"Date": "2026-11-24", "Open": 102, "High": 106, "Low": 101, "Close": 105, "Volume": 120000},
            {"Date": "2026-11-23", "Open": 100, "High": 105, "Low": 98, "Close": 103, "Volume": 100000},
        ]
        frame, meta = sip.normalize_market_payload(records, symbol="AAPL")
        self.assertEqual(len(frame), 2)
        self.assertEqual(meta["symbol"], "AAPL")
        # Sorted chronologically
        self.assertEqual(str(frame["Date"].iloc[0].date()), "2026-11-23")
        self.assertEqual(str(frame["Date"].iloc[1].date()), "2026-11-24")

    def test_nested_market_data_dictionary(self):
        for key in ("data", "records", "ohlcv", "prices"):
            payload = {
                "symbol": "AAPL",
                "provider": "yfinance",
                key: [
                    {
                        "Date": "2026-11-23",
                        "Open": 100,
                        "High": 105,
                        "Low": 98,
                        "Close": 103,
                        "Volume": 100000,
                    }
                ],
            }
            frame, meta = sip.normalize_market_payload(payload)
            self.assertEqual(len(frame), 1, msg=f"Failed for container key={key}")
            self.assertEqual(meta["symbol"], "AAPL")
            self.assertEqual(meta["provider"], "yfinance")
            self.assertNotIn(key, meta)

    def test_column_oriented_dictionary(self):
        payload = {
            "Date": pd.Series(["2026-11-23", "2026-11-24"]),
            "Open": [100.0, 102.0],
            "High": [105.0, 107.0],
            "Low": [98.0, 101.0],
            "Close": np.array([103.0, 106.0]),
            "Volume": [100000, 110000],
        }
        frame, meta = sip.normalize_market_payload(payload, symbol="NVDA")
        self.assertEqual(len(frame), 2)
        self.assertEqual(meta["symbol"], "NVDA")
        self.assertEqual(list(frame["Close"]), [103.0, 106.0])

    def test_single_ohlcv_record(self):
        payload = {
            "symbol": "GOOG",
            "Date": "2026-11-23",
            "Open": 180.0,
            "High": 184.0,
            "Low": 179.0,
            "Close": 182.5,
            "Volume": 500000,
        }
        frame, meta = sip.normalize_market_payload(payload)
        self.assertEqual(len(frame), 1)
        self.assertEqual(meta["symbol"], "GOOG")
        self.assertEqual(float(frame["Close"].iloc[0]), 182.5)

    def test_mixed_metadata_and_ohlcv_payload_from_build_market_payload(self):
        """Reproduce the exact Phase 5.2 ValueError: Mixing dicts with non-Series."""
        raw = _sample_df(60, symbol="AAPL")
        payload = sip.build_market_payload(raw, "AAPL")
        # Confirm payload has both list columns, scalar symbol, empty Adj Close list, and nested dict latest_metrics
        self.assertIsInstance(payload["latest_metrics"], dict)
        self.assertEqual(payload["Adj Close"], [])
        frame, meta = sip.normalize_market_payload(payload)
        self.assertEqual(len(frame), 60)
        self.assertEqual(meta["symbol"], "AAPL")
        self.assertIn("latest_metrics", meta)
        self.assertAlmostEqual(meta["latest_metrics"]["close"], float(raw["Close"].iloc[-1]))

    def test_empty_payload(self):
        for empty in (None, {}, [], pd.DataFrame()):
            frame, meta = sip.normalize_market_payload(empty)
            self.assertTrue(frame.empty)
            self.assertIsInstance(meta, dict)

    def test_invalid_payload_and_ohlc_validation(self):
        # Unsupported type
        frame, meta = sip.normalize_market_payload(12345)
        self.assertTrue(frame.empty)
        self.assertIn("validation_error", meta)

        # Invalid OHLC relationship (High < Low) and duplicate date
        records = [
            {"Date": "2026-11-23", "Open": 100, "High": 95, "Low": 98, "Close": 99},  # invalid High < Low
            {"Date": "2026-11-24", "Open": 100, "High": 105, "Low": 98, "Close": 103},
            {"Date": "2026-11-24", "Open": 101, "High": 106, "Low": 99, "Close": 104},  # duplicate
        ]
        frame, meta = sip.normalize_market_payload(records, symbol="AAPL")
        self.assertEqual(len(frame), 1)
        self.assertEqual(meta["invalid_ohlc_count"], 1)
        self.assertEqual(meta["duplicate_count"], 1)
        self.assertEqual(float(frame["Close"].iloc[0]), 104.0)

    def test_timezone_aware_dates(self):
        records = [
            {"Date": "2026-11-23T14:30:00-05:00", "Open": 100, "High": 105, "Low": 98, "Close": 103},
            {"Date": "2026-11-24T19:30:00Z", "Open": 103, "High": 108, "Low": 101, "Close": 107},
        ]
        frame, _ = sip.normalize_market_payload(records, symbol="AAPL")
        self.assertEqual(len(frame), 2)
        self.assertIsNone(frame["Date"].dt.tz)
        self.assertTrue(frame["Date"].is_monotonic_increasing)

    def test_missing_ohlcv_fields_does_not_fabricate(self):
        records = [{"Date": "2026-11-23", "Open": 100, "Close": 103}]  # missing High, Low
        frame, meta = sip.normalize_market_payload(records, symbol="AAPL")
        self.assertTrue(frame.empty)
        self.assertEqual(set(meta["missing_columns"]), {"High", "Low"})

    def test_multi_stock_data_separation(self):
        aapl = _sample_df(15, symbol="AAPL")
        msft = _sample_df(15, symbol="MSFT")
        combined = pd.concat([aapl, msft], ignore_index=True)
        frame, meta = sip.normalize_market_payload(combined)
        self.assertEqual(len(frame), 30)
        self.assertEqual(set(meta["symbols"]), {"AAPL", "MSFT"})
        self.assertEqual(len(frame[frame["Symbol"] == "AAPL"]), 15)
        self.assertEqual(len(frame[frame["Symbol"] == "MSFT"]), 15)

    def test_render_market_analytics_with_build_market_payload_and_invalid(self):
        import app  # noqa: E402

        raw = _sample_df(220, symbol="AAPL")
        payload = sip.build_market_payload(raw, "AAPL")
        prediction = {
            "ok": True,
            "symbol": "AAPL",
            "prediction": "UP",
            "probabilities": {"UP": 65.0, "DOWN": 35.0},
            "raw_output": 0.65,
        }

        fake_st = MagicMock()
        fake_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        with patch.object(app, "st", fake_st):
            app.render_market_analytics(payload, prediction)
            # Expect 8 Plotly charts: candle, line+SMAs, volume, RSI, MACD, BB, returns, probability
            self.assertEqual(fake_st.plotly_chart.call_count, 8)

            # Empty/invalid payload warns gracefully without crashing
            fake_st.reset_mock()
            app.render_market_analytics({})
            fake_st.warning.assert_called_once()
            fake_st.plotly_chart.assert_not_called()


if __name__ == "__main__":
    unittest.main()

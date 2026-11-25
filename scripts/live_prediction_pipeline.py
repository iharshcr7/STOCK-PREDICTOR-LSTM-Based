"""
live_prediction_pipeline.py
===========================
Phase 5 — automated live market-data acquisition and prediction orchestration
for the EXISTING NASDAQ-trained LSTM.

This module is inference-only:

    * it never trains, retrains, fine-tunes, recompiles or replaces a model;
    * it delegates all model work to ``lstm_inference.predict_from_frame``;
    * it reuses ``fetch_live_data`` providers and ``mongo_store`` persistence.

Responsibilities added in Phase 5:

    * ticker validation and provider availability checks;
    * throttled, cached data retrieval with provider fallback;
    * market-hours awareness and completed-session handling;
    * data freshness classification (fresh / delayed / stale / intraday);
    * single- and multi-symbol orchestration with controlled concurrency;
    * per-symbol monitoring, error classification and recovery;
    * deduplicated MongoDB persistence with an in-memory retry queue;
    * a bounded scheduler suitable for a separate process (never auto-started
      by Streamlit).

CLI:

    python scripts/live_prediction_pipeline.py --symbols AAPL,MSFT
    python scripts/live_prediction_pipeline.py --symbols AAPL --schedule \
        --interval 900 --max-iterations 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

import lstm_inference as inference
from lstm_data_pipeline import FEATURE_COLUMNS, SEQUENCE_LENGTH
from mongo_store import save_lstm_prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_TZ_NAME = "America/New_York"
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z0-9]{1,4})?$")

STATUS_SUCCESS = "success"
STATUS_STALE = "stale_data"
STATUS_INCOMPATIBLE = "incompatible_input"
STATUS_API_FAILURE = "api_failure"
STATUS_INFERENCE_FAILURE = "inference_failure"
STATUS_INVALID_TICKER = "invalid_ticker"
STATUS_MARKET_CLOSED = "market_closed_skip"

FRESH_FRESH = "fresh"
FRESH_DELAYED = "delayed"
FRESH_STALE = "stale"
FRESH_INTRADAY = "intraday_incomplete"
FRESH_UNKNOWN = "unknown"

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "calls per minute",
    "calls per day",
    "daily quota",
)
_PREMIUM_MARKERS = ("premium", "premium feature", "paid tier", "subscribe")
_NO_DATA_MARKERS = ("no data found", "may be delisted", "empty response", "not found")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_STOCK_DIR = PROJECT_ROOT / "data" / "stock_market_dataset" / "stocks"
LOCAL_FEATURED_CSV = PROJECT_ROOT / "data" / "final_featured_data.csv"

# Provider-specific budgets of the Alpha Vantage free tier. Shared across all
# pipeline instances so a Streamlit rerun cannot silently spend the daily quota.
AV_MAX_PER_MINUTE = 5
AV_MAX_PER_DAY = 25

_SHARED_CACHE: dict[tuple[str, str], tuple[float, pd.DataFrame, str, str]] = {}
_SHARED_LOCK = threading.Lock()
_REQUEST_TIMES: dict[str, deque] = {}
_LAST_REQUEST: dict[str, float] = {}
_AV_DAILY: dict[str, tuple[str, int]] = {}


WARMUP_MIN_ROWS = 260  # >= 200 SMA200 warm-up + 60-session LSTM sequence


def load_local_warmup(symbol: str, asset_type: str = "stocks", limit: int = 2000) -> pd.DataFrame:
    """Load bounded local historical OHLCV rows for indicator warm-up."""
    from stock_intelligence_pipeline import STOCK_DIR, ETF_DIR

    directory = ETF_DIR if asset_type == "etfs" else STOCK_DIR
    path = directory / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    frame = pd.read_csv(path, usecols=lambda c: c in columns or c == "Adj Close")
    frame = frame.tail(limit).copy()
    frame["Symbol"] = symbol
    return frame.reset_index(drop=True)


def _session_date(value) -> pd.Timestamp:
    """
    Convert one date-like value to a tz-naive *trading session date*.

    Naive historical CSV dates already represent the exchange's session date
    and are kept unchanged. Timezone-aware API timestamps are first parsed as
    UTC (``pd.Timestamp`` accepts mixed formats), then converted to
    America/New_York before taking the calendar date. This avoids shifting a
    naive daily bar backwards by one day when UTC-normalising it.
    """
    if value is None:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(_market_timezone()).tz_localize(None)
    return timestamp.normalize()


def merge_warmup_history(
    live_df: pd.DataFrame,
    symbol: str,
    *,
    asset_type: str = "stocks",
    min_rows: int = WARMUP_MIN_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge incremental provider data with local history for warm-up.

    Local rows supply long-window indicators (SMA200); API rows supply the
    latest sessions. Dates are normalised, duplicates are resolved keeping the
    newest record, and the result is sorted chronologically. No indicator
    value is fabricated.
    """
    report = {
        "local_rows": 0, "provider_rows": 0, "merged_rows": 0,
        "sufficient": False, "symbol": symbol, "asset_type": asset_type,
        "invalid_dates_dropped": 0, "duplicate_sessions_removed": 0,
        "sessions": [],
    }
    if live_df is None or getattr(live_df, "empty", True):
        return pd.DataFrame(), report

    live = live_df.copy()
    live["Symbol"] = symbol
    report["provider_rows"] = len(live)
    local = load_local_warmup(symbol, asset_type)
    report["local_rows"] = len(local)
    if not local.empty:
        if "Symbol" not in local.columns:
            local = local.copy()
            local["Symbol"] = symbol
        else:
            local = local.copy()
            local["Symbol"] = local["Symbol"].fillna(symbol)

    combined = pd.concat([local, live], ignore_index=True, sort=False)
    if "Date" not in combined.columns:
        report["error"] = "Merged history has no Date column"
        return pd.DataFrame(), report

    # Element-wise parsing avoids pandas' "Mixed timezones detected" error
    # while preserving naive CSV session dates and converting aware API
    # timestamps to the exchange calendar date.
    combined["Session_Date"] = combined["Date"].map(_session_date)
    invalid = combined["Session_Date"].isna()
    report["invalid_dates_dropped"] = int(invalid.sum())
    combined = combined.loc[~invalid].copy()

    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        else:
            combined[column] = np.nan
    combined = combined.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Live rows were concatenated last, so keep="last" keeps the freshest
    # provider record for a session that also exists in local history.
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["Symbol", "Session_Date"], keep="last"
    )
    report["duplicate_sessions_removed"] = int(before_dedup - len(combined))

    combined = combined.sort_values(["Symbol", "Session_Date"]).reset_index(drop=True)
    combined["Date"] = combined["Session_Date"]
    combined = combined.drop(columns=["Session_Date"])

    report["merged_rows"] = len(combined)
    report["sufficient"] = len(combined) >= min_rows
    report["sessions"] = [
        str(date_value.date()) for date_value in combined["Date"]
    ]
    report["latest_date"] = report["sessions"][-1] if report["sessions"] else None
    return combined, report


def _market_timezone():
    """Return the US/Eastern zone, falling back to a fixed offset if tzdata is absent."""
    try:
        return ZoneInfo(MARKET_TZ_NAME)
    except ZoneInfoNotFoundError:  # pragma: no cover - depends on host tzdata
        logger.warning("tzdata unavailable; using fixed UTC-5 market offset")
        return timezone(timedelta(hours=-5))


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__} {exc}".lower()


def _is_rate_limit(exc: Exception) -> bool:
    text = _error_text(exc)
    if any(marker in text for marker in _PREMIUM_MARKERS):
        return False
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _is_premium(exc: Exception) -> bool:
    return any(marker in _error_text(exc) for marker in _PREMIUM_MARKERS)


def _is_no_data(exc: Exception) -> bool:
    return any(marker in _error_text(exc) for marker in _NO_DATA_MARKERS)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MarketDataError(RuntimeError):
    """Base class for acquisition failures."""


class InvalidTickerError(ValueError):
    """Ticker failed format validation."""


class RateLimitError(MarketDataError):
    """A provider signalled rate limiting or quota exhaustion."""


class PremiumFeatureError(MarketDataError):
    """A paid-only endpoint was requested (e.g. Alpha Vantage outputsize=full)."""


class NoDataError(MarketDataError):
    """Provider returned no usable data (delisted / unknown ticker)."""


class InsufficientHistoryError(MarketDataError):
    """Not enough valid history for a 60-session sequence."""


# ---------------------------------------------------------------------------
# Ticker validation and dataset compatibility
# ---------------------------------------------------------------------------

def validate_ticker(symbol: str) -> str:
    """Validate/normalise a ticker. Never silently accepts free text."""
    if not isinstance(symbol, str):
        raise InvalidTickerError("Ticker must be a string")
    normalised = symbol.strip().upper()
    if not normalised or len(normalised) > 12 or not TICKER_RE.match(normalised):
        raise InvalidTickerError(
            f"Invalid ticker {symbol!r}. Use 1-12 letters/digits, e.g. AAPL or BRK.B"
        )
    return normalised


def normalize_ticker_candidates(symbol: str) -> list[str]:
    """
    Produce safe ticker candidates for provider lookup.

    Provider-specific decorations such as a leading ``$`` are removed first;
    candidates that still fail validation are ignored rather than sent to the
    API. This keeps delisted/unknown tickers from wasting requests.
    """
    if not isinstance(symbol, str):
        return []
    raw = symbol.strip()
    candidates = [raw]
    if raw.startswith("$"):
        candidates.append(raw[1:].strip())
    if "." in raw:
        candidates.append(raw.replace(".", "-"))
        candidates.append(raw.replace(".", ""))

    cleaned: list[str] = []
    for candidate in candidates:
        try:
            normalised = validate_ticker(candidate)
        except InvalidTickerError:
            continue
        if normalised not in cleaned:
            cleaned.append(normalised)
    return cleaned


@dataclass(frozen=True)
class SymbolCompatibility:
    symbol: str
    in_bundled_universe: bool
    warnings: tuple = ()


_bundled_universe_cache: set[str] | None = None


def get_bundled_universe() -> set[str]:
    """Symbols present in the bundled dataset (file listing only, no data loaded)."""
    global _bundled_universe_cache
    if _bundled_universe_cache is None:
        try:
            from symbol_mapper import get_supported_stocks
            _bundled_universe_cache = {s.upper() for s in get_supported_stocks()}
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Could not discover bundled symbol universe: %s", exc)
            _bundled_universe_cache = set()
    return _bundled_universe_cache


def check_symbol_compatibility(symbol: str) -> SymbolCompatibility:
    """Flag symbols the NASDAQ-trained model was not trained on."""
    symbol = validate_ticker(symbol)
    universe = get_bundled_universe()
    in_universe = (not universe) or (symbol in universe)
    warnings = ()
    if universe and not in_universe:
        warnings = (
            f"{symbol} is not in the bundled NASDAQ training universe; the model "
            "was not trained on this ticker and its prediction may be unreliable.",
        )
    return SymbolCompatibility(symbol, in_universe, warnings)


# ---------------------------------------------------------------------------
# Market hours / data freshness
# ---------------------------------------------------------------------------

def market_session_state(now: datetime | None = None) -> dict:
    """Describe the current US market session using America/New_York wall time."""
    tz = _market_timezone()
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    weekday = now.weekday()  # 0=Mon .. 6=Sun
    clock = now.time()
    if weekday >= 5:
        phase = "weekend"
    elif clock < MARKET_OPEN:
        phase = "pre_open"
    elif clock < MARKET_CLOSE:
        phase = "open"
    else:
        phase = "after_close"

    # The last session whose daily bar can be treated as completed.
    completed = now.date()
    if phase in ("weekend", "pre_open", "open"):
        completed -= timedelta(days=1)
        while completed.weekday() >= 5:
            completed -= timedelta(days=1)

    return {
        "now_market_time": now.isoformat(),
        "phase": phase,
        "is_trading_day": weekday < 5,
        "is_intraday": phase == "open",
        "last_completed_session_date": completed.isoformat(),
    }


@dataclass
class DataFreshness:
    status: str
    latest_bar_date: str | None = None
    completed_session_date: str | None = None
    age_days: int | None = None
    has_incomplete_bar: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _market_dates(series: pd.Series) -> pd.Series:
    """
    Normalise a Date column to tz-naive market-local midnight timestamps.

    A daily bar date is a *calendar date on the exchange*, not an instant:
    converting naive midnight through UTC would shift it to the previous day
    in America/New_York. Tz-aware bars (yfinance) are converted first.
    """
    values = pd.to_datetime(series, errors="coerce", utc=False)
    try:
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_convert(_market_timezone()).dt.tz_localize(None)
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass
    return values


def classify_freshness(
    df: pd.DataFrame | None,
    *,
    now: datetime | None = None,
    max_age_days: int = 4,
) -> DataFreshness:
    """
    Classify the latest daily bar relative to the last *completed* session.

    A bar dated after the last completed session (e.g. today's partial candle
    while the market is open) is flagged as incomplete and must be dropped
    before inference.
    """
    state = market_session_state(now)
    completed_date = date.fromisoformat(state["last_completed_session_date"])
    if df is None or df.empty or "Date" not in df.columns:
        return DataFreshness(FRESH_UNKNOWN, completed_session_date=completed_date.isoformat())

    parsed = _market_dates(df["Date"]).dropna()
    if parsed.empty:
        return DataFreshness(FRESH_UNKNOWN, completed_session_date=completed_date.isoformat())

    latest = parsed.max().date()
    age = (completed_date - latest).days
    incomplete = age < 0

    if incomplete:
        status = FRESH_INTRADAY
        note = (
            f"Latest bar ({latest}) is newer than the last completed session "
            f"({completed_date}); it is an incomplete intraday candle and is excluded."
        )
    elif age <= max_age_days:
        status = FRESH_FRESH
        note = f"Latest completed session is {age} calendar day(s) old."
    elif age <= max_age_days * 2:
        status = FRESH_DELAYED
        note = f"Market data is {age} calendar days old (delayed / holiday period)."
    else:
        status = FRESH_STALE
        note = f"Market data is {age} calendar days old; treat this prediction as stale."

    return DataFreshness(
        status=status,
        latest_bar_date=latest.isoformat(),
        completed_session_date=completed_date.isoformat(),
        age_days=int(age),
        has_incomplete_bar=incomplete,
        note=note,
    )


def drop_incomplete_bars(
    df: pd.DataFrame,
    freshness: DataFreshness,
) -> tuple[pd.DataFrame, int]:
    """Remove intraday/incomplete bars so they are never used as daily candles."""
    if not freshness.has_incomplete_bar or df is None or df.empty or "Date" not in df:
        return df, 0
    cutoff = pd.Timestamp(freshness.completed_session_date)
    dates = _market_dates(df["Date"])
    keep = dates.dt.normalize() <= cutoff
    return df.loc[keep.fillna(False)].reset_index(drop=True), int((~keep.fillna(False)).sum())


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    symbol: str
    df: pd.DataFrame
    provider: str
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    from_cache: bool = False
    attempts: list = field(default_factory=list)
    rate_limited: bool = False
    vintage: str = "delayed_eod"  # delayed_eod | local_historical | unknown


class MarketDataFetcher:
    """
    Throttled, cached acquisition wrapper around the existing providers.

    Provider order (``auto``): yfinance → Alpha Vantage (free compact tier) →
    local historical CSV. The cache and the request budget are module-level so
    a Streamlit rerun or a freshly constructed pipeline cannot spend the quota
    again. ``force_refresh=True`` is the only way to bypass the cache.
    """

    def __init__(
        self,
        source: str = "auto",
        api_key: str = "",
        provider_order: Iterable[str] | None = None,
        min_request_interval: float = 1.0,
        cache_ttl_seconds: float = 300.0,
        history_loader: Callable[[str, str, str], tuple[pd.DataFrame, str]] | None = None,
    ) -> None:
        self.source = (
            source if source in ("auto", "yfinance", "alpha_vantage", "local") else "auto"
        )
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.provider_order = list(provider_order or ["yfinance", "alpha_vantage", "local"])
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self.history_loader = history_loader
        self._cache = _SHARED_CACHE
        self._lock = _SHARED_LOCK

    @staticmethod
    def _budget(provider: str) -> tuple[bool, str]:
        if provider != "alpha_vantage":
            return True, ""
        today = datetime.now(timezone.utc).date().isoformat()
        day, count = _AV_DAILY.get(provider, (today, 0))
        if day != today:
            day, count = today, 0
        if count >= AV_MAX_PER_DAY:
            _AV_DAILY[provider] = (day, count)
            return False, f"Alpha Vantage daily quota ({AV_MAX_PER_DAY}) exhausted"
        now = time.monotonic()
        times = _REQUEST_TIMES.setdefault(provider, deque())
        while times and now - times[0] > 60.0:
            times.popleft()
        if len(times) >= AV_MAX_PER_MINUTE:
            _AV_DAILY[provider] = (day, count)
            return False, f"Alpha Vantage per-minute quota ({AV_MAX_PER_MINUTE}) exhausted"
        return True, ""

    @staticmethod
    def _record_request(provider: str) -> None:
        if provider != "alpha_vantage":
            return
        now = time.monotonic()
        times = _REQUEST_TIMES.setdefault(provider, deque())
        times.append(now)
        today = datetime.now(timezone.utc).date().isoformat()
        day, count = _AV_DAILY.get(provider, (today, 0))
        if day != today:
            day, count = today, 0
        _AV_DAILY[provider] = (day, count + 1)

    def _providers(self) -> list[str]:
        if self.source != "auto":
            return [self.source]
        providers = [
            p for p in self.provider_order
            if p != "alpha_vantage" or bool(self.api_key)
        ]
        return providers or ["local"]

    def _throttle(self, provider: str) -> None:
        with self._lock:
            wait = self.min_request_interval - (
                time.monotonic() - _LAST_REQUEST.get(provider, 0.0)
            )
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            _LAST_REQUEST[provider] = time.monotonic()

    @staticmethod
    def _load_local_history(symbol: str) -> tuple[pd.DataFrame, str]:
        """Read only one symbol's local CSV (or bounded chunks), never the full dataset."""
        path = LOCAL_STOCK_DIR / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(
                path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"]
            )
        elif LOCAL_FEATURED_CSV.exists():
            frames = []
            for chunk in pd.read_csv(
                LOCAL_FEATURED_CSV,
                usecols=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"],
                chunksize=50_000,
                low_memory=False,
            ):
                mask = chunk["Symbol"].astype(str).str.upper() == symbol
                if mask.any():
                    frames.append(
                        chunk.loc[mask, ["Date", "Open", "High", "Low", "Close", "Volume"]]
                    )
            if not frames:
                raise NoDataError(f"{symbol}: not present in the local historical dataset")
            df = pd.concat(frames, ignore_index=True)
        else:
            raise NoDataError(f"{symbol}: local historical dataset is not available")

        df = df.tail(400).reset_index(drop=True)  # bounded memory
        if df.empty:
            raise NoDataError(f"{symbol}: local historical dataset is empty")
        return df, "local"

    def _default_history_loader(self, symbol: str, provider: str) -> tuple[pd.DataFrame, str]:
        if provider == "local":
            return self._load_local_history(symbol)
        from fetch_live_data import fetch_alpha_vantage, fetch_yfinance

        if provider == "alpha_vantage":
            # Free tier only supports outputsize='compact' (~100 sessions),
            # which still covers a 60-session window plus indicator warm-up.
            return fetch_alpha_vantage(symbol, self.api_key, outputsize="compact"), provider
        return fetch_yfinance(symbol, period="1y"), provider

    def fetch(self, symbol: str, *, force_refresh: bool = False) -> MarketData:
        """Retrieve OHLCV history, trying providers until one returns usable data."""
        candidates = normalize_ticker_candidates(symbol)
        if not candidates:
            raise InvalidTickerError(f"No valid ticker candidate could be derived from {symbol!r}")
        canonical = candidates[0]
        key = (canonical, self.source)

        if not force_refresh:
            with self._lock:
                cached = self._cache.get(key)
            if cached and (time.monotonic() - cached[0]) <= self.cache_ttl_seconds:
                logger.info("Provider cache hit for %s (%s)", canonical, cached[2])
                return MarketData(
                    symbol=canonical, df=cached[1].copy(), provider=cached[2],
                    from_cache=True, attempts=["provider cache"], vintage=cached[3],
                )

        attempts: list[str] = []
        rate_limited = False
        loader = self.history_loader or self._default_history_loader

        for candidate in candidates:
            for provider in self._providers():
                if provider == "alpha_vantage":
                    ok, reason = self._budget(provider)
                    if not ok:
                        attempts.append(reason)
                        logger.info("Skipping Alpha Vantage for %s: %s", candidate, reason)
                        continue
                self._throttle(provider)
                if provider == "alpha_vantage":
                    self._record_request(provider)
                try:
                    df, used_provider = loader(candidate, provider)
                except Exception as exc:
                    if _is_premium(exc):
                        error = PremiumFeatureError(f"{provider}: {exc}")
                    elif _is_rate_limit(exc):
                        error = RateLimitError(f"{provider}: {exc}")
                        rate_limited = True
                    elif _is_no_data(exc):
                        error = NoDataError(f"{provider}: {exc}")
                    else:
                        error = MarketDataError(f"{provider}: {exc}")
                    attempts.append(str(error))
                    logger.warning("Provider %s failed for %s: %s", provider, candidate, exc)
                    if self.source != "auto":
                        continue
                    continue

                if df is None or df.empty:
                    attempts.append(f"{provider}: empty response for {candidate}")
                    continue

                vintage = "local_historical" if used_provider == "local" else "delayed_eod"
                with self._lock:
                    self._cache[key] = (time.monotonic(), df.copy(), used_provider, vintage)
                return MarketData(
                    symbol=candidate, df=df, provider=used_provider,
                    from_cache=False, attempts=attempts,
                    rate_limited=rate_limited, vintage=vintage,
                )

        detail = "; ".join(attempts) if attempts else "no providers configured"
        hint = " (symbol may be delisted or invalid)" if all(
            _is_no_data(Exception(a)) for a in attempts
        ) else ""
        raise MarketDataError(
            f"No usable history for {symbol}: {detail}{hint}"
        )

    def clear_cache(self) -> None:
        """Clear cached market data (does not reset provider budgets)."""
        with self._lock:
            self._cache.clear()

    @staticmethod
    def reset_runtime_state() -> None:
        """
        Clear cache, throttle timestamps and provider budgets.

        Intended for tests and controlled process restarts; production code
        should keep the budget so daily/minute API limits are respected.
        """
        with _SHARED_LOCK:
            _SHARED_CACHE.clear()
            _REQUEST_TIMES.clear()
            _LAST_REQUEST.clear()
            _AV_DAILY.clear()


# ---------------------------------------------------------------------------
# Monitoring and results
# ---------------------------------------------------------------------------

@dataclass
class PipelineMonitor:
    total_runs: int = 0
    last_successful_fetch: dict = field(default_factory=dict)
    last_successful_inference: dict = field(default_factory=dict)
    latest_market_data_timestamp: str | None = None
    api_failures: int = 0
    rate_limit_errors: int = 0
    missing_sessions: int = 0
    validation_failures: int = 0
    prediction_failures: int = 0
    mongo_failures: int = 0
    per_symbol_status: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PipelineResult:
    symbol: str
    status: str
    success: bool = False
    prediction: str | None = None
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    raw_output: float = 0.0
    prediction_timestamp: str | None = None
    market_data_timestamp: str | None = None
    provider: str | None = None
    freshness: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    persisted: bool = False
    mongo_id: str = ""
    model_sha256: str = ""
    feature_config: str = ""
    compatibility_status: str = "unknown"
    dropped_incomplete_bars: int = 0
    market_session: dict = field(default_factory=dict)
    data_vintage: str = "unknown"
    chart_data: dict = field(default_factory=dict)

    def to_dict(self, *, include_chart: bool = False) -> dict:
        data = dict(self.__dict__)
        if not include_chart:
            data.pop("chart_data", None)
        return data

    def to_inference_dict(self) -> dict:
        """Adapt this result to the Phase 4 storage-record contract."""
        compat = dict(self.freshness or {})
        return {
            "ok": self.success,
            "symbol": self.symbol,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "prediction_timestamp": self.prediction_timestamp,
            "input_data_timestamp": self.market_data_timestamp,
            "source": self.provider or "unknown",
            "model_sha256": self.model_sha256,
            "feature_config": self.feature_config,
            "raw_output": self.raw_output,
            "predicted_next_return_pct": None,
            "next_return_status": (
                "unavailable: binary LSTM has no regression head; "
                "no three-class checkpoint exists"
            ),
            "sequence_length": SEQUENCE_LENGTH,
            "capabilities": {
                "direction_binary": True,
                "three_class": False,
                "next_session_return": False,
            },
            "warnings": list(self.warnings),
            "compatibility": {
                "out_of_range_fraction": compat.get("out_of_range_fraction", 0.0),
                "blocks_in_window": compat.get("blocks_in_window", 1),
                "max_gap_days": compat.get("max_gap_days", 0.0),
            },
        }


def build_chart_data(df: pd.DataFrame | None, *, limit: int = 120) -> dict:
    """Extract a small per-symbol OHLCV payload for interactive charts."""
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty or not required.issubset(df.columns):
        return {}
    tail = df.loc[:, sorted(required)].tail(limit)
    payload = {"Date": [str(pd.to_datetime(v).date()) for v in tail["Date"]]}
    for column in ("Open", "High", "Low", "Close", "Volume"):
        payload[column] = [
            None if pd.isna(v) else float(v) for v in tail[column]
        ]
    return payload


def coerce_results(value) -> list[PipelineResult]:
    """
    Normalise anything stored in session state into a list of results.

    Streamlit widget keys can overwrite session values with booleans; this
    helper makes ``None``, ``bool``, ``dict``, single results and iterables safe
    without ever masking a genuine pipeline error inside an iteration.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, PipelineResult):
        return [value]
    if isinstance(value, dict):
        allowed = getattr(PipelineResult, "__dataclass_fields__", {})
        kwargs = {k: v for k, v in value.items() if k in allowed}
        if not kwargs:
            return []
        return [PipelineResult(**kwargs)]
    if isinstance(value, (list, tuple, set)):
        results: list[PipelineResult] = []
        for item in value:
            results.extend(coerce_results(item))
        return results
    return []


def summarize_results(value) -> dict:
    """Explicitly describe success, failure and partial-success outcomes."""
    results = coerce_results(value)
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "partial_success": bool(successful and failed),
        "all_failed": bool(results and not successful),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class LivePredictionPipeline:
    """Coordinate data retrieval, validation, inference, storage and monitoring."""

    def __init__(
        self,
        source: str = "auto",
        api_key: str = "",
        freshness_max_days: int = 4,
        data_fetcher: MarketDataFetcher | None = None,
        collection=None,
        min_request_interval: float = 1.0,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self.fetcher = data_fetcher or MarketDataFetcher(
            source=source,
            api_key=api_key,
            min_request_interval=min_request_interval,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self.freshness_max_days = int(freshness_max_days)
        self.collection = collection
        self.monitor = PipelineMonitor()
        self._pending_records: list[dict] = []
        self._monitor_lock = threading.Lock()

    # -- persistence ------------------------------------------------------

    def build_record(self, result: PipelineResult) -> dict:
        record = inference.to_storage_record(result.to_inference_dict())
        record["meta"].update({
            "provider": result.provider,
            "data_freshness": result.freshness.get("status"),
            "freshness_note": result.freshness.get("note", ""),
            "freshness_age_days": result.freshness.get("age_days"),
            "latest_market_data_timestamp": result.market_data_timestamp,
            "compatibility_status": result.compatibility_status,
            "dropped_incomplete_bars": result.dropped_incomplete_bars,
            "market_session": result.market_session.get("phase"),
            "pipeline_status": result.status,
            "rate_limited": result.freshness.get("rate_limited", False),
            "data_vintage": result.data_vintage,
        })
        return record

    def _persist(self, result: PipelineResult) -> None:
        try:
            mongo_id = save_lstm_prediction(self.build_record(result), collection=self.collection)
        except Exception as exc:
            mongo_id = ""
            logger.warning("MongoDB persistence failed: %s", exc)
        if mongo_id:
            result.persisted, result.mongo_id = True, mongo_id
        else:
            with self._monitor_lock:
                self.monitor.mongo_failures += 1
                self._pending_records.append(self.build_record(result))
            logger.warning("Prediction retained in memory for MongoDB retry (%s)", result.symbol)

    def retry_pending(self) -> int:
        """Retry predictions that failed to persist during a MongoDB outage."""
        if not self._pending_records:
            return 0
        remaining: list[dict] = []
        saved = 0
        for record in self._pending_records:
            try:
                if save_lstm_prediction(record, collection=self.collection):
                    saved += 1
                else:
                    remaining.append(record)
            except Exception:
                remaining.append(record)
        self._pending_records = remaining
        if saved:
            logger.info("Retried %d pending prediction record(s)", saved)
        return saved

    # -- single symbol ----------------------------------------------------

    def _empty_result(self, symbol: str, status: str, errors: list | None = None) -> PipelineResult:
        return PipelineResult(
            symbol=symbol, status=status, success=False,
            errors=list(errors or []),
            market_session=market_session_state(),
        )

    def run_single(
        self,
        symbol: str,
        *,
        force_refresh: bool = False,
        persist: bool = True,
        now: datetime | None = None,
    ) -> PipelineResult:
        """Run the complete pipeline for one symbol, returning a structured result."""
        self.retry_pending()
        with self._monitor_lock:
            self.monitor.total_runs += 1

        try:
            symbol = validate_ticker(symbol)
        except InvalidTickerError as exc:
            result = self._empty_result(str(symbol), STATUS_INVALID_TICKER, [str(exc)])
            self._record_status(result)
            return result

        result = self._empty_result(symbol, STATUS_API_FAILURE)
        result.market_session = market_session_state(now)

        try:
            market_data = self.fetcher.fetch(symbol, force_refresh=force_refresh)
        except (RateLimitError, MarketDataError) as exc:
            result.errors = [str(exc)]
            result.status = STATUS_API_FAILURE
            if isinstance(exc, RateLimitError) or _is_rate_limit(exc):
                with self._monitor_lock:
                    self.monitor.rate_limit_errors += 1
            with self._monitor_lock:
                self.monitor.api_failures += 1
            self._record_status(result)
            return result
        except Exception as exc:
            result.errors = [f"Unexpected data error: {exc}"]
            result.status = STATUS_API_FAILURE
            with self._monitor_lock:
                self.monitor.api_failures += 1
            self._record_status(result)
            return result

        provider = market_data.provider
        result.provider = provider
        result.data_vintage = market_data.vintage
        if market_data.vintage == "local_historical":
            result.warnings.append(
                "Local historical CSV data is NOT live market data; it is used "
                "only as an explicit last-resort fallback and may be stale."
            )
        with self._monitor_lock:
            self.monitor.last_successful_fetch[symbol] = datetime.now(timezone.utc).isoformat()
            if market_data.rate_limited:
                self.monitor.rate_limit_errors += 1

        freshness = classify_freshness(
            market_data.df, now=now, max_age_days=self.freshness_max_days
        )
        freshness_dict = freshness.to_dict()
        freshness_dict["rate_limited"] = market_data.rate_limited
        result.freshness = freshness_dict
        result.warnings.extend(market_data.attempts)

        if freshness.has_incomplete_bar:
            clean_df, dropped = drop_incomplete_bars(market_data.df, freshness)
            result.dropped_incomplete_bars = dropped
            result.warnings.append(freshness.note)
            if clean_df.empty:
                result.status = STATUS_INCOMPATIBLE
                result.errors.append("Only incomplete intraday bars were available")
                with self._monitor_lock:
                    self.monitor.validation_failures += 1
                self._record_status(result)
                return result
            market_data.df = clean_df
            freshness = classify_freshness(
                clean_df, now=now, max_age_days=self.freshness_max_days
            )
            freshness_dict = freshness.to_dict()
            freshness_dict["rate_limited"] = market_data.rate_limited
            result.freshness = freshness_dict
        if freshness.status in (FRESH_DELAYED, FRESH_STALE):
            result.warnings.append(freshness.note)

        compat = check_symbol_compatibility(symbol)
        result.warnings.extend(compat.warnings)
        result.compatibility_status = (
            "out_of_universe" if not compat.in_bundled_universe else "compatible"
        )
        result.chart_data = build_chart_data(market_data.df)

        if market_data.df is None or market_data.df.empty:
            result.status = STATUS_INCOMPATIBLE
            result.errors.append("No OHLCV rows returned by provider")
            with self._monitor_lock:
                self.monitor.validation_failures += 1
            self._record_status(result)
            return result

        result.market_data_timestamp = freshness.latest_bar_date
        if freshness.latest_bar_date:
            with self._monitor_lock:
                self.monitor.latest_market_data_timestamp = freshness.latest_bar_date

        inference_result = inference.predict_from_frame(
            symbol, market_data.df, source=provider
        )
        if not inference_result.get("ok"):
            result.status = STATUS_INCOMPATIBLE
            result.errors.extend(inference_result.get("errors", []))
            result.warnings.extend(inference_result.get("warnings", []))
            with self._monitor_lock:
                self.monitor.validation_failures += 1
                self.monitor.prediction_failures += 1
            self._record_status(result)
            return result

        # Guard the exact tensor contract before accepting the prediction.
        if inference_result.get("sequence_length") != SEQUENCE_LENGTH or \
                inference_result.get("n_features") != len(FEATURE_COLUMNS):
            result.status = STATUS_INFERENCE_FAILURE
            result.errors.append(
                "Unexpected tensor contract: "
                f"{inference_result.get('sequence_length')}x{inference_result.get('n_features')}"
            )
            with self._monitor_lock:
                self.monitor.prediction_failures += 1
            self._record_status(result)
            return result

        result.success = True
        result.prediction = inference_result["prediction"]
        result.confidence = inference_result["confidence"]
        result.probabilities = dict(inference_result["probabilities"])
        result.raw_output = float(inference_result.get("raw_output", 0.0))
        result.prediction_timestamp = inference_result["prediction_timestamp"]
        result.model_sha256 = inference_result["model_sha256"]
        result.feature_config = inference_result["feature_config"]
        result.warnings.extend(inference_result.get("warnings", []))
        result.compatibility_status = "compatible"
        result.status = STATUS_STALE if freshness.status == FRESH_STALE else STATUS_SUCCESS

        with self._monitor_lock:
            self.monitor.last_successful_inference[symbol] = datetime.now(timezone.utc).isoformat()

        if persist:
            self._persist(result)
        self._record_status(result)
        return result

    # -- multiple symbols -------------------------------------------------

    def run_many(
        self,
        symbols: Iterable[str],
        *,
        force_refresh: bool = False,
        persist: bool = True,
        max_workers: int = 2,
        now: datetime | None = None,
    ) -> list[PipelineResult]:
        """
        Multi-symbol run. Retrieval uses bounded concurrency (network I/O only);
        inference stays sequential so the cached TensorFlow model is never
        shared across threads.
        """
        self.retry_pending()
        valid: list[str] = []
        results: dict[str, PipelineResult] = {}

        for raw in symbols:
            try:
                valid.append(validate_ticker(raw))
            except InvalidTickerError as exc:
                bad = self._empty_result(str(raw), STATUS_INVALID_TICKER, [str(exc)])
                results[str(raw).strip().upper()] = bad
                self._record_status(bad)

        fetched: dict[str, MarketData | Exception] = {}
        if valid:
            workers = max(1, min(int(max_workers), len(valid)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self.fetcher.fetch, sym, force_refresh=force_refresh): sym
                    for sym in valid
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        fetched[sym] = future.result()
                    except Exception as exc:
                        fetched[sym] = exc

        ordered: list[PipelineResult] = []
        for raw in symbols:
            key = raw.strip().upper() if isinstance(raw, str) else str(raw)
            if key in results:
                ordered.append(results[key])
                continue
            item = fetched.get(key)
            ordered.append(
                self._infer_market_data(key, item, persist=persist, now=now)
            )
        return ordered

    def _infer_market_data(
        self,
        symbol: str,
        item: MarketData | Exception | None,
        *,
        persist: bool,
        now: datetime | None,
    ) -> PipelineResult:
        """Shared inference path for single and multi-symbol runs."""
        result = self._empty_result(symbol, STATUS_API_FAILURE)
        result.market_session = market_session_state(now)

        if isinstance(item, Exception):
            result.errors = [str(item)]
            if isinstance(item, RateLimitError) or _is_rate_limit(item):
                with self._monitor_lock:
                    self.monitor.rate_limit_errors += 1
            with self._monitor_lock:
                self.monitor.api_failures += 1
            self._record_status(result)
            return result
        if not isinstance(item, MarketData):
            result.errors = ["No market data was retrieved"]
            with self._monitor_lock:
                self.monitor.api_failures += 1
            self._record_status(result)
            return result

        with self._monitor_lock:
            self.monitor.last_successful_fetch[symbol] = datetime.now(timezone.utc).isoformat()
            if item.rate_limited:
                self.monitor.rate_limit_errors += 1

        freshness = classify_freshness(item.df, now=now, max_age_days=self.freshness_max_days)
        freshness_dict = freshness.to_dict()
        freshness_dict["rate_limited"] = item.rate_limited
        result.freshness = freshness_dict
        result.provider = item.provider
        result.data_vintage = item.vintage
        result.warnings.extend(item.attempts)
        if item.vintage == "local_historical":
            result.warnings.append(
                "Local historical CSV data is NOT live market data; it is used "
                "only as an explicit last-resort fallback and may be stale."
            )

        df = item.df
        if freshness.has_incomplete_bar:
            df, dropped = drop_incomplete_bars(df, freshness)
            result.dropped_incomplete_bars = dropped
            result.warnings.append(freshness.note)
            freshness = classify_freshness(
                df, now=now, max_age_days=self.freshness_max_days
            )
            freshness_dict = freshness.to_dict()
            freshness_dict["rate_limited"] = item.rate_limited
            result.freshness = freshness_dict
        if freshness.status in (FRESH_DELAYED, FRESH_STALE):
            result.warnings.append(freshness.note)

        compat = check_symbol_compatibility(symbol)
        result.warnings.extend(compat.warnings)
        result.compatibility_status = "out_of_universe" if not compat.in_bundled_universe else "compatible"
        result.chart_data = build_chart_data(df)

        if df is None or df.empty:
            result.status = STATUS_INCOMPATIBLE
            result.errors.append("No OHLCV rows available after cleaning")
            with self._monitor_lock:
                self.monitor.validation_failures += 1
            self._record_status(result)
            return result

        result.market_data_timestamp = freshness.latest_bar_date
        if freshness.latest_bar_date:
            with self._monitor_lock:
                self.monitor.latest_market_data_timestamp = freshness.latest_bar_date

        inference_result = inference.predict_from_frame(symbol, df, source=item.provider)
        if not inference_result.get("ok"):
            result.status = STATUS_INCOMPATIBLE
            result.errors.extend(inference_result.get("errors", []))
            result.warnings.extend(inference_result.get("warnings", []))
            with self._monitor_lock:
                self.monitor.validation_failures += 1
                self.monitor.prediction_failures += 1
            self._record_status(result)
            return result

        if inference_result.get("sequence_length") != SEQUENCE_LENGTH or \
                inference_result.get("n_features") != len(FEATURE_COLUMNS):
            result.status = STATUS_INFERENCE_FAILURE
            result.errors.append("Unexpected tensor contract")
            with self._monitor_lock:
                self.monitor.prediction_failures += 1
            self._record_status(result)
            return result

        result.success = True
        result.status = STATUS_STALE if freshness.status == FRESH_STALE else STATUS_SUCCESS
        result.prediction = inference_result["prediction"]
        result.confidence = inference_result["confidence"]
        result.probabilities = dict(inference_result["probabilities"])
        result.raw_output = float(inference_result.get("raw_output", 0.0))
        result.prediction_timestamp = inference_result["prediction_timestamp"]
        result.model_sha256 = inference_result["model_sha256"]
        result.feature_config = inference_result["feature_config"]
        result.warnings.extend(inference_result.get("warnings", []))
        with self._monitor_lock:
            self.monitor.last_successful_inference[symbol] = datetime.now(timezone.utc).isoformat()
        if persist:
            self._persist(result)
        self._record_status(result)
        return result

    def _record_status(self, result: PipelineResult) -> None:
        with self._monitor_lock:
            self.monitor.per_symbol_status[result.symbol] = {
                "status": result.status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction": result.prediction,
                "provider": result.provider,
                "freshness": result.freshness.get("status"),
                "persisted": result.persisted,
            }

    # -- scheduler --------------------------------------------------------

    def run_scheduled(
        self,
        symbols: Iterable[str],
        *,
        interval_seconds: float = 900,
        max_iterations: int | None = None,
        stop_event: threading.Event | None = None,
        market_hours_only: bool = True,
        on_results: Callable[[list[PipelineResult]], None] | None = None,
        now: datetime | None = None,
    ) -> list[list[PipelineResult]]:
        """
        Bounded polling loop for use from a separate process/thread.

        Streamlit never starts this automatically. ``max_iterations`` or a
        ``stop_event`` must eventually end the loop.
        """
        symbols = [validate_ticker(s) for s in symbols]
        history: list[list[PipelineResult]] = []
        iteration = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            state = market_session_state(now)
            if market_hours_only and (not state["is_trading_day"] or state["phase"] == "pre_open"):
                logger.info("Market closed (%s) — skipping scheduled fetch", state["phase"])
                results = [
                    self._empty_result(sym, STATUS_MARKET_CLOSED, ["Market is closed; no fetch performed"])
                    for sym in symbols
                ]
                for r in results:
                    r.market_session = state
                    self._record_status(r)
            else:
                results = self.run_many(symbols, now=now)
            history.append(results)
            if on_results:
                on_results(results)
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            if stop_event is not None:
                if stop_event.wait(max(0.0, interval_seconds)):
                    break
            else:
                time.sleep(max(0.0, interval_seconds))
        return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 live LSTM prediction pipeline")
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated tickers")
    parser.add_argument(
        "--source", default="auto",
        choices=["auto", "yfinance", "alpha_vantage", "local"],
    )
    parser.add_argument("--api-key", default="", help="Alpha Vantage API key (or .env)")
    parser.add_argument("--refresh", action="store_true", help="Bypass cache (manual refresh)")
    parser.add_argument("--no-persist", action="store_true", help="Do not write to MongoDB")
    parser.add_argument("--freshness-max-days", type=int, default=4)
    parser.add_argument("--min-request-interval", type=float, default=1.0)
    parser.add_argument("--cache-ttl", type=float, default=300.0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--schedule", action="store_true", help="Run bounded polling loop")
    parser.add_argument("--interval", type=float, default=900.0)
    parser.add_argument("--max-iterations", type=int, default=1)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    pipeline = LivePredictionPipeline(
        source=args.source,
        api_key=args.api_key,
        freshness_max_days=args.freshness_max_days,
        min_request_interval=args.min_request_interval,
        cache_ttl_seconds=args.cache_ttl,
    )

    if args.schedule:
        runs = pipeline.run_scheduled(
            symbols, interval_seconds=args.interval, max_iterations=args.max_iterations
        )
        payload = {
            "iterations": [
                [r.to_dict() for r in iteration] for iteration in runs
            ],
            "monitor": pipeline.monitor.snapshot(),
        }
    else:
        results = pipeline.run_many(
            symbols,
            force_refresh=args.refresh,
            persist=not args.no_persist,
            max_workers=args.max_workers,
        )
        payload = {
            "results": [r.to_dict() for r in results],
            "monitor": pipeline.monitor.snapshot(),
        }

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

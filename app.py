"""
NASDAQ AI Stock Predictor — Streamlit Dashboard
=================================================
Premium dark-themed dashboard with LSTM deep-learning prediction,
candlestick charts, historical graphs, and confidence scores.

Usage:
    streamlit run app.py
"""

import sys
import os
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs when python-dotenv is unavailable."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    load_env_file(PROJECT_ROOT / ".env")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from mongo_store import (
    fetch_recent_predictions,
    fetch_lstm_predictions,
    save_lstm_prediction,
)
from fetch_live_data import get_historical_for_chart, compute_live_features
from symbol_mapper import (
    search_stocks, get_company_name, resolve_symbol, get_supported_stocks,
)
from predict_lstm import (
    predict_lstm_live,
    is_lstm_available,
    LSTM_METADATA_FILE,
    load_lstm_model,
    load_lstm_scaler,
    FEATURE_COLUMNS as LSTM_FEATURE_COLUMNS,
    SEQUENCE_LENGTH as LSTM_SEQUENCE_LENGTH,
    compute_features as compute_lstm_features,
)
from lstm_inference import (
    infer_symbols as infer_symbols_with_existing_lstm,
    predict_live as predict_live_lstm,
    to_storage_record as lstm_storage_record,
)
from live_prediction_pipeline import (
    LivePredictionPipeline,
    coerce_results,
    summarize_results,
)
from stock_intelligence_pipeline import (
    StockIntelligencePipeline,
    build_market_payload,
    classify_coverage,
    discover_universe,
    get_spark,
    normalize_market_payload,
    normalize_run_result,
)
import hdfs_market_store as hdfs_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NASDAQ AI Predictor",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Dark Theme CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --bg: #0B1220; --surface: #111C2E; --card: #162338;
    --text: #F1F5F9; --muted: #94A3B8; --up: #22C55E; --down: #EF4444;
    --accent: #6366F1; --border: rgba(148,163,184,0.18);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] {
    background: var(--surface); border-right: 1px solid var(--border);
}
.main-header {
    display: flex; justify-content: space-between; align-items: center;
    gap: 1.5rem; padding: 1.25rem 1.5rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 14px; margin-bottom: 1.5rem;
}
.main-header h1 { font-size: 1.9rem; font-weight: 800; color: var(--text); margin: 0; letter-spacing: -0.4px; }
.main-header p { color: var(--muted); font-size: 0.9rem; margin: 0.35rem 0 0 0; }
.header-status { display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: flex-end; }
.pill {
    display: inline-block; padding: 0.4rem 0.75rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px;
    border: 1px solid var(--border); background: rgba(148,163,184,0.08); color: var(--muted);
}
.pill-good { color: var(--up); border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.10); }
.pill-bad { color: var(--down); border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.10); }
.pill-accent { color: #A5B4FC; border-color: rgba(99,102,241,0.4); background: rgba(99,102,241,0.12); }
.metric-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.05rem 1rem; text-align: center; transition: border-color .2s, transform .2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.55); }
.metric-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.9px; color: var(--muted); margin-bottom: 0.45rem;
}
.metric-value { font-size: 1.55rem; font-weight: 800; color: var(--text); }
.pred-up, .pred-down { border-radius: 14px; padding: 1.35rem; text-align: center; border: 1px solid; }
.pred-up { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.45); }
.pred-down { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.45); }
.pred-label { font-size: 1.8rem; font-weight: 800; }
.pred-conf { font-size: 0.95rem; color: var(--muted); margin-top: 0.4rem; }
.divider { height: 1px; background: var(--border); margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Get supported stocks from the symbol database (stock-selection functionality)
SUPPORTED_STOCKS_LIST = get_supported_stocks()

# Build dynamic stock dropdown with company names
STOCKS = {}
for symbol in sorted(SUPPORTED_STOCKS_LIST[:100]):  # Show first 100 for UI performance
    company_name = get_company_name(symbol)
    if company_name:
        STOCKS[f"{company_name} ({symbol})"] = symbol
    else:
        STOCKS[symbol] = symbol

# Add popular stocks at the top if not already present
POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
}

# Merge popular stocks at the beginning
for key, value in POPULAR_STOCKS.items():
    if key not in STOCKS:
        STOCKS = {key: value, **STOCKS}

# ---------------------------------------------------------------------------
# Infrastructure status (cheap, cached — never starts Spark in this check)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120, show_spinner=False)
def infrastructure_status() -> dict:
    status = {"hdfs": False, "mongodb": False, "spark": False, "hadoop_home": ""}
    try:
        status["hadoop_home"] = hdfs_store.resolve_hdfs_bin()
        status["hdfs"] = bool(status["hadoop_home"])
    except Exception:
        pass
    try:
        from pymongo import MongoClient
        client = MongoClient(
            os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
            serverSelectionTimeoutMS=1500,
        )
        client.admin.command("ping")
        status["mongodb"] = True
    except Exception:
        pass
    try:
        import pyspark  # noqa: F401
        status["spark"] = True
    except Exception:
        pass
    return status


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_model_ok = is_lstm_available()
_infra = infrastructure_status()
_monitor_state = st.session_state.get('p5_monitor') or {}
_last_update = _monitor_state.get('latest_market_data_timestamp') or 'no data yet'
st.markdown(f"""
<div class="main-header">
    <div>
        <h1>NASDAQ AI Stock Predictor</h1>
        <p>Stock Market Intelligence Engine · existing LSTM inference · model outputs are not investment advice</p>
    </div>
    <div class="header-status">
        <span class="pill {'pill-good' if _model_ok else 'pill-bad'}">MODEL: {'READY' if _model_ok else 'UNAVAILABLE'}</span>
        <span class="pill pill-accent">MARKET DATA: {'AVAILABLE' if _monitor_state.get('last_successful_fetch') else 'IDLE'}</span>
        <span class="pill {'pill-good' if _infra['hdfs'] else 'pill-bad'}">HDFS: {'READY' if _infra['hdfs'] else 'OFFLINE'}</span>
        <span class="pill {'pill-good' if _infra['spark'] else 'pill-bad'}">SPARK: {'READY' if _infra['spark'] else 'OFFLINE'}</span>
        <span class="pill {'pill-good' if _infra['mongodb'] else 'pill-bad'}">MONGODB: {'READY' if _infra['mongodb'] else 'OFFLINE'}</span>
        <span class="pill">LAST UPDATE: {_last_update}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuration")
    
    # Stock selection with search
    st.markdown("#### Stock Selection")
    
    # Search box for finding stocks
    search_query = st.text_input(
        "Search stocks",
        placeholder="Type symbol or company name (e.g., TSLA, Tesla, Apple)",
        help="Search by stock symbol or company name"
    )
    
    if search_query:
        # Perform search
        search_results = search_stocks(search_query, limit=20)
        
        if search_results:
            st.caption(f"Found {len(search_results)} matches:")
            
            # Build search results dropdown
            search_stocks_dict = {}
            for sym, name in search_results:
                if name:
                    search_stocks_dict[f"{name} ({sym})"] = sym
                else:
                    search_stocks_dict[sym] = sym
            
            selected_stock = st.selectbox(
                "Select from results",
                list(search_stocks_dict.keys()),
                index=0,
                key="search_results"
            )
            symbol = search_stocks_dict[selected_stock]
        else:
            st.warning(f"No stocks found matching '{search_query}'")
            selected_stock = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
            symbol = STOCKS[selected_stock]
    else:
        # Default dropdown with popular stocks
        selected_stock = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
        symbol = STOCKS[selected_stock]
    
    st.caption(f"Total available stocks: {len(SUPPORTED_STOCKS_LIST):,}")

    st.markdown("---")
    mode = st.radio(
        "Mode", ["Live Data", "Distributed Pipeline", "Spark Processing"], index=0
    )

    st.markdown("#### Live Data Source")
    st.selectbox(
        "Preferred provider",
        ["auto", "yfinance", "alpha_vantage", "local"],
        index=0,
        key="p5_source",
        help="auto: Yahoo Finance → Alpha Vantage (free tier) → local historical CSV",
    )

    # -----------------------------------------------------------------------
    # Model availability status
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Models")

    lstm_ready = is_lstm_available()

    if lstm_ready:
        st.success("LSTM — ready")
    else:
        st.warning("LSTM — not trained")
        st.caption("Run: `python scripts/train_lstm_model.py`")

    # Show LSTM metadata if available
    lstm_meta_path = Path(str(LSTM_METADATA_FILE))
    if lstm_ready and lstm_meta_path.exists():
        with st.expander("LSTM Model Info"):
            lstm_meta = json.loads(lstm_meta_path.read_text())
            st.metric("LSTM Accuracy", f"{lstm_meta['metrics']['accuracy']*100:.1f}%")
            st.metric("LSTM ROC AUC", f"{lstm_meta['metrics']['roc_auc']:.3f}")
            st.metric("LSTM F1 Score", f"{lstm_meta['metrics']['f1_score']:.3f}")
            st.caption(f"Seq Length: {lstm_meta.get('sequence_length', 60)} days")
            st.caption(f"Trained: {lstm_meta.get('trained_at', 'N/A')[:10]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_metric(label: str, value: str, col):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def render_prediction(direction: str, confidence: float):
    is_up = "UP" in direction
    cls   = "pred-up" if is_up else "pred-down"
    color = "#00c853" if is_up else "#ff1744"
    arrow = "STOCK WILL GO UP" if is_up else "STOCK WILL GO DOWN"
    st.markdown(f"""
    <div class="{cls}">
        <div class="pred-label" style="color:{color};">{arrow}</div>
        <div class="pred-conf">Confidence: {confidence:.1f}%</div>
    </div>""", unsafe_allow_html=True)


def _chart_frame(chart_data):
    """Convert a per-symbol chart payload into a DataFrame (empty if unusable)."""
    if not isinstance(chart_data, dict) or not chart_data.get("Date"):
        return pd.DataFrame()
    try:
        return pd.DataFrame(chart_data)
    except Exception:
        return pd.DataFrame()


def _style_figure(fig, title, height=320):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111C2E",
        plot_bgcolor="#162338",
        font=dict(color="#F1F5F9", size=12),
        title=dict(text=title, x=0.04, font=dict(size=14, color="#F1F5F9")),
        margin=dict(l=45, r=20, t=55, b=35),
        height=height,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", showgrid=True)
    return fig


def render_symbol_analysis(result):
    """Render the four required charts for ONE successfully processed symbol."""
    frame = _chart_frame(getattr(result, "chart_data", None))
    if frame.empty:
        st.info("No per-symbol OHLCV payload was captured for this result.")
        return

    summary_cols = st.columns(5)
    summary_cols[0].metric("Direction", result.prediction or "-")
    summary_cols[1].metric("UP probability", f"{result.probabilities.get('UP', 0.0):.2f}%")
    summary_cols[2].metric("DOWN probability", f"{result.probabilities.get('DOWN', 0.0):.2f}%")
    summary_cols[3].metric("Confidence", f"{result.confidence:.2f}%")
    summary_cols[4].metric("Freshness", (result.freshness or {}).get("status", "-"))
    st.caption(
        f"Latest session: **{result.market_data_timestamp or '-'}** | "
        f"Provider: **{result.provider or '-'}** ({result.data_vintage}) | "
        f"Model output P(UP): **{result.raw_output:.4f}**"
    )

    safe_key = str(result.symbol).replace(".", "_").replace("-", "_")

    price = go.Figure(go.Scatter(
        x=frame["Date"], y=frame["Close"], mode="lines",
        line=dict(color="#818CF8", width=2), name="Close",
    ))
    st.plotly_chart(
        _style_figure(price, f"{result.symbol} — historical closing price"),
        width="stretch", key=f"p5_price_{safe_key}",
    )

    candle = go.Figure(go.Candlestick(
        x=frame["Date"], open=frame["Open"], high=frame["High"],
        low=frame["Low"], close=frame["Close"], name="OHLC",
    ))
    _style_figure(candle, f"{result.symbol} — candlestick")
    candle.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(candle, width="stretch", key=f"p5_candle_{safe_key}")

    volume = go.Figure(go.Bar(
        x=frame["Date"], y=frame["Volume"], marker_color="#475569", name="Volume",
    ))
    st.plotly_chart(
        _style_figure(volume, f"{result.symbol} — traded volume"),
        width="stretch", key=f"p5_volume_{safe_key}",
    )

    probability = go.Figure(go.Bar(
        x=["UP", "DOWN"],
        y=[result.probabilities.get("UP", 0.0), result.probabilities.get("DOWN", 0.0)],
        marker_color=["#22C55E", "#EF4444"],
        text=[f"{result.probabilities.get('UP', 0.0):.2f}%", f"{result.probabilities.get('DOWN', 0.0):.2f}%"],
        textposition="outside",
    ))
    st.plotly_chart(
        _style_figure(probability, f"{result.symbol} — model probability (binary LSTM)"),
        width="stretch", key=f"p5_prob_{safe_key}",
    )
    st.caption(
        f"Historical context: {len(frame)} sessions shown · latest session "
        f"{frame['Date'].iloc[-1]} · current prediction is a single model output, "
        "not a fabricated historical prediction series."
    )


def _ensure_analytics_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute derived technical indicators from Close if not already present."""
    df = frame.copy()
    close = df["Close"]
    if "MA20" not in df.columns:
        df["MA20"] = close.rolling(20, min_periods=20).mean()
    if "SMA50" not in df.columns:
        df["SMA50"] = close.rolling(50, min_periods=50).mean()
    if "SMA200" not in df.columns:
        df["SMA200"] = close.rolling(200, min_periods=200).mean()
    if "RSI" not in df.columns:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    if "BB_Upper" not in df.columns or "BB_Lower" not in df.columns:
        ma20 = df["MA20"]
        std20 = close.rolling(20, min_periods=20).std()
        df["BB_Upper"] = ma20 + 2.0 * std20
        df["BB_Lower"] = ma20 - 2.0 * std20
    if "Daily_Returns" not in df.columns:
        df["Daily_Returns"] = close.pct_change() * 100.0
    return df


def _derive_latest_metrics(frame: pd.DataFrame, metadata: dict) -> dict:
    """Use preserved latest_metrics or derive them from the normalized OHLCV frame."""
    existing = metadata.get("latest_metrics")
    if isinstance(existing, dict) and existing:
        return existing
    last = frame.iloc[-1]
    prev_close = float(frame["Close"].iloc[-2]) if len(frame) > 1 else None
    latest_close = float(last["Close"]) if pd.notna(last.get("Close")) else None
    return {
        "date": str(last["Date"].date()) if hasattr(last["Date"], "date") else str(last["Date"]),
        "open": float(last["Open"]) if pd.notna(last.get("Open")) else None,
        "high": float(last["High"]) if pd.notna(last.get("High")) else None,
        "low": float(last["Low"]) if pd.notna(last.get("Low")) else None,
        "close": latest_close,
        "adj_close": float(last["Adj Close"]) if "Adj Close" in frame.columns and pd.notna(last.get("Adj Close")) else None,
        "volume": float(last["Volume"]) if "Volume" in frame.columns and pd.notna(last.get("Volume")) else None,
        "previous_close": prev_close,
        "daily_change": (latest_close - prev_close) if (latest_close and prev_close) else None,
        "daily_change_pct": ((latest_close / prev_close - 1.0) * 100.0 if (latest_close and prev_close) else None),
    }


def render_market_analytics(payload: dict | list | pd.DataFrame, prediction: dict | None = None):
    """Render the complete interactive OHLCV + indicator analytics for one symbol."""
    target_symbol = (prediction or {}).get("symbol")
    frame, metadata = normalize_market_payload(payload, symbol=target_symbol)

    if frame.empty:
        reason = metadata.get("validation_error") or "No valid processed OHLCV data is available for this symbol."
        st.warning(f"Market analytics unavailable: {reason}")
        return

    # Keep individual stock datasets separate if a multi-symbol frame is passed
    if "Symbol" in frame.columns and frame["Symbol"].nunique(dropna=True) > 1:
        if target_symbol and (frame["Symbol"] == target_symbol).any():
            frame = frame.loc[frame["Symbol"] == target_symbol].sort_values("Date").reset_index(drop=True)
        else:
            first_symbol = str(frame["Symbol"].dropna().iloc[0])
            frame = frame.loc[frame["Symbol"] == first_symbol].sort_values("Date").reset_index(drop=True)
            metadata["symbol"] = first_symbol

    frame = _ensure_analytics_indicators(frame)
    symbol = metadata.get("symbol") or target_symbol or "SYMBOL"
    metrics = _derive_latest_metrics(frame, metadata)
    safe_key = str(symbol).replace(".", "_").replace("-", "_")
    logger.info(
        "Rendering market analytics: symbol=%s source=%s records=%d columns=%s",
        symbol, metadata.get("source", "hdfs_parquet"), len(frame), list(frame.columns),
    )

    st.markdown(f"#### {symbol} — latest completed session: {metrics.get('date', '-')}")
    cols = st.columns(5)
    labels = [
        ("Open", metrics.get("open")), ("High", metrics.get("high")),
        ("Low", metrics.get("low")), ("Close", metrics.get("close")),
        ("Volume", metrics.get("volume")),
    ]
    for col, (label, value) in zip(cols, labels):
        col.metric(label, "-" if value is None else f"{value:,.2f}")

    cols2 = st.columns(5)
    extra = [
        ("Adj Close", metrics.get("adj_close")),
        ("Previous Close", metrics.get("previous_close")),
        ("Daily Change", metrics.get("daily_change")),
        ("Daily Change %", metrics.get("daily_change_pct")),
        ("Sessions", len(frame)),
    ]
    for col, (label, value) in zip(cols2, extra):
        col.metric(
            label,
            "-" if value is None else (f"{value:,.0f}" if label == "Sessions" else f"{value:,.2f}"),
        )

    if prediction and prediction.get("ok"):
        pcols = st.columns(4)
        pcols[0].metric("Direction", prediction.get("prediction", "-"))
        pcols[1].metric("P(UP)", f"{prediction.get('probabilities', {}).get('UP', 0):.2f}%")
        pcols[2].metric("P(DOWN)", f"{prediction.get('probabilities', {}).get('DOWN', 0):.2f}%")
        pcols[3].metric("Model P(UP)", f"{prediction.get('raw_output', 0):.4f}")
        st.caption(
            f"Model: {prediction.get('model_id', 'lstm_model.keras')} | "
            f"Prediction time: {prediction.get('prediction_timestamp', '-')} | "
            f"Input session: {prediction.get('input_data_timestamp', '-')} | "
            f"Source: {prediction.get('source', 'hdfs_parquet')}"
        )
        st.caption("Model output — not guaranteed investment advice.")

    candle = go.Figure(go.Candlestick(
        x=frame["Date"], open=frame["Open"], high=frame["High"],
        low=frame["Low"], close=frame["Close"], name="OHLC",
    ))
    _style_figure(candle, f"{symbol} — candlestick (OHLC)", 360)
    candle.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(candle, width="stretch", key=f"p52_candle_{safe_key}")

    line = go.Figure()
    line.add_trace(go.Scatter(x=frame["Date"], y=frame["Close"], name="Close",
                              line=dict(color="#F1F5F9", width=2)))
    for col, color in (("MA20", "#6366F1"), ("SMA50", "#F59E0B"), ("SMA200", "#22C55E")):
        if col in frame.columns and frame[col].notna().any():
            line.add_trace(go.Scatter(x=frame["Date"], y=frame[col], name=col,
                                      line=dict(color=color, width=1.4)))
    st.plotly_chart(_style_figure(line, f"{symbol} — close & moving averages", 340),
                    width="stretch", key=f"p52_line_{safe_key}")

    if "Volume" in frame.columns and frame["Volume"].notna().any():
        volume = go.Figure(go.Bar(x=frame["Date"], y=frame["Volume"],
                                  marker_color="#475569", name="Volume"))
        st.plotly_chart(_style_figure(volume, f"{symbol} — trading volume", 280),
                        width="stretch", key=f"p52_volume_{safe_key}")

    if "RSI" in frame.columns and frame["RSI"].notna().any():
        rsi = go.Figure(go.Scatter(x=frame["Date"], y=frame["RSI"],
                                   line=dict(color="#818CF8"), name="RSI(14)"))
        rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444")
        rsi.add_hline(y=30, line_dash="dash", line_color="#22C55E")
        st.plotly_chart(_style_figure(rsi, f"{symbol} — RSI(14)", 260),
                        width="stretch", key=f"p52_rsi_{safe_key}")

    if "MACD" in frame.columns and frame["MACD"].notna().any():
        macd = go.Figure()
        macd.add_trace(go.Scatter(x=frame["Date"], y=frame["MACD"],
                                  name="MACD", line=dict(color="#6366F1")))
        if "MACD_Signal" in frame.columns and frame["MACD_Signal"].notna().any():
            macd.add_trace(go.Scatter(x=frame["Date"], y=frame["MACD_Signal"],
                                      name="Signal", line=dict(color="#F59E0B")))
        st.plotly_chart(_style_figure(macd, f"{symbol} — MACD", 260),
                        width="stretch", key=f"p52_macd_{safe_key}")

    if "BB_Upper" in frame.columns and frame["BB_Upper"].notna().any():
        bb = go.Figure()
        bb.add_trace(go.Scatter(x=frame["Date"], y=frame["Close"], name="Close",
                                line=dict(color="#F1F5F9", width=1.6)))
        bb.add_trace(go.Scatter(x=frame["Date"], y=frame["BB_Upper"], name="Upper",
                                line=dict(color="#6366F1", dash="dash")))
        if "BB_Lower" in frame.columns and frame["BB_Lower"].notna().any():
            bb.add_trace(go.Scatter(x=frame["Date"], y=frame["BB_Lower"], name="Lower",
                                    line=dict(color="#6366F1", dash="dash")))
        st.plotly_chart(_style_figure(bb, f"{symbol} — Bollinger Bands", 300),
                        width="stretch", key=f"p52_bb_{safe_key}")

    if "Daily_Returns" in frame.columns and frame["Daily_Returns"].notna().any():
        hist = go.Figure(go.Histogram(x=frame["Daily_Returns"].dropna(), nbinsx=40,
                                      marker_color="#475569"))
        st.plotly_chart(_style_figure(hist, f"{symbol} — daily returns distribution", 260),
                        width="stretch", key=f"p52_returns_{safe_key}")

    if prediction and prediction.get("ok"):
        prob = go.Figure(go.Bar(
            x=["DOWN", "UP"],
            y=[prediction["probabilities"]["DOWN"], prediction["probabilities"]["UP"]],
            marker_color=["#EF4444", "#22C55E"],
            text=[f"{prediction['probabilities']['DOWN']:.2f}%",
                  f"{prediction['probabilities']['UP']:.2f}%"],
            textposition="outside",
        ))
        st.plotly_chart(_style_figure(prob, f"{symbol} — model probability", 280),
                        width="stretch", key=f"p52_prob_{safe_key}")


def make_candlestick(df, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        height=450, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,40,0.8)",
        xaxis_rangeslider_visible=False,
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def make_close_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], mode="lines", name="Close",
        line=dict(color="#818cf8", width=2),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.1)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        height=350, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40), yaxis_title="Price ($)",
    )
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Volume Bar Chart
# ---------------------------------------------------------------------------

def create_volume_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create live volume bar chart with bullish/bearish coloring.
    
    Args:
        df: DataFrame with Date, Open, Close, Volume columns
        symbol: Stock symbol for title
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Volume Chart - No Data Available",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Determine bar colors based on price movement
    # Green for bullish (Close > Open), Red for bearish (Close <= Open)
    colors = [
        "#00c853" if close > open_price else "#ff1744"
        for open_price, close in zip(df["Open"], df["Close"])
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name="Volume",
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        hovertemplate=(
            "<b>Date:</b> %{x}<br>"
            "<b>Volume:</b> %{y:,.0f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Live Volume Analysis",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Volume", tickformat=","),
        showlegend=False,
        hovermode="x unified"
    )
    
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Correlation Heatmap
# ---------------------------------------------------------------------------

def create_correlation_heatmap(df: pd.DataFrame, features: dict = None, window: int = 50) -> go.Figure:
    """
    Create live correlation heatmap from recent rolling window of data.
    
    Args:
        df: Historical DataFrame with technical indicators
        features: Dictionary of live features (optional, for single-point analysis)
        window: Rolling window size (default: 50 recent records)
        
    Returns:
        Plotly Figure object
    """
    # Define features to analyze
    feature_cols = [
        "Close", "Volume", "RSI", "MACD", "Volatility",
        "MA5", "MA20", "Daily_Returns", "Trend_Strength"
    ]
    
    # If we have a DataFrame with historical data, use rolling window
    if df is not None and not df.empty and len(df) > 1:
        # Take last N records for live rolling correlation
        df_window = df.tail(min(window, len(df))).copy()
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df_window.columns]
        
        if len(available_cols) < 2:
            # Not enough features, return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap - Insufficient Data",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,40,0.8)",
            )
            return fig
        
        # Compute correlation matrix
        corr_matrix = df_window[available_cols].corr()
        
    elif features is not None:
        # Single point - create synthetic correlation from feature dict
        # This is a fallback for when we only have current features
        available_cols = [col for col in feature_cols if col in features]
        
        if len(available_cols) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap - Insufficient Features",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,40,0.8)",
            )
            return fig
        
        # Create identity matrix as placeholder
        corr_matrix = pd.DataFrame(
            np.eye(len(available_cols)),
            index=available_cols,
            columns=available_cols
        )
    else:
        # No data available
        fig = go.Figure()
        fig.update_layout(
            title="Correlation Heatmap - No Data Available",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, "#ff1744"],      # Strong negative - Red
            [0.25, "#ff5252"],     # Negative - Light Red
            [0.5, "#424242"],      # Neutral - Dark Gray
            [0.75, "#69f0ae"],     # Positive - Light Green
            [1.0, "#00c853"]       # Strong positive - Green
        ],
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate=(
            "<b>%{y} vs %{x}</b><br>"
            "Correlation: %{z:.3f}<br>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Correlation", side="right"),
            tickmode="linear",
            tick0=-1,
            dtick=0.5,
            thickness=15,
            len=0.7,
            tickfont=dict(color="#94a3b8")
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Live Correlation Matrix (Last {min(window, len(df) if df is not None else 0)} Records)",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=100, r=50, t=80, b=80),
        xaxis=dict(
            side="bottom",
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            tickfont=dict(size=11)
        )
    )
    
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Daily Returns Histogram
# ---------------------------------------------------------------------------

def create_returns_histogram(df: pd.DataFrame, symbol: str, window: int = 100) -> go.Figure:
    """
    Create live daily returns histogram from recent rolling window.
    
    Args:
        df: DataFrame with Date and Close columns
        symbol: Stock symbol for title
        window: Rolling window size (default: 100 recent records)
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty or len(df) < 2:
        # Return empty figure if insufficient data
        fig = go.Figure()
        fig.update_layout(
            title="Daily Returns Histogram - Insufficient Data",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Take last N records for live rolling analysis
    df_window = df.tail(min(window, len(df))).copy()
    
    # Compute daily returns if not already present
    if "Daily_Returns" not in df_window.columns:
        df_window["Daily_Returns"] = (
            (df_window["Close"] - df_window["Close"].shift(1)) / 
            df_window["Close"].shift(1) * 100
        )
    
    # Remove NaN values
    returns = df_window["Daily_Returns"].dropna()
    
    if len(returns) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Daily Returns Histogram - No Valid Returns",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name="Daily Returns",
        marker=dict(
            color="#818cf8",
            line=dict(color="#a78bfa", width=1)
        ),
        hovertemplate=(
            "<b>Return Range:</b> %{x:.2f}%<br>"
            "<b>Frequency:</b> %{y}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_return,
        line=dict(color="#00c853", width=2, dash="dash"),
        annotation=dict(
            text=f"Mean: {mean_return:.2f}%",
            font=dict(size=11, color="#00c853")
        )
    )
    
    # Add +/- 1 std lines
    fig.add_vline(
        x=mean_return + std_return,
        line=dict(color="#ff9800", width=1, dash="dot"),
        annotation=dict(
            text=f"+1σ: {mean_return + std_return:.2f}%",
            font=dict(size=10, color="#ff9800")
        )
    )
    
    fig.add_vline(
        x=mean_return - std_return,
        line=dict(color="#ff9800", width=1, dash="dot"),
        annotation=dict(
            text=f"-1σ: {mean_return - std_return:.2f}%",
            font=dict(size=10, color="#ff9800")
        )
    )
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Live Daily Returns Distribution (Last {len(returns)} Days)",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=80, b=40),
        xaxis=dict(
            title="Daily Return (%)",
            tickformat=".2f"
        ),
        yaxis=dict(
            title="Frequency"
        ),
        showlegend=False,
        hovermode="x"
    )
    
    # Add statistics annotation
    fig.add_annotation(
        text=(
            f"<b>Statistics:</b><br>"
            f"Mean: {mean_return:.3f}%<br>"
            f"Std Dev: {std_return:.3f}%<br>"
            f"Min: {returns.min():.2f}%<br>"
            f"Max: {returns.max():.2f}%"
        ),
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bgcolor="rgba(30,30,60,0.8)",
        bordercolor="rgba(99,102,241,0.3)",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, color="#e2e8f0")
    )
    
    return fig


# ---------------------------------------------------------------------------
# LSTM True Stock-Specific Helper Functions
# ---------------------------------------------------------------------------

def run_lstm_backtest(hist_df: pd.DataFrame, symbol: str):
    """
    Sequential backtest of the LSTM model over the last 30 trading days of the stock.
    Runs in a single efficient batch inference pass.
    """
    if hist_df is None or hist_df.empty or len(hist_df) < LSTM_SEQUENCE_LENGTH + 5:
        return pd.DataFrame(), 0.0
        
    try:
        # 1. Compute features using predict_lstm feature pipeline
        df_feat = compute_lstm_features(hist_df, symbol=symbol)
        df_feat = df_feat.dropna(subset=LSTM_FEATURE_COLUMNS).copy()
        
        if len(df_feat) < LSTM_SEQUENCE_LENGTH + 1:
            return pd.DataFrame(), 0.0
            
        # Load model and scaler (fast cached load)
        model = load_lstm_model()
        scaler = load_lstm_scaler()
        
        # Determine number of test days
        num_test_days = min(30, len(df_feat) - LSTM_SEQUENCE_LENGTH)
        
        # Build batch of sequences
        X_batch = []
        features_data = df_feat[LSTM_FEATURE_COLUMNS].values.astype(np.float32)
        features_scaled = scaler.transform(features_data)
        
        for offset in range(num_test_days, 0, -1):
            k = len(df_feat) - offset
            seq = features_scaled[k - LSTM_SEQUENCE_LENGTH : k]
            X_batch.append(seq)
            
        X_batch = np.array(X_batch, dtype=np.float32)
        
        # Predict batch (returns Sigmoid probability of UP direction)
        y_probas = model.predict(X_batch, verbose=0).flatten()
        
        # Extract dates, prices, predictions and actual outcomes
        dates = []
        closes = []
        probs = []
        predictions = []
        actuals = []
        correctness = []
        
        for idx, offset in enumerate(range(num_test_days, 0, -1)):
            k = len(df_feat) - offset
            y_proba = float(y_probas[idx])
            pred_signal = 1 if y_proba >= 0.5 else 0
            
            # Day T is k - 1 (the final day of sequence)
            # Day T+1 is k (the day we are predicting movement FOR)
            close_today = df_feat.iloc[k - 1]["Close"]
            close_next = df_feat.iloc[k]["Close"]
            
            actual_signal = 1 if close_next > close_today else 0
            
            dates.append(df_feat.iloc[k]["Date"])
            closes.append(close_next)
            probs.append(y_proba)
            predictions.append("UP" if pred_signal == 1 else "DOWN")
            actuals.append("UP" if actual_signal == 1 else "DOWN")
            correctness.append(pred_signal == actual_signal)
            
        backtest_df = pd.DataFrame({
            "Date": dates,
            "Close": closes,
            "Prob": probs,
            "Prediction": predictions,
            "Actual": actuals,
            "Correct": correctness
        })
        
        accuracy = (backtest_df["Correct"].sum() / len(backtest_df)) * 100 if len(backtest_df) > 0 else 0.0
        return backtest_df, accuracy
        
    except Exception as e:
        logger.error(f"Error in run_lstm_backtest: {e}")
        return pd.DataFrame(), 0.0


def create_lstm_forecast_chart(hist_df: pd.DataFrame, lstm_res: dict) -> go.Figure:
    """
    Plots the last 60 trading days of Close prices (the sequence input)
    and draws tomorrow's forecasted next-day trajectory.
    """
    symbol = lstm_res["symbol"]
    pred = lstm_res["prediction"] # "UP" or "DOWN"
    confidence = lstm_res["confidence"]
    
    # Slice the last 60 days
    df_seq = hist_df.tail(60).copy()
    
    fig = go.Figure()
    
    # 1. Historical Close line
    fig.add_trace(go.Scatter(
        x=df_seq["Date"], y=df_seq["Close"],
        mode="lines", name="Historical Close Price",
        line=dict(color="#6366f1", width=3),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.05)"
    ))
    
    # Calculate projection T+1 point
    latest_date = df_seq["Date"].iloc[-1]
    latest_close = df_seq["Close"].iloc[-1]
    
    if isinstance(latest_date, str):
        try:
            latest_dt = pd.to_datetime(latest_date)
            next_dt = latest_dt + pd.tseries.offsets.BDay(1)
        except Exception:
            next_dt = latest_date + " (T+1)"
    else:
        next_dt = latest_date + pd.tseries.offsets.BDay(1)
        
    # Projected price point (2% directional trend lines for aesthetics)
    proj_change = (latest_close * 0.02) if "UP" in pred else (-latest_close * 0.02)
    proj_price = latest_close + proj_change
    
    proj_color = "#00c853" if "UP" in pred else "#ff1744"
    
    # 2. Projection line (dashed)
    fig.add_trace(go.Scatter(
        x=[latest_date, next_dt],
        y=[latest_close, proj_price],
        mode="lines+markers",
        name=f"Projected {pred.split()[0]}",
        line=dict(color=proj_color, width=3, dash="dash"),
        marker=dict(size=10, symbol="triangle-up" if "UP" in pred else "triangle-down", color=proj_color)
    ))
    
    # 3. Highlight final close point
    fig.add_trace(go.Scatter(
        x=[latest_date], y=[latest_close],
        mode="markers", name="Latest Close",
        marker=dict(size=12, color="#a78bfa", line=dict(color="#ffffff", width=2))
    ))
    
    # Annotation for forecast
    fig.add_annotation(
        x=next_dt, y=proj_price,
        text=f"<b>LSTM FORECAST: {pred}</b><br>Confidence: {confidence:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=proj_color,
        ax=40,
        ay=-30 if "UP" in pred else 30,
        bgcolor="rgba(30,30,60,0.9)",
        bordercolor=proj_color,
        borderwidth=2,
        borderpad=6,
        font=dict(size=12, color="#e2e8f0")
    )
    
    fig.update_layout(
        title=dict(
            text=f"LSTM 60-Day Input Sequence & Forecast Timeline ({symbol})",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=80, t=80, b=40),
        yaxis=dict(
            title="Close Price ($)",
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="$"
        ),
        xaxis=dict(
            title="Trading Date",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        showlegend=False
    )
    
    return fig


def create_lstm_backtest_chart(backtest_df: pd.DataFrame, symbol: str, accuracy: float) -> go.Figure:
    """
    Plots the close price over the last 30 days and overlays color-coded prediction signals
    with solid green/red for correct predictions and empty circles/red crosses for incorrect predictions.
    """
    fig = go.Figure()
    
    if backtest_df.empty:
        fig.update_layout(
            title="LSTM Backtest - Insufficient Data",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
        
    # 1. Plot Close Price line of the backtest period
    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Close"],
        mode="lines", name="Actual Close",
        line=dict(color="#94a3b8", width=2, dash="solid"),
    ))
    
    # 2. Add scatter points for Correct predictions
    correct_df = backtest_df[backtest_df["Correct"] == True]
    if not correct_df.empty:
        fig.add_trace(go.Scatter(
            x=correct_df["Date"], y=correct_df["Close"],
            mode="markers",
            name="Correct Signal",
            marker=dict(
                size=12,
                color="#00c853",
                line=dict(color="#ffffff", width=1.5),
                symbol="circle",
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Price:</b> $%{y:.2f}<br>"
                "<b>LSTM Prediction:</b> %{customdata[0]} (Correct)<br>"
                "<extra></extra>"
            ),
            customdata=np.stack([correct_df["Prediction"]], axis=-1)
        ))
        
    # 3. Add scatter points for Incorrect predictions
    incorrect_df = backtest_df[backtest_df["Correct"] == False]
    if not incorrect_df.empty:
        fig.add_trace(go.Scatter(
            x=incorrect_df["Date"], y=incorrect_df["Close"],
            mode="markers",
            name="Incorrect Signal",
            marker=dict(
                size=12,
                color="#ff1744",
                line=dict(color="#ffffff", width=1.5),
                symbol="x",
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Price:</b> $%{y:.2f}<br>"
                "<b>LSTM Prediction:</b> %{customdata[0]} (Incorrect)<br>"
                "<extra></extra>"
            ),
            customdata=np.stack([incorrect_df["Prediction"]], axis=-1)
        ))
        
    fig.update_layout(
        title=dict(
            text=f"LSTM 30-Day Signal Backtest ({symbol}) — Historical Accuracy: {accuracy:.1f}%",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=80, b=40),
        yaxis=dict(
            title="Close Price ($)",
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="$"
        ),
        xaxis=dict(
            title="Trading Date",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    return fig


def create_lstm_metric_chart(metric_name: str) -> go.Figure:
    """
    Plots the global LSTM model training history (Epoch vs Loss/Accuracy).
    metric_name can be "loss" or "accuracy".
    """
    history_file = PROJECT_ROOT / "models" / "lstm_history.json"
    fig = go.Figure()
    
    if not history_file.exists():
        fig.update_layout(
            title=f"LSTM {metric_name.capitalize()} History - No Data Found",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
        
    try:
        history = json.loads(history_file.read_text())
        train_data = history.get(metric_name, [])
        val_name = f"val_{metric_name}"
        val_data = history.get(val_name, [])
        
        # Build x-axes dynamically based on list length
        train_epochs = list(range(1, len(train_data) + 1))
        val_epochs = list(range(1, len(val_data) + 1))
        
        color_train = "#6366f1" if metric_name == "loss" else "#00c853"
        color_val = "#a78bfa" if metric_name == "loss" else "#ff1744"
        
        if train_data:
            fig.add_trace(go.Scatter(
                x=train_epochs, y=train_data,
                mode="lines", name=f"Train {metric_name.capitalize()}",
                line=dict(color=color_train, width=2.5)
            ))
            
        if val_data:
            fig.add_trace(go.Scatter(
                x=val_epochs, y=val_data,
                mode="lines+markers", name=f"Validation {metric_name.capitalize()}",
                line=dict(color=color_val, width=2, dash="dash"),
                marker=dict(size=4, color=color_val)
            ))
            
        fig.update_layout(
            title=dict(
                text=f"Global LSTM Training {metric_name.capitalize()} vs Epochs",
                font=dict(size=14, color="#e2e8f0")
            ),
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=50, r=30, t=60, b=40),
            yaxis=dict(
                title=metric_name.capitalize(),
                gridcolor="rgba(255,255,255,0.05)"
            ),
            xaxis=dict(
                title="Epoch",
                gridcolor="rgba(255,255,255,0.05)"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)"
            )
        )
    except Exception as e:
        logger.error(f"Error creating metric chart for {metric_name}: {e}")
        
    return fig


# ---------------------------------------------------------------------------
# LIVE MODE - PRODUCTION PIPELINE
# ---------------------------------------------------------------------------

if mode == "Live Data":
    st.markdown(f"### Live Prediction Pipeline — {selected_stock}")

    if symbol not in SUPPORTED_STOCKS_LIST:
        st.warning(f"{symbol} is not in the production pipeline.")
        st.stop()

    st.markdown("---")

    # ── Options row ──────────────────────────────────────────────────────────
    data_source = st.selectbox(
        "Data Source",
        ["auto", "alpha_vantage", "yfinance"],
        index=0,
        help="auto: tries Alpha Vantage first, falls back to Yahoo Finance"
    )

    st.markdown("")

    # ── LSTM prediction button ───────────────────────────────────────────────
    lstm_button = st.button(
        "Predict with LSTM",
        type="primary" if lstm_ready else "secondary",
        width="stretch",
        disabled=not lstm_ready,
        help="Predict using LSTM — uses 60-day sequence of historical data"
        if lstm_ready else "Train LSTM first: python scripts/train_lstm_model.py"
    )

    st.markdown("---")

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    # ── Multi-stock inference (Phase 4) ──────────────────────────────────────
    # Reuses the same cached LSTM; every symbol is validated independently and
    # incompatible symbols are reported instead of producing misleading output.
    with st.expander("Multi-Stock Inference (existing LSTM)", expanded=False):
        st.caption(
            "Run the already-trained LSTM on several symbols. Each symbol is "
            "checked for sufficient history, feature compatibility and scaling "
            "compatibility before a prediction is produced."
        )
        batch_input = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,NVDA",
            key="lstm_batch_symbols",
        )
        if st.button("Run batch inference", key="lstm_batch_button"):
            batch_symbols = [s.strip().upper() for s in batch_input.split(",") if s.strip()]
            if not batch_symbols:
                st.warning("Enter at least one stock symbol.")
            else:
                with st.spinner(f"Running inference for {len(batch_symbols)} symbol(s)..."):
                    try:
                        batch_results = infer_symbols_with_existing_lstm(
                            batch_symbols, source=data_source, api_key=api_key
                        )
                        st.session_state['lstm_batch_results'] = batch_results
                        stored_count = 0
                        for batch_res in batch_results:
                            if batch_res.get("ok"):
                                try:
                                    if save_lstm_prediction(lstm_storage_record(batch_res)):
                                        stored_count += 1
                                except Exception as store_exc:
                                    logger.warning("Batch prediction storage failed: %s", store_exc)
                        st.caption(f"Stored {stored_count} new prediction(s) in MongoDB.")
                    except Exception as batch_exc:
                        st.error(f"Batch inference failed: {batch_exc}")

        batch_results = st.session_state.get('lstm_batch_results', [])
        if batch_results:
            batch_rows = [{
                "Symbol": r.get("symbol"),
                "Status": "OK" if r.get("ok") else "SKIPPED",
                "Direction": r.get("prediction") or "-",
                "UP Probability (%)": r.get("probabilities", {}).get("UP", 0.0),
                "Confidence (%)": r.get("confidence", 0.0),
                "Next Return (%)": (
                    "n/a" if r.get("predicted_next_return_pct") is None
                    else r.get("predicted_next_return_pct")
                ),
                "Notes": "; ".join(r.get("errors", []) or r.get("warnings", [])),
            } for r in batch_results]
            st.dataframe(pd.DataFrame(batch_rows), width="stretch", hide_index=True)

    # ── Phase 5 — automated live market-data pipeline ────────────────────────
    st.markdown("---")
    st.markdown(
        "### Live Market Data Pipeline "
        "<span style='font-size:0.75rem;color:#94a3b8;'>— automated</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Fresh OHLCV data → validation → the existing cached LSTM → MongoDB. "
        "Model output only — not investment advice."
    )
    with st.expander("Configure & run live pipeline", expanded=False):
        p5_provider = st.session_state.get("p5_source", "auto")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Provider (sidebar)", p5_provider)
        p5_ttl = pcol2.number_input(
            "Cache TTL (seconds)", min_value=30, max_value=86400, value=900, step=30,
            key="p5_ttl", help="Reuse recent data instead of repeating API requests",
        )
        p5_freshness = pcol3.number_input(
            "Freshness limit (days)", min_value=1, max_value=14, value=4,
            key="p5_freshness", help="Older data is labelled stale, never presented as fresh",
        )
        p5_ticker = st.text_input("Ticker(s)", value=symbol, key="p5_ticker_input").strip().upper()

        # Widget keys and result keys MUST differ: reusing a widget key for
        # results makes Streamlit write a bool into the result on the next
        # rerun, which previously caused "'bool' object is not iterable".
        run_col1, run_col2, run_col3 = st.columns(3)
        p5_run = run_col1.button("Run live prediction", key="p5_run_btn")
        p5_refresh = run_col2.button("Refresh market data", key="p5_refresh_btn")
        p5_batch = run_col3.button("Run multi-stock pipeline", key="p5_batch_btn")

        if p5_run or p5_refresh or p5_batch:
            pipeline = LivePredictionPipeline(
                source=p5_provider,
                api_key=api_key,
                freshness_max_days=int(p5_freshness),
                cache_ttl_seconds=float(p5_ttl),
                min_request_interval=1.0,
            )
            with st.spinner("Running live pipeline (validation + existing LSTM)..."):
                try:
                    if p5_batch:
                        batch_symbols = [
                            s.strip().upper() for s in p5_ticker.split(",") if s.strip()
                        ]
                        st.session_state['p5_batch_results'] = coerce_results(
                            pipeline.run_many(
                                batch_symbols, force_refresh=p5_refresh, max_workers=2
                            )
                        )
                        st.session_state['p5_last_result'] = None
                    else:
                        st.session_state['p5_last_result'] = pipeline.run_single(
                            p5_ticker, force_refresh=p5_refresh
                        )
                        st.session_state['p5_batch_results'] = []
                    st.session_state['p5_monitor_snapshot'] = pipeline.monitor.snapshot()
                except Exception as p5_exc:
                    logger.exception("Live pipeline failed")
                    st.error(f"Live pipeline failed: {p5_exc}")

    p5_single = st.session_state.get('p5_last_result')
    if p5_single is not None and not isinstance(p5_single, bool):
        freshness = p5_single.freshness or {}
        status = p5_single.status
        if status == "success":
            st.success(f"Pipeline status: {status} — prediction below uses validated data.")
        elif status == "stale_data":
            st.warning("Pipeline status: stale data — prediction is NOT fresh.")
        else:
            st.error(f"Pipeline status: {status} — no reliable prediction was produced.")

        st.markdown(
            "**Model prediction (not investment advice):** "
            f"{p5_single.prediction or 'unavailable'}"
        )
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("UP probability", f"{p5_single.probabilities.get('UP', 0.0):.2f}%")
        pc2.metric("DOWN probability", f"{p5_single.probabilities.get('DOWN', 0.0):.2f}%")
        pc3.metric("Freshness", str(freshness.get("status", "-")))
        st.caption(
            f"Provider: **{p5_single.provider or '-'}** ({p5_single.data_vintage}) | "
            f"Market data timestamp: **{p5_single.market_data_timestamp or '-'}** | "
            f"Prediction timestamp: **{p5_single.prediction_timestamp or '-'}** | "
            f"Data age: **{freshness.get('age_days', '-')} day(s)**"
        )
        st.caption(
            f"Market session: **{p5_single.market_session.get('phase', '-')}** | "
            f"Compatibility: **{p5_single.compatibility_status}** | "
            f"Model: **{p5_single.model_sha256[:12] if p5_single.model_sha256 else '-'}** | "
            f"Persisted: **{'yes' if p5_single.persisted else 'no'}**"
        )
        for p5_warning in p5_single.warnings:
            st.warning(p5_warning)
        for p5_error in p5_single.errors:
            st.error(p5_error)
        if p5_single.success and getattr(p5_single, "chart_data", None):
            with st.expander(f"{p5_single.symbol} — price & probability charts", expanded=False):
                render_symbol_analysis(p5_single)

    p5_batch_results = coerce_results(st.session_state.get('p5_batch_results'))
    if p5_batch_results:
        summary = summarize_results(p5_batch_results)
        st.markdown("**Multi-stock pipeline status**")
        st.caption(
            f"{summary['successful']} succeeded · {summary['failed']} failed · "
            f"{'partial success' if summary['partial_success'] else 'complete result'}"
        )
        p5_rows = [{
            "Symbol": r.symbol,
            "Status": r.status,
            "Direction": r.prediction or "-",
            "UP (%)": r.probabilities.get("UP", 0.0),
            "DOWN (%)": r.probabilities.get("DOWN", 0.0),
            "Confidence (%)": r.confidence,
            "Next Return": "n/a",
            "Provider": r.provider or "-",
            "Freshness": (r.freshness or {}).get("status", "-"),
            "Data time": r.market_data_timestamp or "-",
            "Persisted": "yes" if r.persisted else "no",
            "Notes": "; ".join(r.errors or r.warnings),
        } for r in p5_batch_results]
        st.dataframe(pd.DataFrame(p5_rows), width="stretch", hide_index=True)

        successful = [r for r in p5_batch_results if r.success]
        if successful:
            st.markdown("#### Multi-Stock Analysis")
            tabs = st.tabs(["Overview"] + [r.symbol for r in successful])
            tabs[0].caption(
                "Select a stock tab to see that symbol's own OHLCV charts and "
                "prediction probabilities. Failed symbols have no charts."
            )
            for failed in [r for r in p5_batch_results if not r.success]:
                tabs[0].error(f"{failed.symbol}: {'; '.join(failed.errors) or failed.status}")
            for tab, result in zip(tabs[1:], successful):
                with tab:
                    render_symbol_analysis(result)
        else:
            for failed in p5_batch_results:
                st.error(f"{failed.symbol}: {'; '.join(failed.errors) or failed.status}")

    p5_monitor = st.session_state.get('p5_monitor_snapshot')
    if p5_monitor:
        with st.expander("Pipeline monitoring", expanded=False):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("API failures", p5_monitor.get("api_failures", 0))
            mc2.metric("Rate-limit errors", p5_monitor.get("rate_limit_errors", 0))
            mc3.metric("Validation failures", p5_monitor.get("validation_failures", 0))
            mc4.metric("MongoDB failures", p5_monitor.get("mongo_failures", 0))
            st.caption(
                f"Last successful fetch: {p5_monitor.get('last_successful_fetch')} | "
                f"Last successful inference: {p5_monitor.get('last_successful_inference')} | "
                f"Latest market data: {p5_monitor.get('latest_market_data_timestamp')}"
            )
            per_symbol = p5_monitor.get("per_symbol_status") or {}
            if per_symbol:
                st.dataframe(
                    pd.DataFrame(per_symbol).T, width="stretch", hide_index=True
                )
            st.caption("Run the CLI with --schedule for unattended polling; Streamlit never auto-starts a background loop.")

    # ── LSTM Button clicked ───────────────────────────────────────────────────
    if lstm_button and lstm_ready:
        for k in ['chart_df', 'lstm_result', 'lstm_backtest', 'lstm_accuracy', 'active_model']:
            st.session_state.pop(k, None)

        with st.spinner(f"Running LSTM prediction for {symbol} (60-day sequence)..."):
            try:
                lstm_res = predict_live_lstm(symbol=symbol, source=data_source, api_key=api_key)
                hist_df  = get_historical_for_chart(symbol=symbol, api_key=api_key, source=data_source, period="1y")
                backtest_df = pd.DataFrame()
                backtest_acc = 0.0
                if hist_df is not None and not hist_df.empty:
                    try:
                        hist_df = compute_live_features(hist_df)
                        # Run sequential backtesting
                        backtest_df, backtest_acc = run_lstm_backtest(hist_df, symbol)
                    except Exception as he:
                        logger.warning(f"Could not compute features or backtest: {he}")
                if not lstm_res.get("ok", True):
                    st.error(
                        "LSTM inference was not run: "
                        + "; ".join(lstm_res.get("errors", []))
                    )
                else:
                    st.session_state['lstm_result']  = lstm_res
                    st.session_state['chart_df']     = hist_df
                    st.session_state['lstm_backtest'] = backtest_df
                    st.session_state['lstm_accuracy'] = backtest_acc
                    st.session_state['active_model'] = 'lstm'

                    # Persist the inference result (deduplicated, non-blocking).
                    try:
                        st.session_state['lstm_mongo_id'] = save_lstm_prediction(
                            lstm_storage_record(lstm_res)
                        )
                    except Exception as store_exc:
                        logger.warning("LSTM prediction storage failed: %s", store_exc)
            except Exception as e:
                st.error(f"LSTM prediction failed: {e}")

    # ── Display results ───────────────────────────────────────────────────────
    active_model = st.session_state.get('active_model')

    # ── LSTM Results ──────────────────────────────────────────────────────────
    if active_model == 'lstm' and 'lstm_result' in st.session_state:
        lstm_res = st.session_state['lstm_result']
        hist_df  = st.session_state.get('chart_df')

        # Spark UI Availability Panel
        spark_ui_url = st.session_state.get('spark_ui_url') or lstm_res.get('spark_ui_url')
        if spark_ui_url:
            st.markdown("""
            <div style="background-color: #0d2a1d; border-left: 5px solid #10b981; padding: 12px 16px; border-radius: 4px; color: #34d399; font-weight: 600; font-size: 0.95rem; margin-top: 15px; margin-bottom: 20px;">
            Spark UI Available
            </div>
            """, unsafe_allow_html=True)
            
            col_ui1, col_ui2, col_ui3, col_ui4 = st.columns(4)
            with col_ui1:
                st.markdown(f"**[Overview]({spark_ui_url})**")
                st.caption("Main dashboard")
                st.caption("Spark UI will remain available for 5 minutes after processing completes")
                
                from urllib.parse import urlparse
                try:
                    parsed_url = urlparse(spark_ui_url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                except Exception:
                    base_url = spark_ui_url
                st.caption(f"Base URL: [{base_url}]({spark_ui_url})")
                
            with col_ui2:
                st.markdown(f"**[Jobs]({spark_ui_url}/jobs/)**")
                st.caption("All Spark jobs")
                
            with col_ui3:
                st.markdown(f"**[Stages]({spark_ui_url}/stages/)**")
                st.caption("Stage execution")
                
            with col_ui4:
                st.markdown(f"**[Storage]({spark_ui_url}/storage/)**")
                st.caption("RDD storage")
                
            st.markdown("")
            with st.expander("More Spark UI Pages"):
                col_more1, col_more2 = st.columns(2)
                with col_more1:
                    st.markdown("Monitoring:")
                    st.markdown(f"- [Environment]({spark_ui_url}/environment/) - Spark configuration")
                    st.markdown(f"- [Executors]({spark_ui_url}/executors/) - Executor metrics")
                    st.markdown(f"- [SQL]({spark_ui_url}/SQL/) - SQL query execution")
                with col_more2:
                    st.markdown("Analysis:")
                    st.markdown(f"- [Stages]({spark_ui_url}/stages/) - View completed stages")
                    st.markdown(f"- [Jobs]({spark_ui_url}/jobs/) - Click any job to see DAG")
                    st.markdown(f"- [Storage]({spark_ui_url}/storage/) - Cached data")
            
            st.markdown("---")

        # Quote (from hist_df if available)
        if hist_df is not None and not hist_df.empty:
            latest_row = hist_df.iloc[-1]
            st.markdown("### Latest Quote Data")
            qc1, qc2, qc3, qc4, qc5 = st.columns(5)
            render_metric("Open",   f"${float(latest_row['Open']):.2f}",   qc1)
            render_metric("High",   f"${float(latest_row['High']):.2f}",   qc2)
            render_metric("Low",    f"${float(latest_row['Low']):.2f}",    qc3)
            render_metric("Close",  f"${float(latest_row['Close']):.2f}",  qc4)
            render_metric("Volume", f"{int(latest_row['Volume']):,}",       qc5)
            st.markdown("")

        # Prediction
        st.markdown("### Prediction Result  <span style='font-size:0.75rem;color:#94a3b8;'>— LSTM</span>", unsafe_allow_html=True)
        render_prediction(lstm_res["prediction"], lstm_res["confidence"])
        st.markdown("")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("DOWN Probability", f"{lstm_res['probabilities']['DOWN']:.2f}%")
        lc2.metric("UP Probability",   f"{lstm_res['probabilities']['UP']:.2f}%")
        lc3.metric("Confidence", f"{lstm_res['confidence']:.2f}%",
                   delta="High" if lstm_res['confidence'] > 80 else ("Medium" if lstm_res['confidence'] > 60 else "Low"))
        st.markdown("")
        st.caption(
            f"Latest Trading Day: **{lstm_res['latest_date']}** | "
            f"Model: **LSTM** | "
            f"Sequence: **{lstm_res['sequence_length']} days** | "
            f"Source: **{lstm_res['source']}**"
        )

        # Phase 4 — prediction metadata, next-session return availability and
        # per-symbol compatibility feedback.
        next_return = lstm_res.get("predicted_next_return_pct")
        nr1, nr2 = st.columns(2)
        nr1.metric(
            "Next-session return",
            "n/a" if next_return is None else f"{next_return:+.2f}%",
        )
        nr2.metric(
            "Model output",
            f"P(UP) = {lstm_res.get('raw_output', 0.0):.4f}",
        )
        st.caption(
            f"Prediction time (UTC): **{str(lstm_res.get('prediction_timestamp', '-'))[:19].replace('T', ' ')}** | "
            f"Input data timestamp: **{lstm_res.get('input_data_timestamp') or '-'}** | "
            f"Feature config: **{lstm_res.get('feature_config', '-')}** | "
            f"Model ID: **{lstm_res.get('model_id', 'lstm_model.keras')}**"
        )
        if next_return is None:
            st.caption(f"Note: {lstm_res.get('next_return_status', '')}")
        for compat_warning in lstm_res.get("warnings", []):
            st.warning(compat_warning)
        mongo_id = st.session_state.get('lstm_mongo_id', '')
        st.caption(
            "MongoDB: " + ("prediction saved (deduplicated)" if mongo_id else "unavailable — not persisted")
        )

        st.markdown("---")

        # True LSTM Stock-Specific Visualizations
        if hist_df is not None and not hist_df.empty:
            st.markdown("### LSTM Neural Network Visualizations")
            
            # Forecast chart
            fig_forecast = create_lstm_forecast_chart(hist_df, lstm_res)
            st.plotly_chart(fig_forecast, width="stretch", key="lstm_forecast_chart_plotly")
            
            # Backtest section
            backtest_df = st.session_state.get('lstm_backtest', pd.DataFrame())
            backtest_acc = st.session_state.get('lstm_accuracy', 0.0)
            
            if not backtest_df.empty:
                st.markdown("---")
                # Visual metric cards for backtest performance
                st.markdown("### LSTM Backtesting Report (Recent 30 Days)")
                
                btc1, btc2, btc3 = st.columns(3)
                btc1.metric("Sequential Backtest Accuracy", f"{backtest_acc:.1f}%")
                btc2.metric("Total Signals Tested", f"{len(backtest_df)} days")
                correct_signals = backtest_df["Correct"].sum()
                btc3.metric("Correct Directional Signals", f"{correct_signals} / {len(backtest_df)}")
                
                fig_backtest = create_lstm_backtest_chart(backtest_df, symbol, backtest_acc)
                st.plotly_chart(fig_backtest, width="stretch", key="lstm_backtest_chart_plotly")
                
                with st.expander("Detailed Daily Backtest Logs", expanded=False):
                    st.dataframe(
                        backtest_df[["Date", "Close", "Prediction", "Actual", "Correct"]].style.map(
                            lambda x: "background-color: rgba(0, 200, 83, 0.15); color: #00c853;" if x == "UP" or x is True else "background-color: rgba(255, 23, 68, 0.15); color: #ff1744;",
                            subset=["Prediction", "Actual", "Correct"]
                        ),
                        width="stretch"
                    )
            else:
                st.info("LSTM Backtest: To see historical backtesting metrics, make sure you fetched enough historical data.")
                
        st.markdown("---")

        # Global LSTM training history (Epoch vs Loss/Accuracy)
        st.markdown("### Global LSTM Model Training Performance (Model-Wide)")
        st.caption(
            "These training history curves reflect the global performance of the trained neural network model "
            "across the historical dataset epochs. They represent the shared underlying model structure."
        )
        
        hist_tab1, hist_tab2 = st.tabs(["Loss History", "Accuracy History"])
        with hist_tab1:
            fig_loss = create_lstm_metric_chart("loss")
            st.plotly_chart(fig_loss, width="stretch", key="lstm_global_loss_history_plotly")
        with hist_tab2:
            fig_acc = create_lstm_metric_chart("accuracy")
            st.plotly_chart(fig_acc, width="stretch", key="lstm_global_acc_history_plotly")
            
        st.markdown("---")

        # LSTM Model info expander
        with st.expander("Model Information"):
            mi1, mi2 = st.columns(2)
            with mi1:
                st.markdown("**Model Details:**")
                st.write("- Model File: `lstm_model.keras`")
                st.write("- Model Type: LSTM (Long Short-Term Memory)")
                st.write(f"- Sequence Length: {lstm_res['sequence_length']} days")
                st.write("- Features: 21 technical indicators per day")
                lstm_meta_p = PROJECT_ROOT / "models" / "lstm_metadata.json"
                if lstm_meta_p.exists():
                    lm = json.loads(lstm_meta_p.read_text())
                    st.write(f"- Training Accuracy: {lm['metrics']['accuracy']*100:.2f}%")
                    st.write(f"- ROC AUC: {lm['metrics']['roc_auc']:.4f}")
                    st.write(f"- F1 Score: {lm['metrics']['f1_score']:.4f}")
                    arch = lm.get("architecture", {})
                    if arch.get("layers"):
                        st.markdown("**Architecture:**")
                        for layer in arch["layers"]:
                            st.write(f"  - {layer}")
            with mi2:
                st.markdown("**Prediction Details:**")
                st.write(f"- Symbol: {lstm_res['symbol']}")
                st.write(f"- Prediction: {lstm_res['prediction']}")
                st.write(f"- Confidence: {lstm_res['confidence']:.2f}%")
                st.write(f"- DOWN Probability: {lstm_res['probabilities']['DOWN']:.2f}%")
                st.write(f"- UP Probability: {lstm_res['probabilities']['UP']:.2f}%")
                st.write(f"- Data Source: {lstm_res['source']}")
                st.write(f"- Latest Date: {lstm_res['latest_date']}")

        with st.expander("Recent Prediction History (MongoDB)"):
            try:
                recent = fetch_recent_predictions(limit=15)
                if recent:
                    hdf = pd.DataFrame(recent)
                    dcols = ['symbol','prediction','confidence','timestamp','source','model']
                    if all(c in hdf.columns for c in dcols):
                        hdf = hdf[dcols]
                        hdf.columns = ['Symbol','Prediction','Confidence (%)','Timestamp','Source','Model']
                        hdf['Timestamp'] = pd.to_datetime(hdf['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(hdf, width="stretch", hide_index=True)
                    else:
                        st.dataframe(pd.DataFrame(recent), width="stretch")
                else:
                    st.info("No records found or MongoDB unavailable.")
            except Exception as e:
                st.warning(f"Could not fetch history: {e}")

        with st.expander("Recent LSTM Records (MongoDB)"):
            try:
                all_lstm_rows = fetch_lstm_predictions(limit=50)
                history_symbols = sorted({
                    str(r.get("symbol")) for r in all_lstm_rows if r.get("symbol")
                })
                history_symbol = st.selectbox(
                    "Filter by stock", ["All symbols"] + history_symbols,
                    key="p5_history_symbol",
                )
                lstm_rows = (
                    all_lstm_rows if history_symbol == "All symbols"
                    else [r for r in all_lstm_rows if r.get("symbol") == history_symbol]
                )[:15]
                if lstm_rows:
                    ldf = pd.DataFrame(lstm_rows)
                    columns = {
                        "symbol": "Symbol",
                        "prediction": "Direction",
                        "confidence": "Confidence (%)",
                        "timestamp": "Prediction Time",
                    }
                    available = [c for c in columns if c in ldf.columns]
                    ldf = ldf[available].rename(columns=columns)
                    if "Prediction Time" in ldf.columns:
                        ldf["Prediction Time"] = pd.to_datetime(
                            ldf["Prediction Time"]
                        ).dt.strftime("%Y-%m-%d %H:%M:%S")
                    ldf["Next Return (%)"] = [
                        (r.get("meta") or {}).get("predicted_next_return_pct", "n/a")
                        or "n/a"
                        for r in lstm_rows
                    ]
                    ldf["Feature Config"] = [
                        (r.get("meta") or {}).get("feature_config_version", "-")
                        for r in lstm_rows
                    ]
                    st.dataframe(ldf, width="stretch", hide_index=True)
                else:
                    st.info("No LSTM prediction records found or MongoDB unavailable.")
            except Exception as e:
                st.warning(f"Could not fetch LSTM history: {e}")

    else:
        # No prediction yet — show instructions
        st.info("Click **Predict with LSTM** to get a prediction.")
        st.markdown("")
        st.markdown("#### LSTM (Deep Learning)")
        st.markdown(f"""
- Uses **60-day sequence** of historical data
- Captures temporal patterns & trends
- {"Model trained and ready" if lstm_ready else "Not trained yet — run `python scripts/train_lstm_model.py`"}
- Shows training curves (loss & accuracy)
        """)

# ---------------------------------------------------------------------------
# DISTRIBUTED PIPELINE MODE — HDFS -> Spark -> Parquet -> LSTM -> MongoDB
# ---------------------------------------------------------------------------

elif mode == "Distributed Pipeline":
    st.markdown("### Distributed Stock Intelligence Pipeline")
    st.caption(
        "Enforced flow: market data → HDFS raw → Apache Spark → HDFS Parquet → "
        "existing NASDAQ LSTM → MongoDB. Predictions are model outputs, not "
        "guaranteed investment advice."
    )

    all_universe = discover_universe("all")
    coverage = classify_coverage(all_universe)

    ov1, ov2, ov3, ov4, ov5 = st.columns(5)
    ov1.metric("Total symbols", f"{coverage['discovered']:,}")
    ov2.metric("Stocks", f"{coverage['stocks']:,}")
    ov3.metric("ETFs", f"{coverage['etfs']:,}")
    ov4.metric("With local history", f"{coverage['with_local_data']:,}")
    ov5.metric("HDFS", "READY" if _infra['hdfs'] else "OFFLINE")

    with st.expander("Stock selection & pipeline run", expanded=True):
        asset_type = st.selectbox("Asset category", ["stocks", "etfs"], key="p52_asset")
        universe = discover_universe(asset_type)
        search_term = st.text_input(
            "Search ticker", placeholder="e.g. AAPL or Apple", key="p52_search"
        )
        if search_term:
            matches = [
                s.symbol for s in universe
                if search_term.strip().upper() in s.symbol
                or search_term.lower() in s.security_name.lower()
            ][:20]
            st.caption(f"Matches: {', '.join(matches) if matches else 'none'}")
        p52_symbols = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,NVDA,AMZN,GOOGL",
            key="p52_symbols",
            help="Full universe is supported; batches are bounded and resumable.",
        )
        data_mode = st.selectbox(
            "Data mode", ["local_historical", "live_incremental"], key="p52_mode"
        )
        # Result keys MUST differ from widget keys: Streamlit forbids writing
        # to a key already bound to a widget and stores a bool there otherwise.
        st.session_state.setdefault('p52_run_result', None)
        b1, b2, b3 = st.columns(3)
        run_clicked = b1.button("Run HDFS+Spark pipeline", key="p52_run_button", type="primary")
        coverage_clicked = b2.button("Full-universe coverage", key="p52_coverage_button")
        refresh_clicked = b3.button("Refresh live data to HDFS", key="p52_refresh_button")

        if coverage_clicked:
            report = classify_coverage(discover_universe(asset_type))
            st.json({k: v for k, v in report.items() if k != "symbols"})

        if refresh_clicked:
            first = p52_symbols.split(",")[0].strip().upper() if p52_symbols else ""
            if first:
                with st.spinner(f"Acquiring {first} and writing to HDFS..."):
                    try:
                        status = StockIntelligencePipeline().ingest_live(first)
                        st.json(status)
                    except Exception as exc:
                        logger.exception("Live HDFS refresh failed")
                        st.error(f"Live refresh failed: {exc}")

        if run_clicked:
            selected = [s.strip().upper() for s in p52_symbols.split(",") if s.strip()]
            request_signature = f"{selected}|{data_mode}|{asset_type}"
            if not selected:
                st.warning("Enter at least one symbol.")
            elif st.session_state.get('p52_last_request') == request_signature:
                st.caption("This exact pipeline request has already completed; use Refresh or change inputs to run again.")
            else:
                st.session_state['p52_last_request'] = request_signature
                with st.spinner("Running HDFS → Spark → Parquet → LSTM → MongoDB..."):
                    pipeline = StockIntelligencePipeline()
                    payloads = {}
                    try:
                        run = pipeline.run(selected, mode=data_mode, asset_type=asset_type)
                        result = normalize_run_result(run)
                        if run.predictions:
                            spark = get_spark()
                            for prediction in run.predictions:
                                sym = prediction["symbol"]
                                frame = pipeline.feature_reader(
                                    spark, run.run_id, asset_type, sym, limit=300
                                )
                                payloads[sym] = build_market_payload(frame, sym)
                    except Exception as exc:
                        logger.exception("Distributed pipeline failed")
                        result = normalize_run_result({
                            "run_id": f"failed-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "status": "failed",
                            "mode": data_mode,
                            "asset_type": asset_type,
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "failed_symbols": {s: "pipeline_failed" for s in selected},
                        })
                        st.error(f"Distributed pipeline failed: {exc}")
                    st.session_state['p52_run_result'] = result
                    st.session_state['p52_payloads_result'] = payloads

    run_state = normalize_run_result(st.session_state.get('p52_run_result'))
    if run_state:
        st.markdown("#### Pipeline execution metrics")
        st.caption(
            f"Run ID: **{run_state['run_id']}** | Status: **{run_state['status']}** | "
            f"Mode: **{run_state['mode']}** | Started: {run_state['started_at']} | "
            f"Finished: {run_state['finished_at']}"
        )
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("Symbols ingested", run_state["symbols_ingested"])
        rc2.metric("Symbols processed", run_state["symbols_processed"])
        rc3.metric("Spark input rows", f"{run_state['input_records']:,}")
        rc4.metric("Parquet output rows", f"{run_state['output_records']:,}")
        rc5.metric("Predictions", len(run_state["predictions"]))
        st.caption(
            f"HDFS parquet: **{run_state['hdfs_paths'].get('parquet', '-')}** | "
            f"Manifest: **{run_state['hdfs_paths'].get('manifest', '-')}** | "
            f"Spark: **{run_state['spark_info'].get('spark_version', '-')}** | "
            f"MongoDB writes: **{len(run_state['mongo_ids'])}**"
        )
        if run_state.get("failed_symbols"):
            st.error("Failed symbols: " + "; ".join(
                f"{k} ({v})" for k, v in run_state["failed_symbols"].items()
            ))
        if run_state.get("incompatible_symbols"):
            st.warning("Incompatible symbols: " + "; ".join(
                f"{k} ({v})" for k, v in run_state["incompatible_symbols"].items()
            ))
        with st.expander("Spark processing details", expanded=False):
            st.json(run_state.get("spark_info", {}))

        successful = [p for p in run_state["predictions"] if p.get("ok")]
        if successful:
            st.markdown("#### Multi-Stock Analysis (processed Parquet inputs)")
            payloads = st.session_state.get('p52_payloads_result', {})
            tabs = st.tabs(["Overview"] + [p["symbol"] for p in successful])
            summary_rows = [{
                "Symbol": p["symbol"],
                "Direction": p["prediction"],
                "P(UP) %": p["probabilities"]["UP"],
                "P(DOWN) %": p["probabilities"]["DOWN"],
                "Confidence %": p["confidence"],
                "Model P(UP)": p["raw_output"],
                "Input session": p["input_data_timestamp"],
                "Prediction time": p["prediction_timestamp"],
            } for p in successful]
            tabs[0].dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
            tabs[0].caption(
                "Select a stock tab for its own candlestick, price, volume, RSI, "
                "MACD, Bollinger, returns and probability charts."
            )
            for tab, prediction in zip(tabs[1:], successful):
                with tab:
                    render_market_analytics(
                        payloads.get(prediction["symbol"], {}), prediction
                    )

    with st.expander("Historical predictions (MongoDB)", expanded=False):
        try:
            rows = fetch_lstm_predictions(limit=20)
            if rows:
                history_rows = []
                for row in rows:
                    meta = row.get("meta") or {}
                    history_rows.append({
                        "Symbol": row.get("symbol"),
                        "Direction": row.get("prediction"),
                        "UP %": meta.get("direction_probabilities", {}).get("UP"),
                        "DOWN %": meta.get("direction_probabilities", {}).get("DOWN"),
                        "Prediction time": row.get("timestamp"),
                        "Input session": meta.get("latest_market_session"),
                        "Run ID": meta.get("processing_run_id", "-"),
                        "HDFS Parquet": (
                            str(meta.get("hdfs_parquet_reference", "-"))[:48]
                        ),
                    })
                st.dataframe(
                    pd.DataFrame(history_rows), width="stretch", hide_index=True
                )
            else:
                st.info("No prediction records found or MongoDB unavailable.")
        except Exception as exc:
            st.warning(f"Could not fetch prediction history: {exc}")

# ---------------------------------------------------------------------------
# SPARK PROCESSING MODE (big-data demo; predictions are LSTM-only in Live mode)
# ---------------------------------------------------------------------------

elif mode == "Spark Processing":
    st.markdown(f"### Spark Big Data Processing — {selected_stock}")
    st.caption(
        "Distributed feature-engineering demo over the training dataset. "
        "Stock predictions are generated by the LSTM model in Live Data mode."
    )

    if st.button("Run Spark Processing", width="stretch", type="primary"):
        # --- Spark Processing ---
        st.markdown("---")
        st.markdown("### Spark Big Data Processing")
        
        with st.spinner("Running Spark distributed processing on training data..."):
            try:
                import threading
                from spark_processing import (
                    build_spark, read_input_data, validate_columns,
                    clean_data, generate_features as spark_generate_features,
                    save_output, compute_and_save_summary, visualize_dag,
                    demonstrate_rdd_operations, REQUIRED_COLUMNS,
                )
                
                # Build Spark session with optimized partitions for faster processing
                spark_session = build_spark(app_name="StockManualPrediction", use_hdfs=False)
                
                # CRITICAL: Reduce partitions from default 200 → 4 for fast processing
                spark_session.conf.set("spark.sql.shuffle.partitions", "4")
                spark_session.conf.set("spark.default.parallelism", "4")
                
                spark_ui_url = spark_session.sparkContext.uiWebUrl
                if spark_ui_url:
                    st.session_state['spark_ui_url'] = spark_ui_url
                
                # Read training data
                input_path = str(PROJECT_ROOT / "data" / "final_featured_data.csv")
                df = read_input_data(spark_session, input_path)
                
                # Validate columns
                validate_columns(df, REQUIRED_COLUMNS)
                
                # Clean data
                df = clean_data(df)
                
                # Generate Spark features (window functions, rolling calcs)
                df = spark_generate_features(df)
                
                # Cache DataFrame (visible in Spark UI Storage tab)
                df.cache()
                row_count = df.count()
                
                # Visualize DAG
                visualize_dag(df, str(PROJECT_ROOT / "spark_visualizations"))
                
                # Demonstrate RDD operations
                demonstrate_rdd_operations(spark_session, df, str(PROJECT_ROOT / "spark_visualizations"))
                
                # Save processed output
                save_output(df, str(PROJECT_ROOT / "spark_output"))
                
                # Compute and save summary statistics
                compute_and_save_summary(df, str(PROJECT_ROOT / "spark_summary"))
                
                # Show Spark UI info
                if spark_ui_url:
                    st.success(f"**Spark UI Available — [Open Spark UI]({spark_ui_url})**")
                    
                    col_ui1, col_ui2, col_ui3, col_ui4 = st.columns(4)
                    with col_ui1:
                        st.markdown(f"**[Overview]({spark_ui_url})**")
                        st.caption("Main dashboard")
                    with col_ui2:
                        st.markdown(f"**[Jobs]({spark_ui_url}/jobs/)**")
                        st.caption("All Spark jobs")
                    with col_ui3:
                        st.markdown(f"**[Stages]({spark_ui_url}/stages/)**")
                        st.caption("Stage execution")
                    with col_ui4:
                        st.markdown(f"**[Storage]({spark_ui_url}/storage/)**")
                        st.caption("Cached data")
                    
                    st.caption(f"Spark UI will remain available **until app restarts** at: {spark_ui_url}")
                    st.caption("Now run **Live Predict** — new Spark jobs will appear in the same UI!")
                    
                    with st.expander("More Spark UI Pages"):
                        col_more1, col_more2 = st.columns(2)
                        with col_more1:
                            st.markdown("**Monitoring:**")
                            st.markdown(f"- [Environment]({spark_ui_url}/environment/) - Spark configuration")
                            st.markdown(f"- [Executors]({spark_ui_url}/executors/) - Executor metrics")
                            st.markdown(f"- [SQL]({spark_ui_url}/SQL/) - SQL query execution")
                        with col_more2:
                            st.markdown("**Analysis:**")
                            st.markdown(f"- [Stages]({spark_ui_url}/stages/) - View completed stages")
                            st.markdown(f"- [Jobs]({spark_ui_url}/jobs/) - Click any job to see DAG")
                            st.markdown(f"- [Storage]({spark_ui_url}/storage/) - Cached data")
                        st.info("**Tip:** Click on any Job ID in the Jobs page to see its DAG visualization")
                
                # Show processing results
                st.markdown("")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Rows Processed", f"{row_count:,}")
                with col_s2:
                    st.metric("Spark Output", "Parquet Saved")
                with col_s3:
                    st.metric("DAG & RDD Demo", "Generated")
                
                # Keep Spark UI alive INDEFINITELY (no time limit)
                # This way when user switches to Live Predict, the Spark UI
                # is still active and new jobs will appear alongside Manual jobs
                # Spark will only stop when the Streamlit app is restarted
                
            except Exception as e:
                st.warning(f"Spark processing failed (non-critical): {e}")
                st.caption("Spark processing is optional and non-critical for the LSTM prediction workflow.")



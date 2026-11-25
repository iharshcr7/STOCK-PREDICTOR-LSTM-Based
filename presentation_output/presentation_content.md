# Stock Market Intelligence Engine — Presentation Content

**Deck:** `Stock_Market_Intelligence_Engine.pptx`
**Format:** 16:9 widescreen (13.333 × 7.5 in), 10 slides
**Audience:** University project evaluation and viva
**Duration:** Approximately 7–10 minutes

---

## Design system

| Element | Specification |
|---|---|
| Background | `#0B1220` dark navy; secondary surface `#111C2E` |
| Cards | `#162338` rounded rectangles with subtle `#2A3A54` borders |
| Primary text | `#F1F5F9` |
| Secondary text | `#94A3B8` |
| Positive / negative | `#22C55E` / `#EF4444` |
| Accent | `#6366F1` with `#818CF8` highlights |
| Supporting colours | `#38BDF8` (sky), `#F59E0B` (amber), `#64748B` (slate) |
| Typography | Arial headings, Calibri body, Consolas for table values |
| Motif | Consistent micro-labels, numbered badges, colour-coded phase chips, thin grid lines |

All charts, diagrams, icons and dashboard elements are built from editable PowerPoint shapes.

---

## Slide 1 — Title

**Title:** STOCK MARKET INTELLIGENCE ENGINE
**Subtitle:** An AI-Based Stock Market Prediction and Distributed Analytics System

**Technology chips:** Apache Hadoop · Apache Spark · LSTM Deep Learning · MongoDB · Streamlit

**Student detail placeholders:**
- Student Name: `[Student Name]`
- Enrollment Number: `[Enrollment Number]`
- Department: Computer Science and Engineering
- Project Guide: `[Project Guide]`
- Academic Year: 2026–2027

**Visual:** Dark financial terminal background with a faint candlestick chart and trend line on the right; detail card across the bottom.

---

## Slide 2 — Introduction & Problem Statement

**Overview:** An integrated platform that processes large-scale historical and updated stock data, generates AI-based directional predictions, and delivers interactive financial analytics.

**Problem statement:**
1. Large volumes of historical and continuously changing market data.
2. Processing thousands of stock records efficiently is challenging.
3. Manual analysis requires significant time and effort.
4. Market information is distributed across multiple sources.
5. A unified platform is needed for distributed processing and prediction.

**Proposed solution (Problem → Solution infographic):**
- Distributed storage — Hadoop HDFS
- Parallel processing — Apache Spark
- Deep-learning inference — Existing trained LSTM
- Interactive analytics — Streamlit dashboard

---

## Slide 3 — Project Objectives

Eight numbered icon cards:

1. **Data Collection** — Assemble large-scale historical stock market datasets.
2. **HDFS Storage** — Store raw datasets reliably in Hadoop HDFS.
3. **Spark Processing** — Apply distributed, parallel data processing.
4. **Feature Engineering** — Derive technical indicators and predictive features.
5. **LSTM Inference** — Reuse an existing trained model for direction prediction.
6. **MongoDB Storage** — Persist prediction records with full metadata.
7. **Streamlit Dashboard** — Deliver interactive financial analytics.
8. **Large Stock Universe** — Support analysis across thousands of stocks and ETFs.

**Footer strip:** Eight connected objectives — from raw market data to an interactive analytics dashboard.

---

## Slide 4 — System Architecture

Exact workflow, shown as a three-row snake-flow diagram with colour-coded phase chips:

1. **Market Data Sources** — Historical stock and ETF market data. *(Data source)*
2. **Data Acquisition** — Collect and organise incoming datasets. *(Data source)*
3. **Hadoop HDFS** — Distributed storage of raw data. *(Raw storage)*
4. **Apache Spark** — Distributed preprocessing at scale. *(Processing)*
5. **Feature Engineering & Validation** — Indicators, features and validation. *(Processing)*
6. **HDFS — Processed Parquet** — Columnar processed dataset storage. *(Processed storage)*
7. **Existing LSTM Model** — Binary UP / DOWN inference. *(AI inference)*
8. **MongoDB** — Persist predictions with metadata. *(Prediction storage)*
9. **Streamlit** — Interactive analytics dashboard. *(Visualisation)*

**Legend:** Source · Storage · Processing · AI / Output.

---

## Slide 5 — Dataset & Feature Engineering

**Dataset inventory:**
| Dataset | Count / volume |
|---|---:|
| Historical stock CSV files | 5,884 |
| ETF CSV files | 2,165 |
| Stock data volume | Approximately 2.33 GB |
| ETF data volume | Approximately 417 MB |

**Editable bar chart:** Dataset composition by file count (scale 0–7,000).

**OHLCV fields:** Open · High · Low · Close · Adjusted Close · Volume

**Technical indicators:** SMA · EMA · MACD · RSI · Bollinger Bands · Daily Returns

**Preprocessing workflow:**
1. Data cleaning
2. Duplicate removal
3. Missing-value validation
4. Date normalisation
5. Technical indicator generation
6. Feature engineering
7. Time-series sequence preparation

**Highlight:** 21 engineered input features are used by the existing LSTM model.

---

## Slide 6 — Hadoop & Spark Distributed Processing

**Apache Hadoop — distributed storage**
- HDFS distributed storage
- Historical data management
- Raw data organisation
- Processed dataset storage

**Apache Spark — parallel processing**
- Distributed data reading
- Cleaning and transformation
- Feature engineering at scale
- Batch processing → Parquet

**Parquet — analytical format**
- Columnar storage format
- Efficient analytical reads
- Compressed processed datasets
- Structured data organisation

**Processing flow:** HDFS Raw → Spark Processing → Feature Engineering → HDFS Parquet

**Accuracy note:** Architecture description only — no measured speed improvement is claimed without benchmark evidence.

---

## Slide 7 — Existing LSTM Stock Prediction Model

**Model configuration:**
| Parameter | Value |
|---|---|
| Model | LSTM |
| Sequence length | 60 sessions |
| Input features | 21 |
| Input shape | (None, 60, 21) |
| Output shape | (None, 1) |
| Activation | Sigmoid |
| Classification | Binary |
| Threshold | 0.5 |

**Explanation:** Each prediction consumes 60 historical trading sessions with 21 engineered features per session. The saved checkpoint outputs a single sigmoid probability for upward direction.

**Prediction mapping:** `0 = DOWN`, `1 = UP`

**Inference flow:** Historical features (60 × 21) → LSTM layers (checkpoint) → Dense output (1 unit) → Sigmoid (probability) → UP / DOWN (threshold 0.5)

**Accuracy guardrails:** Binary classification only — no three-class or return-value output. Existing-model inference, not new training. No accuracy or performance metric is claimed.

---

## Slide 8 — MongoDB & Streamlit Dashboard

**MongoDB — prediction store:**
- Prediction storage
- Stock symbols
- UP / DOWN probabilities
- Prediction timestamps
- Model identification
- Data provenance

**Streamlit dashboard capabilities (illustrative mockup):**
- Stock selection
- Single-stock inference
- Multi-stock inference
- OHLCV information
- Candlestick charts
- Historical price charts
- Volume visualisation
- Technical indicators
- Prediction history

**Mockup contents:** Stock selector, single/multi-stock inference chips, four OHLCV metric cards (values shown as `—`), candlestick chart, UP/DOWN prediction cards, volume chart.

**Accuracy note:** Clearly labelled "Illustrative mockup — no live values". No real or invented prices are presented.

---

## Slide 9 — Implementation Status, Testing & Limitations

**Completed — Phase 4:**
- Existing trained LSTM loaded successfully
- Original fitted scaler reused
- CPU inference validated
- Multi-stock inference implemented
- MongoDB persistence integration implemented
- Streamlit integration implemented
- 46 Phase 4 tests passed

**Ongoing work:**
- Complete distributed HDFS–Spark–Parquet integration
- Full-universe processing validation
- End-to-end MongoDB verification
- Live market-data reliability
- Complete market analytics integration

**Limitations:**
- Binary prediction only
- Generalisation to non-NASDAQ stocks requires validation
- Market data depends on provider availability
- Prediction probabilities do not guarantee actual market movement

**Banner:** Important — 46 tests refer to Phase 4 validation only, not the complete distributed project.

---

## Slide 10 — Conclusion & Future Scope

**Conclusion:** The Stock Market Intelligence Engine combines distributed data infrastructure, feature engineering, existing deep-learning inference and interactive financial analytics. It demonstrates how big data technologies can support large-scale market information processing and predictive insight, while keeping trained-model inference separate from model development.

**Future scope:**
1. Complete full-universe distributed processing
2. Improve automated market data ingestion
3. Complete end-to-end HDFS–Spark–MongoDB integration
4. Expand multi-stock analytics
5. Improve prediction monitoring and evaluation
6. Introduce additional financial indicators
7. Develop portfolio and risk analytics

**Closing:** THANK YOU — Questions & Discussion

---

## Content accuracy rules observed

- No prediction accuracy, model performance metric, processing speed-up or live success rate is claimed.
- No completed full-universe processing or verified end-to-end pipeline result is claimed.
- The LSTM is described strictly as existing-model binary inference, never as three-class or return-value prediction, and never as trained on all 5,884 stocks.
- Completed implementation is separated from ongoing development on Slide 9.
- Dashboard figures are placeholders, not live or invented market data.

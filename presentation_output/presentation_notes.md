# Stock Market Intelligence Engine — Speaker Notes

Ten slides, approximately 7–10 minutes total. Suggested timing per slide is shown in brackets. These notes are also embedded in the PowerPoint file as native speaker notes.

---

## Slide 1 — Title [0:30]

Introduce the project as a university technical project submitted in the 2026–2027 academic year. The system combines Hadoop, Spark, an existing LSTM checkpoint, MongoDB and Streamlit. Before submission, replace the bracketed student name, enrollment number and project guide. Emphasise that the deep-learning component is an already-trained model used for inference, not a newly trained model. The candlestick artwork is decorative only.

---

## Slide 2 — Introduction & Problem Statement [1:00]

Motivate the project: stock datasets are large and constantly updated. A single machine and manual spreadsheet work do not scale, and the data lives in several places. Walk through the five problem points briefly. Then use the arrow to frame the solution: one pipeline that stores data in HDFS, processes it with Spark, runs inference with an existing LSTM checkpoint, and exposes results through a Streamlit dashboard. Note that this slide presents the intended design, not a claim that the pipeline is already complete.

---

## Slide 3 — Project Objectives [1:00]

These eight objectives map directly onto the pipeline. Point out the ordering: collection, storage, distributed processing, feature engineering, reuse of the trained LSTM, persistence in MongoDB, visualisation in Streamlit, and coverage of a large stock universe. Objective five is deliberately worded as *reuse* of an existing model. The large-universe objective is a target, not a completed result.

---

## Slide 4 — System Architecture [1:30]

This is the most important slide. Follow the numbers in order: sources, acquisition, HDFS raw storage, Spark processing, feature engineering and validation, processed Parquet back in HDFS, the existing LSTM checkpoint for inference, MongoDB for prediction records, and finally Streamlit. The arrows snake across three rows, so trace stage three down to four, then right to left on the middle row, then down to seven and left to right again. The colours distinguish data sources, storage layers, processing, and AI/output stages. Stress that the AI stage is inference from a saved checkpoint, not training.

---

## Slide 5 — Dataset & Feature Engineering [1:00]

Give the inventory first: 5,884 historical stock CSV files and 2,165 ETF CSV files, about 2.33 GB and 417 MB respectively. Each record carries open, high, low, close, adjusted close and volume. The preprocessing workflow cleans data, removes duplicates, validates missing values, normalises dates, derives indicators, engineers features and finally prepares fixed-length time-series sequences. Indicators include SMA, EMA, MACD, RSI, Bollinger Bands and daily returns. The critical number is 21: the existing LSTM expects exactly 21 input features per time step. Do not say the model was trained on all 5,884 stocks; it is only applied across this universe.

---

## Slide 6 — Hadoop & Spark Distributed Processing [1:00]

Hadoop provides the storage layer: HDFS holds raw datasets and later the processed Parquet output. Spark does the compute: reading data in parallel, cleaning it, engineering features and writing batches. Parquet is chosen because its columnar layout suits analytical reads and compresses well. The flow diagram shows the hand-off from HDFS raw to Spark, to feature engineering, and back to HDFS as Parquet. Be careful in the viva: this slide describes the intended architecture, not measured performance. We do not claim any speed-up without benchmark evidence.

---

## Slide 7 — Existing LSTM Stock Prediction Model [1:00]

Be precise here. The checkpoint is an LSTM that takes a 60-session sequence of 21 features and produces one sigmoid probability. A threshold of 0.5 maps that probability to a binary label: 0 means DOWN, 1 means UP. This is inference using an already-trained model and its original fitted scaler; no new training is performed in this phase. Do not claim three classes, do not claim predicted returns, and do not invent accuracy numbers — no performance metric is being presented.

---

## Slide 8 — MongoDB & Streamlit Dashboard [1:00]

MongoDB stores each inference record: symbol, UP and DOWN probabilities, timestamp, model identification and data provenance. On the right is an illustrative Streamlit dashboard layout, not a live screenshot. It shows a stock selector, single and multi-stock inference modes, OHLCV metrics, a candlestick chart, a volume chart, technical indicators and prediction history. All values are shown as dashes deliberately — no real or invented prices are presented. Describe the capabilities of the dashboard rather than treating this mockup as evidence of live data.

---

## Slide 9 — Implementation Status, Testing & Limitations [1:00]

Separate completed work from ongoing work clearly. The verified Phase 4 result is that the existing checkpoint and its original fitted scaler load, CPU inference runs, multi-stock inference works, MongoDB and Streamlit integration exist, and 46 tests pass. Those 46 tests validate Phase 4 only — they are not evidence that the entire distributed pipeline is complete. Still in progress: full HDFS–Spark–Parquet integration, full-universe validation, end-to-end MongoDB checks, data-provider reliability and analytics integration. Concede the limitations openly: binary output only, non-NASDAQ generalisation unvalidated, provider dependence, and probabilities are not guarantees.

---

## Slide 10 — Conclusion & Future Scope [0:30]

Summarise the contribution: one system that spans distributed storage, parallel processing, feature engineering, reuse of a trained LSTM for binary direction inference, persistence and interactive analytics. The future scope is explicitly future work: finish full-universe distributed processing, automate ingestion, close the end-to-end integration, expand analytics, improve monitoring, add indicators and explore portfolio and risk analytics. Close with the limitations already stated — no accuracy claim is made. Invite questions.

---

## Viva guardrails

If asked about performance or completeness, answer along these lines:

- **Accuracy?** Not measured or reported for this presentation. The model is binary classification from an existing checkpoint.
- **Was the model trained on all 5,884 stocks?** No. The existing checkpoint is reused for inference across the dataset; training is out of scope here.
- **Is the pipeline live end-to-end?** No. HDFS–Spark–Parquet integration, full-universe validation and end-to-end MongoDB verification are ongoing.
- **Are dashboard numbers real?** No. The dashboard is an illustrative mockup with placeholder values.
- **Three classes or predicted returns?** No. Output is one sigmoid probability mapped to UP or DOWN at threshold 0.5.
- **Real-time intraday data?** No. Market data is provider-dependent and may be delayed daily data.

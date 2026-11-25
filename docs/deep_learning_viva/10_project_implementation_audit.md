# 10 — Project Implementation Audit (READ-ONLY)

> **Scope:** This document records what was actually found in the repository and in the saved TensorFlow/Keras artifacts. It was produced without modifying any project code, model weights, scaler, configuration, dataset or existing artifact. Analysis scripts were written only to a temporary directory outside the project.

---

## 10.1 Method

| Step | Method | Evidence |
|---|---|---|
| Artifact hashing | SHA-256 of files read in binary mode | §10.3 |
| Architecture | `zipfile` read of `config.json` / `metadata.json` inside the `.keras` archive, then `keras.models.load_model()` and `model.summary()` | §10.4 |
| Weight/parameter counts | `model.count_params()`, per-layer `layer.weights` shapes | §10.4 |
| Compile state | Top-level `compile_config` key of `config.json` | §10.4 |
| Inference sanity | `model.predict()` on a `(2, 60, 21)` zero array; weights compared before/after | §10.4 |
| Scaler | `joblib.load()` of `models/lstm_scaler.pkl`; attributes `data_min_`, `data_max_`, `scale_`, `min_` read only | §10.5 |
| Source inspection | Line-numbered reads and repository-wide keyword search | §10.6–10.9 |
| Model execution | **No training, no retraining, no fine-tuning, no recompilation, no scaler refit** | — |

Environment used for inspection: Python 3.11, TensorFlow 2.15.0, Keras 2.15.0, scikit-learn 1.7.2 (project `.venv`), joblib. `python -B` was used to prevent `__pycache__` creation inside the repository.

---

## 10.2 Executive summary

The shipped deep-learning component is a **binary LSTM classifier**: it consumes a 60-session × 21-feature window and emits one sigmoid probability `P(UP)`. It was created by an **earlier version** of the training pipeline (preserved in `phase2_backup_20261124/`) that trained on five stable NASDAQ symbols with an 80/20 chronological split. The current scripts are a **Phase 4 refactor** that wraps this existing checkpoint with artifact discovery, validation, multi-stock inference, MongoDB persistence and Streamlit integration; they do **not** retrain it. A Phase 3 multi-task model (3 classes + return regression) is designed in code but **no Phase 3 checkpoint exists**, so it is not part of the running system.

---

## 10.3 Artifacts (verified)

| Artifact | Size (bytes) | SHA-256 | Purpose |
|---|---:|---|---|
| `models/lstm_model.keras` | 1,580,074 | `f4a8ba0f7d44a6d75d79d3f76f283da158de13b8a3bcad8d1ba70c84d23ce4ce` | Trained Keras checkpoint |
| `models/lstm_scaler.pkl` | 837 | `aa5949f47e6791aba64e3cec3691d7c3d7bb1f028418e3ddc605717c54dbe518` | Fitted MinMaxScaler |
| `models/lstm_metadata.json` | 2,629 | `4bfa321dd90b7185e9fb262237010a7e4eea6dc21baae04b6d0614760a105be1` | Feature schema, labels, recorded metrics |
| `models/lstm_history.json` | 2,977 | — | 29-epoch training curves |

`.keras` archive contains exactly: `metadata.json`, `config.json`, `model.weights.h5`. Keras version recorded inside: **2.15.0**; date saved **2026-05-20@13:50:40**.

---

## 10.4 Exact checkpoint architecture (extracted, not assumed)

`config.json` → `class_name: Sequential`, 7 entries in `layers` (1 `InputLayer` + 6 compute layers). `keras.models.load_model()` reports **6 layers** because Keras 2.15 does not count `InputLayer` in `model.layers`.

| # | Layer (exact class) | Configuration from checkpoint | Output shape | Parameters |
|---:|---|---|---|---:|
| — | `InputLayer` `input_1` | `batch_input_shape=[None, 60, 21]`, `dtype=float32` | `(None, 60, 21)` | 0 |
| 1 | `LSTM` `lstm` | `units=128`, `activation=tanh`, `recurrent_activation=sigmoid`, `return_sequences=True`, `return_state=False`, `dropout=0.0`, `recurrent_dropout=0.0`, `go_backwards=False`, `stateful=False`, `unroll=False`, `implementation=2`, `unit_forget_bias=True`, GlorotUniform kernel, Orthogonal recurrent kernel | `(None, 60, 128)` | 76,800 |
| 2 | `Dropout` `dropout` | `rate=0.2` | `(None, 60, 128)` | 0 |
| 3 | `LSTM` `lstm_1` | `units=64`, same defaults, `return_sequences=False` | `(None, 64)` | 49,408 |
| 4 | `Dropout` `dropout_1` | `rate=0.2` | `(None, 64)` | 0 |
| 5 | `Dense` `dense` | `units=32`, `activation=relu` | `(None, 32)` | 2,080 |
| 6 | `Dense` `dense_1` | `units=1`, `activation=sigmoid` | `(None, 1)` | 33 |

**Totals:** 128,321 parameters — **all 128,321 trainable, 0 non-trainable** (501.25 KB). Input dtype `float32`, output dtype `float32`.

Weight shapes confirm the parameter arithmetic:

| Layer | Weight | Shape |
|---|---|---|
| `lstm` | kernel | `(21, 512)` = 21 × (4 × 128) |
| `lstm` | recurrent_kernel | `(128, 512)` |
| `lstm` | bias | `(512,)` |
| `lstm_1` | kernel | `(128, 256)` = 128 × (4 × 64) |
| `lstm_1` | recurrent_kernel | `(64, 256)` |
| `lstm_1` | bias | `(256,)` |
| `dense` | kernel / bias | `(64, 32)` / `(32,)` |
| `dense_1` | kernel / bias | `(32, 1)` / `(1,)` |

**Compile state stored in the checkpoint** (`config.json` → `compile_config`):

| Key | Value |
|---|---|
| optimizer | Adam, `learning_rate = 0.0002500000118743628`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7` |
| loss | `binary_crossentropy` |
| metrics | `accuracy` |

**Inference sanity check:** `predict(np.zeros((2,60,21)))` returned shape `(2, 1)`, dtype `float32`, values `[0.5652, 0.5652]` (meaningless input — shape check only). Comparing every weight array before and after showed **no weight change**, confirming the forward pass does not update the model.

### Verified capability boundaries

- Single sigmoid unit → binary `P(UP)` only.
- No softmax layer → no 3-class output.
- No linear regression head → no predicted next-session return.
- `lstm_inference.py` lines 302–306 reports `capabilities = {direction_binary: True, three_class: False, next_session_return: False}`.
- `tests/test_phase4_lstm_inference.py` lines 184–189 asserts these capability flags.

---

## 10.5 Scaler (verified)

| Property | Value |
|---|---|
| Type | `sklearn.preprocessing.MinMaxScaler` |
| `n_features_in_` | **21** |
| `feature_range` | `(0, 1)` |
| `data_mean_` | absent (confirms MinMax, not Standard) |
| Pickle version | scikit-learn 1.7.2 (per `scripts/scaler_compat.py` lines 6–9) |
| Load pattern | `joblib.load()` in `predict_lstm.py` line 123; never `.fit()` |

Fitted boundaries (read-only, first/last values shown):

| Feature | `data_min_` | `data_max_` |
|---|---:|---:|
| Open | 0.094618 | 2173.070 |
| High | 0.095486 | 2176.790 |
| Low | 0.090278 | 2127.450 |
| Close | 0.093750 | 2153.100 |
| Volume | 347,200 | 1,855,410,176 |
| RSI | 13.0255 | 93.3958 |
| MACD | −93.9105 | 78.2351 |
| Daily_Returns | −51.8692 | 34.4714 |
| BB_Width | 1.6948 | 188.6249 |
| Avg_5D_Volume_Trend | 0.01009 | 4.03328 |

These ranges are consistent with the **current** feature definitions in `lstm_data_pipeline.py` lines 140–188 (percent-form returns/indicators, raw prices, raw volume). `transform_sequences()` (lines 412–420) applies `scaler.transform()` and clips the result to `[0, 1]`.

---

## 10.6 The 21 features — verified names and order

Canonical sources agree exactly:

- `scripts/model_config.py` lines 28–50 (`FINAL_FEATURE_COLUMNS`)
- `scripts/lstm_data_pipeline.py` lines 65–70 (imports it as `FEATURE_COLUMNS`)
- `models/lstm_metadata.json` lines 5–27 (`feature_columns`)
- `scripts/lstm_inference.py` lines 189–195 cross-checks all three at startup

Order (positional index 0–20):

`Open, High, Low, Close, Volume, MA5, MA10, MA20, EMA12, RSI, MACD, MACD_Signal, Daily_Returns, Volatility, Price_Change_Pct, Weekly_Momentum, Trend_Strength, BB_Width, Avg_5D_Volume_Trend, Lag_1, Lag_3`

Diagnostic columns computed but **never fed to the model** (`lstm_data_pipeline.py` lines 190–197): `Volume_Change_Pct`, `BB_Position`, `MACD_Hist`, `Avg_Volume_Trend`, `Rolling_Std_Returns`. Labels/target metadata (`Symbol`, `Date`, `Target`, `Target_Date`, `Next_Return_Pct`, `Sequence_Block`) are excluded by `NON_FEATURE_COLUMNS` (lines 82–84).

---

## 10.7 Training facts: recorded vs. current code

### Recorded in `models/lstm_metadata.json` (from the run that produced the shipped checkpoint)

| Field | Value | Lines |
|---|---|---|
| `sequence_length` | 60 | 36 |
| Architecture | LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid) | 37–45 |
| Optimizer / LR / loss (recorded) | Adam / 0.001 / binary_crossentropy | 46–48 |
| `epochs_configured` / `epochs_trained` | 30 / 29 | 51–52 |
| `batch_size` | 32 | 53 |
| `split_strategy` | `chronological_80_20` | 54 |
| `early_stopping` | true | 55 |
| `stable_symbols` | AAPL, AMZN, GOOGL, MSFT, NVDA | 29–35 |
| `prediction_config` | threshold 0.5; `0=DOWN`, `1=UP` | 82–89 |
| `trained_at` | `2026-05-20T13:55:55.398539` | 95 |

`models/lstm_history.json` contains 29 values per curve; minimum `val_loss` = 0.689764 at epoch 22 (1-based), maximum `val_accuracy` = 0.532139 at epochs 16, 22 and 25.

### Recorded evaluation results (present in the repository — not re-measured in this audit)

From `models/lstm_metadata.json` lines 57–81:

| Metric | Recorded value |
|---|---:|
| Accuracy | 0.532139 |
| Precision (UP) | 0.534701 |
| Recall (UP) | 0.883350 |
| F1 (UP) | 0.666165 |
| ROC AUC | 0.537280 |
| Mean / median / p90 confidence | 0.5233 / 0.5187 / 0.5534 |
| Best val accuracy / val loss | 0.532139 / 0.689764 |

Confusion matrix `[[248, 1542], [234, 1772]]` (rows = actual DOWN, UP; columns = predicted DOWN, UP). Classification report: DOWN precision 0.51 / recall 0.14 / F1 0.22 (support 1790); UP precision 0.53 / recall 0.88 / F1 0.67 (support 2006); 3,796 test sequences.

**Interpretation:** accuracy ≈ 53.2% and ROC AUC ≈ 0.537 are close to chance. The high UP recall is driven by class imbalance (UP is 52.8% of the test set) and a strong bias toward predicting UP. These are *recorded historical metrics*, not a claim of predictive skill, and they were not independently recomputed during this read-only audit.

### Discrepancies found (important for viva honesty)

| Topic | Shipped artifact / original script | Current refactored script | Status |
|---|---|---|---|
| Learning rate | Checkpoint stores **2.5 × 10⁻⁴**; metadata records 0.001 | `train_lstm_model.py` line 92 sets 0.001 | The checkpoint's lower LR is consistent with `ReduceLROnPlateau(factor=0.5, patience=3)` (current script lines 223–229; backup lines 221–227) changing the optimizer before `ModelCheckpoint` saved the best model. **Plausible inference, not proven.** |
| Split | `chronological_80_20`, 5 symbols (metadata) | per-symbol chronological **70/15/15**, all eligible symbols (`train_lstm_model.py` lines 81–88, 151–162) | Current script was refactored **after** the checkpoint was trained; metadata schema shows it was not re-run. |
| Scaler fitting | Backup script lines 394–398 fits `MinMaxScaler` on **all** rows before splitting | Current script lines 409–423 fits on **training partition only** | The shipped scaler likely reflects the earlier full-data fit → **potential leakage in the original training run**. Current code is leakage-safe. |
| Target | Backup `feature_engineering.py` lines 196–213: UP if next return > +1%, DOWN if < −1%, neutral rows dropped | Current `lstm_data_pipeline.py` lines 202–219: 3-class with FLAT ±1%; `legacy_binary` mode skips FLAT labels (lines 285–286) | Both map to binary 0/1 for the shipped head; FLAT handling differs. |
| Metadata schema | `stable_symbols`, `split_strategy: chronological_80_20` | Current writer emits `target_convention`, `label_mode_used`, `partition_report` | Confirms two different generator versions. |

Evidence that the shipped metadata came from the backup script: `phase2_backup_20261124/train_lstm_model.py` lines 319–374 writes exactly the same keys (`stable_symbols`, `chronological_80_20`, architecture list, metrics block) in the same shape.

---

## 10.8 Inference implementation (verified)

| Concern | Implementation | Location |
|---|---|---|
| Model loading | `keras.models.load_model()`, cached in `_LSTM_MODEL_CACHE` | `scripts/predict_lstm.py` lines 79–107 |
| Scaler loading | `joblib.load()`, cached in `_LSTM_SCALER_CACHE`, never refit | lines 110–129 |
| Preflight | `prepare_symbol_frame()` → validate OHLCV, sort, dedupe, block-tag; gap > 30 days blocks inference | `scripts/lstm_inference.py` lines 369–450 |
| Feature computation | `compute_technical_features()` (the canonical pandas implementation) | `scripts/lstm_data_pipeline.py` lines 127–199 |
| Sequence | `df_feat.tail(60)`, scaled via `transform_sequences`, reshaped to `(1,60,21)` | `scripts/predict_lstm.py` lines 151–179 |
| Prediction | `model.predict(X, verbose=0)`, flatten, take first value | lines 252–255; `lstm_inference.py` line 510 |
| Threshold | `1 if prob_up >= 0.5 else 0` | `predict_lstm.py` line 256; `lstm_inference.py` line 541 |
| Output record | probabilities in %, confidence = `max(P_UP, P_DOWN) × 100` | `predict_lstm.py` lines 258–283; `lstm_inference.py` lines 539–563 |
| Parquet path | Spark-computed 21 features → sort by Date → `tail(60)` → original scaler → model | `lstm_inference.py` lines 567–707 |
| Data-source check | >25% of scaled values outside `[0,1]` raises a warning | lines 73–75, 438–445, 640–648 |
| Multi-stock | model/scaler loaded once; per-symbol failure isolated | lines 762–798 |
| Storage | `dedup_key = SHA256(symbol | model_sha256 | input timestamp | feature config)` | lines 805–851 |
| Spark parity | Spark computes the same 21 canonical columns per symbol | `scripts/spark_live_processing.py` lines 258–333 |
| Dashboard | displays recorded accuracy/AUC/F1 from metadata; single and multi-stock inference | `app.py` lines 320–324, 1408–1459, 1638–1782 |

Note: `app.py` line 320 displays `lstm_meta['metrics']['accuracy']` directly from `models/lstm_metadata.json` — it does **not** recompute accuracy at runtime.

---

## 10.9 What is NOT available or NOT verified

| Item | Status |
|---|---|
| Exact number of sequences used for training/validation/test of the shipped checkpoint | **NOT AVAILABLE** (metadata has no partition report) |
| Exact per-epoch optimizer learning-rate trace during training | **NOT AVAILABLE** (only final saved LR is readable) |
| Initial/random weight values before training | **NOT INSPECTED** (not needed for the audit) |
| Independent re-measurement of accuracy, AUC, loss | **NOT PERFORMED** (read-only task; recorded metrics only) |
| Test-suite execution results | **NOT PERFORMED**. `tests/test_phase4_lstm_inference.py` contains 12 test methods; the repository has 149 `def test_` methods across all test files. The “46 Phase 4 tests” figure in project reports was not re-verified here. |
| Phase 3 multi-task checkpoint | **NOT AVAILABLE** — no `phase3_*.keras` file in `models/`; only design code (`phase3_model.py`, `phase3_config.py`, `train_phase3_model.py`) |
| Live market prediction success rate | **NOT AVAILABLE** and not claimed |

---

## 10.10 Bottom line for the viva

1. The checkpoint is a real, loadable, 128,321-parameter Keras 2.15 model with a 3-input-tensor-free sequential stack: two LSTM layers, two Dropout layers, two Dense layers.
2. Its interface is exactly `(None, 60, 21) → (None, 1)` with sigmoid activation; it is binary UP/DOWN with threshold 0.5.
3. All preprocessing at inference reuses the **original** fitted scaler through `transform()`; no code path fits or modifies it.
4. The recorded accuracy (~53%) and AUC (~0.537) show **weak, near-chance signal** — present them as historical metadata values with the class-imbalance caveat, never as a performance guarantee.
5. The current training code is a newer, leakage-safe refactor that has **not** been rerun; do not describe it as the code that produced the shipped checkpoint.
6. Phase 3 (3-class + return) is design/future work with no artifact; say so explicitly.

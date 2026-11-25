# 08 — Deep Learning Comparison Tables

Quick-revision tables. "Relevance" columns refer to the verified project implementation.

---

## 8.1 ANN vs RNN vs LSTM

| Property | ANN (feed-forward/MLP) | RNN | LSTM (this project) |
|---|---|---|---|
| Structure | Input → hidden → output layers | Recurrent loop with hidden state | Recurrent loop + gated cell state |
| Memory | None between samples | Short hidden state | Hidden + long-term cell state |
| Input shape | `(batch, features)` | `(batch, timesteps, features)` | `(batch, 60, 21)` |
| Handles order? | No | Yes | Yes |
| Long dependencies | N/A | Poor | Good (gated additive memory) |
| Parameters | Fewest | Moderate | ~4× an RNN unit |
| Typical use | Tabular classification | Short sequences | Long sequences, time series |
| In this project | Random Forest and Dense head use row-level features | Not used | Deployed model uses two LSTM layers |

**Relevance:** the final `Dense(32, relu)` is an ANN-style layer applied to the LSTM's summary vector; the recurrence is what handles the 60-day window.

---

## 8.2 RNN vs LSTM vs GRU

| Property | Vanilla RNN | LSTM | GRU (not used here) |
|---|---|---|---|
| Gates | None | Forget, input, output | Reset, update |
| States | $h_t$ | $h_t$, $C_t$ | $h_t$ only |
| Vanishing gradient | Severe | Mitigated | Mitigated |
| Params per unit | $n^2+n$ | $4n^2+4n$ | $3n^2+3n$ |
| Training speed | Fastest | Slowest | Middle |
| Data needs | Moderate | More | Less |
| Project status | Not used | **Deployed** | Theoretically possible, not present |

**Relevance:** never claim the project uses GRU. If asked why not GRU, answer that the shipped checkpoint is LSTM and changing it would require retraining.

---

## 8.3 CNN vs RNN vs LSTM

| Property | CNN (1-D) | RNN | LSTM |
|---|---|---|---|
| Inductive bias | Local translation-invariant patterns | Sequential recurrence | Gated long-term recurrence |
| Receptive field | Fixed by kernel/dilation | Theoretically unbounded | Theoretically unbounded, more stable |
| Parallelism | High | Low (sequential) | Low |
| Long-range memory | Needs many layers/dilations | Weak | Strong |
| Best for | Local motifs (e.g. 5-day spike shapes) | Short sequences | Long sequential dependencies |
| In this project | None | None | Used |

**Relevance:** a 1-D CNN could detect local candlestick-like patterns in the 60-day window, but the project chose LSTM because trend persistence over the window matters.

---

## 8.4 Classification vs regression

| Property | Classification | Regression |
|---|---|---|
| Output type | Discrete class label | Continuous number |
| Output layer | Sigmoid / softmax | Linear |
| Loss | Cross-entropy | MSE / MAE / Huber |
| Example here | UP / DOWN direction | Predicted +1.4% return |
| Project status | **Implemented** | Not implemented (Phase 3 design only) |

**Relevance:** the shipped model is classification. A numeric next-session return is impossible without a different output head and target.

---

## 8.5 Binary vs multiclass classification

| Property | Binary | Multiclass |
|---|---|---|
| Classes | 2 | ≥3 |
| Output units / activation | 1 / sigmoid | $K$ / softmax |
| Loss | binary_crossentropy | categorical_crossentropy |
| Labels here | 0=DOWN, 1=UP | 0=DOWN, 1=FLAT, 2=UP (Phase 3 design) |
| Project status | **Deployed** | Designed but no checkpoint |

**Relevance:** the current data pipeline defines a 3-class target with a ±1% FLAT band, but `LABEL_MODE="legacy_binary"` skips FLAT labels so the deployed binary head remains usable. Do not claim three-class output.

---

## 8.6 Training vs inference

| Aspect | Training | Inference (this project) |
|---|---|---|
| Goal | Learn weights | Produce a prediction |
| Data | Labeled historical sequences | Latest 60 sessions, no label |
| Forward pass | Yes | Yes |
| Backpropagation | Yes (BPTT) | **No** |
| Weight updates | Yes | **No** |
| Dropout | Active | Inactive |
| Optimizer | Adam | Not used |
| Artifacts written | model, scaler, metadata, history | MongoDB prediction record |
| Code | `train_lstm_model.py` (not rerun for deployed model) | `predict_lstm.py`, `lstm_inference.py` |

---

## 8.7 MinMaxScaler vs StandardScaler

| Property | MinMaxScaler | StandardScaler |
|---|---|---|
| Transform | $(x-\min)/(\max-\min)$ | $(x-\mu)/\sigma$ |
| Output range | [0,1] (configurable) | Unbounded, ~mean 0 / std 1 |
| Sensitive to outliers | Yes | Less |
| Stored statistics | `data_min_`, `data_max_`, `scale_`, `min_` | `mean_`, `scale_` |
| Used here? | **Yes** — `models/lstm_scaler.pkl` | No |
| Changing it would | Preserve trained input mapping | Invalidate the deployed model |

**Relevance:** the audited scaler has `n_features_in_=21`, `feature_range=(0,1)` and no `data_mean_`, confirming MinMax.

---

## 8.8 Sigmoid vs Softmax vs Tanh vs ReLU

| Function | Formula | Range | Used here for |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | (0,1) | LSTM gates, final binary output |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | (0,1), sums to 1 | Not used (Phase 3 design only) |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | (−1,1) | LSTM candidate state and hidden transform |
| ReLU | $\max(0,z)$ | $[0,\infty)$ | `Dense(32)` hidden layer |

**Relevance:** sigmoid output → one probability; tanh inside cells; ReLU in the dense head. No softmax means no multiclass capability.

---

## 8.9 Batch GD vs SGD vs mini-batch GD

| Method | Samples per update | Pros | Cons | Project |
|---|---|---|---|---|
| Batch GD | All $N$ | Stable gradient | Slow, memory-heavy | Not used |
| SGD | 1 | Fast, noisy (escapes local minima) | Unstable | Not used |
| Mini-batch GD | $B$ (here 32) | Balanced, GPU/CPU efficient | Needs batch-size tuning | **Used** (batch_size=32) |

Iterations per epoch ≈ $N_\text{train}/32$. The exact $N_\text{train}$ for the shipped run is **NOT AVAILABLE**.

---

## 8.10 Overfitting vs underfitting

| Property | Underfitting | Overfitting |
|---|---|---|
| Training error | High | Very low |
| Validation error | High | High and rising |
| Cause | Model too simple / insufficient training | Memorising noise / too many epochs |
| Remedies | More capacity, better features, longer training | Dropout, early stopping, more data, regularisation |
| Signals here | Not observed | Mitigated by Dropout(0.2) and early stopping |

**Relevance:** actually, with accuracy ≈53% and loss ≈0.69 (near $\ln 2$), the larger practical risk is **underfitting / weak signal**, not overfitting. The history shows loss decreasing only slightly from 0.693 to 0.690 — the model learned very little.

---

## 8.11 One-glance project decisions

| Decision | Actual choice | Reason |
|---|---|---|
| Sequence model | LSTM (2 layers) | Temporal dependencies over 60 sessions |
| Output | 1 sigmoid neuron | Binary UP/DOWN |
| Loss | binary_crossentropy | Binary labels |
| Optimizer | Adam, stored LR 2.5e−4 | Adaptive learning rates |
| Regularisation | Dropout 0.2 ×2 + early stopping | Reduce overfitting |
| Scaling | MinMaxScaler (0,1) | Bounded, reused exactly |
| Split | Chronological (recorded 80/20; current code 70/15/15) | Avoid time leakage |
| Features | 21 canonical | Shared schema across training/live/Spark |
| Batch size | 32 | Mini-batch training |
| Multi-class / return | Not deployed | Requires Phase 3 checkpoint that does not exist |

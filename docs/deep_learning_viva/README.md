# Deep Learning Viva Preparation — Stock Market Intelligence Engine

A 10-document, implementation-grounded study pack explaining the deep-learning component of this project: theory first, then the **exact** architecture, features, scaling, inference path and recorded results actually present in the repository.

> **Read this first:** everything labelled **VERIFIED** comes from the saved artifacts or source files and is cited with paths and line numbers in [`10_project_implementation_audit.md`](10_project_implementation_audit.md). Everything labelled **THEORETICAL** is general background that is *not necessarily implemented* here.

---

## 30-second facts to memorise

| Question | Answer |
|---|---|
| What is the model? | A Keras `Sequential` LSTM binary classifier |
| Exact stack? | `LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid)` |
| Input / output? | `(None, 60, 21)` → `(None, 1)`, `float32` |
| Parameters? | 128,321 trainable, 0 non-trainable |
| Optimizer / loss in checkpoint? | Adam (`lr = 2.5e-4`), `binary_crossentropy` |
| Features? | 21, in the exact order listed in `scripts/model_config.py` lines 28–50 |
| Sequence? | 60 trailing trading sessions per symbol |
| Scaler? | The original fitted `MinMaxScaler(feature_range=(0,1))`, loaded with `joblib.load()` and used **only** via `transform()` |
| Output meaning? | One sigmoid probability `P(UP)`; `>= 0.5 → UP (1)`, else `DOWN (0)` |
| Does it predict price or return? | **No.** Binary direction only |
| Does it use a 3-class head? | **No.** No softmax head or Phase 3 checkpoint exists |
| Recorded accuracy / AUC? | 0.5321 / 0.5373 (from metadata; near chance, class-imbalanced) |
| Who produced the checkpoint? | The earlier Phase 2 training script preserved in `phase2_backup_20261124/`, not the current refactored trainer |

---

## Recommended learning order

### Tier 1 — Must know before entering the viva room

1. [`01_deep_learning_fundamentals.md`](01_deep_learning_fundamentals.md) — AI/ML/DL/ANN, neurons, forward pass, activations, loss, backprop, gradient descent, overfitting, train/val/test.
2. [`02_rnn_lstm_architecture.md`](02_rnn_lstm_architecture.md) — why sequences need RNN/LSTM, vanishing gradients, RNN vs LSTM vs GRU, why LSTM for stocks.
3. [`03_lstm_mathematics.md`](03_lstm_mathematics.md) — the seven gate equations, what each gate does, and how this differs from the exact Keras layer configuration.
4. [`04_time_series_and_sequences.md`](04_time_series_and_sequences.md) — timesteps, features, 60-window sliding, `(None, 60, 21)`, alignment of label to `t → t+1`.
5. [`05_feature_engineering_and_scaling.md`](05_feature_engineering_and_scaling.md) — the 21 exact features and MinMaxScaler / `fit_transform` vs `transform` / leakage.
6. [`06_binary_classification_and_metrics.md`](06_binary_classification_and_metrics.md) — sigmoid, threshold 0.5, confusion matrix, precision/recall/F1/ROC/AUC, why probability ≠ certainty.
7. [`10_project_implementation_audit.md`](10_project_implementation_audit.md) — the ground truth: exact architecture, hashes, recorded metrics, discrepancies.

### Tier 2 — Strongly recommended

8. [`07_model_training_vs_inference.md`](07_model_training_vs_inference.md) — training theory vs. the inference-only lifecycle actually running in this project.
9. [`08_deep_learning_comparisons.md`](08_deep_learning_comparisons.md) — ten comparison tables (ANN/RNN/LSTM, GRU, CNN, classification types, scalers, activations, optimizers, over/underfitting).
10. [`09_viva_questions_and_answers.md`](09_viva_questions_and_answers.md) — 70+ questions with short and detailed answers, including tricky examiner probes.

### Fastest route if you have one hour

Read §10.2–10.10 of document 10 → the summary tables of documents 4, 5 and 6 → the 20 "core questions" at the start of document 9.

---

## Most important concepts for THIS viva, ranked

| Rank | Concept | Why the examiner will ask |
|---:|---|---|
| 1 | Binary sigmoid output, not price/return/3-class | Common misconception; the prompt and code explicitly reject those claims |
| 2 | Why the **original** scaler must be reused with `transform()` | Direct check of whether you understand deployment consistency and leakage |
| 3 | `(None, 60, 21)` shape and what each dimension means | Tests whether you can read a Keras shape |
| 4 | Two LSTM layers and why the first needs `return_sequences=True` | Tests architecture understanding |
| 5 | Forget/input/output gates and cell state | Core LSTM theory |
| 6 | Sliding windows and per-symbol/per-block construction | Tests time-series competence and leakage awareness |
| 7 | Recorded accuracy ≈ 53% and AUC ≈ 0.537 | Tests intellectual honesty about weak signal |
| 8 | Training vs inference distinction | The project reuses a checkpoint; no training happens at runtime |
| 9 | Chronological split and leakage | The earlier pipeline fitted the scaler before splitting |
| 10 | Dropout as regularisation, present only during training | Tests whether you know inference behaviour |

---

## Document map

| File | Covers | Key deliverable |
|---|---|---|
| `01_deep_learning_fundamentals.md` | AI → DL hierarchy, neurons, backprop, gradient descent, epochs, regularisation | Foundation vocabulary |
| `02_rnn_lstm_architecture.md` | Sequential data, RNN limitations, LSTM/GRU, stock-market fit | Why LSTM was chosen |
| `03_lstm_mathematics.md` | Gate equations, numerical walk-through, Keras cell details | Math + Mermaid cell diagram |
| `04_time_series_and_sequences.md` | 60-step windows, batch dimension, sliding example, latest sequence | Sequence mechanics |
| `05_feature_engineering_and_scaling.md` | All 21 features with formulas, MinMaxScaler, leakage | Feature/scaler mastery |
| `06_binary_classification_and_metrics.md` | Sigmoid, threshold, metrics, imbalance, probability meaning | honest metric discussion |
| `07_model_training_vs_inference.md` | BPTT, optimizers, checkpoints vs. the actual inference pipeline | Lifecycle clarity |
| `08_deep_learning_comparisons.md` | 10 side-by-side tables | Fast revision |
| `09_viva_questions_and_answers.md` | 70+ Q&A in 11 categories | Rehearsal |
| `10_project_implementation_audit.md` | Evidence, hashes, exact config, discrepancies, unknowns | Source of truth |

---

## Accuracy rules used throughout this pack

- No invented layer, feature name or hyperparameter.
- No invented accuracy, loss or evaluation number; recorded values are explicitly cited and labelled as recorded.
- General theory is clearly separated from verified implementation.
- Unavailable information is marked **UNKNOWN / NOT AVAILABLE**.
- Anything not implemented (e.g. Phase 3 multi-task model, log returns, StandardScaler) is labelled **THEORETICAL / NOT USED**.
- This documentation task was strictly read-only: no code, weights, scaler, config or dataset was changed.

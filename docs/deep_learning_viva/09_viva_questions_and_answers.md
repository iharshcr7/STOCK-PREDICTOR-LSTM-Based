# 09 — Viva Questions and Answers

**Part A** contains the 20 core questions you are most likely to be asked. **Part B** is a categorized bank of additional questions (total 80+). Answers marked **Detailed** are the ones examiners probe further; **Tricky** questions test whether you actually understand the implementation.

All project-specific answers are grounded in [`10_project_implementation_audit.md`](10_project_implementation_audit.md).

---

# PART A — The 20 core questions

**A1. Why did you choose LSTM?**
Stock prices are a time series: today's direction depends on prior sessions. An LSTM carries hidden and cell states across a 60-day window, so it can learn trends and momentum that a single-row model cannot. In this project it is a reuse decision — the deployed checkpoint is already an LSTM.

**A2. What problem does the forget gate solve?**
Without forgetting, old information would accumulate indefinitely and interfere with new patterns. The forget gate multiplies the previous cell state by a learned value in (0,1), discarding stale memories such as an obsolete breakout.

**A3. What is the difference between hidden state and cell state?**
The cell state $C_t$ is internal long-term memory updated additively. The hidden state $h_t$ is the output-facing summary, produced by gating $\tanh(C_t)$, passed to the next timestep and finally to the prediction head.

**A4. Why are 60 timesteps used?**
60 is the configured sequence length of the existing checkpoint, not a scientifically proven optimum. It gives roughly three months of context and is long enough for the 26-period MACD and 20-period Bollinger windows to be meaningful.

**A5. Why are there 21 features?**
The first LSTM kernel has 21 input connections per unit and the saved MinMaxScaler has `n_features_in_=21`; the exact ordered list is in `scripts/model_config.py` lines 28–50. The number is fixed by the trained artifact, not chosen arbitrarily at inference.

**A6. Why is MinMaxScaler required?**
It rescales heterogeneous inputs (prices in thousands, RSI in 0–100, volume in billions) to a common [0,1] range. That conditions the loss surface, prevents saturation and makes the learned weights meaningful.

**A7. Why do you reuse the same scaler?**
The model's weights were learned on inputs produced by that exact fitted min/max mapping. Refitting on live data changes the mapping, so identical raw prices would produce different model inputs and inconsistent predictions.

**A8. What does the sigmoid output represent?**
A single estimated probability $P(\text{UP})$ for the session after the last input date. It is a model estimate under the training distribution, not a guarantee.

**A9. Why is the output layer a single neuron?**
Because the task is binary: one probability is sufficient, with 0→DOWN and 1→UP at threshold 0.5. Two neurons with softmax would be needed for multiclass.

**A10. Why is the model binary?**
The shipped checkpoint has one sigmoid output and the recorded labels are 0=DOWN, 1=UP. The three-class design (DOWN/FLAT/UP) exists only in unshipped Phase 3 code.

**A11. Does the model predict the exact next-day stock price?**
No. It outputs a binary direction label derived from a probability; no price value is produced anywhere in the inference path.

**A12. Does a probability of 0.8 guarantee an 80% chance of success?**
No. Sigmoid outputs need not be calibrated, the market distribution shifts, and important information (news, macro events) is absent from the features. Treat it as a model score, not a real-world odds statement.

**A13. How do you prevent data leakage?**
Use a chronological train/validation/test split by target date, fit the scaler only on training data, and compute features from trailing windows only. **Honest caveat:** the earlier script that produced the shipped checkpoint fitted the scaler before splitting; the current trainer fixes this.

**A14. What happens if a feature is missing?**
Preflight validation reports an explicit error and prediction is skipped — no zero-fill or invented value is used. `validate_feature_schema()` also raises on ordering mismatches.

**A15. Can the same model work for every stock?**
It can be *applied* to any symbol with enough valid history, but generalisation is not guaranteed. Metadata lists five stable training symbols; out-of-range warnings flag non-NASDAQ distributions, and generalisation requires validation.

**A16. What happens if the input shape changes?**
The tensor must be exactly `(batch, 60, 21)`. `load_bundle()` and `discover_artifacts()` validate the loaded model's input/output shapes; a 59- or 61-step window is rejected before prediction.

**A17. Why is inference different from training?**
Training uses labelled sequences and BPTT to update weights; inference uses unlabeled latest windows and performs only a forward pass. Dropout is inactive at inference and no optimizer step occurs.

**A18. How do you evaluate an LSTM model?**
On a chronological holdout using accuracy, precision, recall, F1, ROC AUC and a confusion matrix. The recorded test results are accuracy 0.532, AUC 0.537 with 3,796 sequences — near chance; they come from metadata and were not recomputed in this audit.

**A19. What are the limitations of LSTM for stock prediction?**
Weak/noisy signal, non-stationarity, class imbalance, fixed 60-step context, no attention, scaler sensitivity, binary output only and no calibrated uncertainty. These are stated explicitly in the project docs.

**A20. How can this project be improved?**
Complete full-universe distributed processing, automate ingestion, finish end-to-end HDFS–Spark–MongoDB integration, add a validated 3-class/regression checkpoint (Phase 3 design), improve monitoring and add portfolio/risk analytics — while keeping the deployed binary model honest.

---

# PART B — Extended question bank

## B1. Basic deep-learning concepts (5)

**B1.1 Define AI, ML and DL and their relationship.**
AI is the broad goal of intelligent machines; ML learns patterns from data; DL uses multi-layer neural networks. DL ⊂ ML ⊂ AI.

**B1.2 Is a Random Forest deep learning?**
No. It is an ensemble of decision trees — machine learning, not a neural network. This repository also contains RF models (`models/model_metadata.json`).

**B1.3 What does "end-to-end" mean in this project?**
The intended flow from data acquisition through HDFS/Spark, feature engineering, LSTM inference, MongoDB storage and Streamlit. The deployed binary model is the proven part; full pipeline integration is ongoing.

**B1.4 What is a model checkpoint?**
A saved snapshot of architecture and learned weights. Here `models/lstm_model.keras` (1.58 MB, Keras 2.15) is loaded with `keras.models.load_model()`.

**B1.5 What is a hyperparameter vs a parameter?**
Parameters (128,321 weights/biases) are learned; hyperparameters (batch size 32, 60 timesteps, dropout 0.2) are set before training.

## B2. Neural networks (5)

**B2.1 Write the neuron equation.**
$z=\mathbf{w}^\top\mathbf{x}+b$, $a=\phi(z)$.

**B2.2 Why non-linear activations?**
Without them, stacked linear layers collapse to one linear transform and cannot model complex patterns.

**B2.3 What is forward propagation?**
Layer-by-layer computation of the output; no learning occurs. Here it is `model.predict(X)`.

**B2.4 What is backpropagation?**
The chain rule computes gradients of the loss with respect to every weight, from output back to input.

**B2.5 What is gradient descent and what is a learning rate?**
Weights move opposite to the loss gradient, scaled by $\eta$. Here Adam with stored $\eta=2.5\times10^{-4}$.

## B3. RNN and LSTM (6)

**B3.1 What is sequential data?**
Data where order matters — here consecutive trading sessions of one symbol.

**B3.2 What is an RNN?**
A network that reuses weights across timesteps and carries a hidden state.

**B3.3 What is the vanishing gradient problem?**
Gradients shrink across timesteps, so distant inputs barely affect learning. Shortened memory would hurt a 60-step window.

**B3.4 What is an exploding gradient?**
Gradients grow exponentially, destabilising updates; gradient clipping is the standard remedy.

**B3.5 LSTM vs GRU — key difference.**
LSTM has three gates and two states; GRU has two gates and one state, so it is cheaper. GRU is not used here.

**B3.6 Why two LSTM layers?**
The first (128 units) produces rich per-timestep features with `return_sequences=True`; the second (64 units) compresses the window into one vector for the Dense head.

## B4. LSTM cell architecture (6)

**B4.1 State all seven LSTM equations.**
See document 03, section 3.2: forget, input, candidate, cell state, output, hidden state, final sigmoid output.

**B4.2 What are the dimensions of the gates in LSTM 1?**
Each gate vector has 128 components, computed from the 21 inputs concatenated with 128 previous hidden values.

**B4.3 Why sigmoid for gates and tanh for the candidate?**
Sigmoid yields gating coefficients in (0,1); tanh yields zero-centred content in (−1,1).

**B4.4 What is element-wise multiplication for?**
It gates each memory dimension independently, enabling selective retention.

**B4.5 Where does the parameter count 76,800 come from?**
$4n(d+n)+4n$ with $n=128$, $d=21$: four gate matrices of size 128×(21+128) plus biases.

**B4.6 Are the LSTM layers' internal dropout rates 0.2?**
No — they are 0.0. Dropout 0.2 comes from two separate `Dropout` layers. (Tricky; a common mistake.)

## B5. Time-series processing (6)

**B5.1 What does the batch dimension `None` mean?**
Variable number of sequences per call: 1 for live inference, many for training/evaluation.

**B5.2 How many overlapping windows come from 100 records with L=60?**
41 naive; fewer after dropping indicator warm-up rows. Illustrative only.

**B5.3 What is a `Sequence_Block`?**
A contiguous history segment; a >7-day calendar gap starts a new block, and windows never cross blocks.

**B5.4 How is the final sequence selected?**
The last 60 complete, ordered feature rows for the chosen symbol (`tail(60)`).

**B5.5 What label does a window ending at t carry?**
The direction of the t→t+1 return: below −1% DOWN, above +1% UP, within ±1% FLAT/skipped for the binary head.

**B5.6 Why must feature order stay fixed?**
The trained kernel columns and scaler statistics are position-indexed; swapping features corrupts the mapping.

## B6. Feature engineering (6)

**B6.1 Name the 21 features in order.**
Open, High, Low, Close, Volume, MA5, MA10, MA20, EMA12, RSI, MACD, MACD_Signal, Daily_Returns, Volatility, Price_Change_Pct, Weekly_Momentum, Trend_Strength, BB_Width, Avg_5D_Volume_Trend, Lag_1, Lag_3.

**B6.2 What is RSI(14) here?**
Wilder-smoothed relative strength index with α=1/14 and explicit zero-loss conventions.

**B6.3 What is MACD and its signal line?**
EMA12 − EMA26; signal = 9-period EMA of MACD.

**B6.4 Is Volatility the standard deviation of returns?**
No. It is `High − Low`. (Tricky.)

**B6.5 Are log returns used?**
No — simple percentage returns only.

**B6.6 Which indicators are computed but not modelled?**
`BB_Position`, `MACD_Hist`, `Volume_Change_Pct`, `Avg_Volume_Trend`, `Rolling_Std_Returns` (diagnostic only).

## B7. Data scaling (5)

**B7.1 What is the MinMaxScaler equation?**
$x'=(x-\min)/(\max-\min)$, implemented as $x\cdot\text{scale\_}+\text{min\_}$, range (0,1).

**B7.2 What is the difference between `fit_transform` and `transform`?**
`fit_transform` learns statistics and applies them; `transform` applies already-learned statistics only.

**B7.3 Why can't you fit a new scaler on live data?**
It changes the input distribution and would leak future information; the model expects the original mapping.

**B7.4 What happens to out-of-range live values?**
They are clipped to [0,1]; a warning is raised if >25% are out of range.

**B7.5 What evidence confirms the scaler type?**
`data_min_`, `data_max_`, `scale_`, `min_` exist, `data_mean_` does not, `n_features_in_=21`, `feature_range=(0,1)`.

## B8. Binary classification (6)

**B8.1 What is a decision threshold?**
The probability cut-off that converts $P(\text{UP})$ into a label; here 0.5, with `>=` so 0.5 → UP.

**B8.2 Define precision, recall and F1.**
See document 06, section 6.6; formulas and project values there.

**B8.3 What is ROC AUC?**
Area under the TPR–FPR curve across thresholds; 0.5 is random, recorded value 0.537.

**B8.4 Why is accuracy poor here?**
Majority-class baseline ≈52.8% versus recorded 53.2%.

**B8.5 What does the confusion matrix show?**
UP predicted 87% of the time; high UP recall (0.88) but DOWN recall only 0.14.

**B8.6 Why can't one sigmoid neuron produce three classes?**
A single sigmoid yields one scalar; three classes need three softmax logits and a different trained checkpoint.

## B9. Model inference (6)

**B9.1 What is the exact inference chain?**
Validation → features → 60-row sequence → original scaler → original checkpoint → sigmoid → threshold → record (document 07, section 7.3).

**B9.2 How are model and scaler loaded?**
`keras.models.load_model()` and `joblib.load()`, both cached in module-level variables.

**B9.3 What is artifact discovery?**
A read-only check of `.keras/config.json`, metadata and scaler that validates shapes, order and capabilities before loading.

**B9.4 What does the MongoDB dedup key contain?**
SHA-256 of symbol + model hash + input timestamp + feature-config version.

**B9.5 What is multi-stock inference?**
`infer_symbols()` loops over symbols with one cached model; one failure does not abort the batch.

**B9.6 Why are no weights updated at inference?**
`predict()` is forward-only, the optimizer is never called, and the audit verified weights unchanged.

## B10. Stock-market applications (4)

**B10.1 Why not just use a moving-average rule?**
Rules are fixed; the LSTM learns how combinations of indicators map to direction.

**B10.2 What is the risk of using delayed daily data?**
The prediction refers to the session after the last *available* date, not necessarily tomorrow; the record stores that timestamp.

**B10.3 Why mention NASDAQ training?**
Metadata lists AAPL, AMZN, GOOGL, MSFT, NVDA; generalisation to non-NASDAQ stocks needs validation.

**B10.4 Should a 62% probability justify a trade?**
No. There are no transaction costs, risk model or calibrated uncertainty; probabilities do not guarantee movement.

## B11. Project-specific implementation (13)

**B11.1 What is the exact architecture?**
`LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid)`; 128,321 parameters.

**B11.2 What are the input and output shapes/dtypes?**
`(None,60,21) → (None,1)`, `float32`.

**B11.3 What optimizer and loss are stored in the checkpoint?**
Adam (`lr≈2.5e−4`, β₁=0.9, β₂=0.999) and binary_crossentropy with accuracy metric.

**B11.4 Why does metadata say 0.001 while the checkpoint says 0.00025?**
Metadata records the configured LR; `ReduceLROnPlateau` likely halved it before the best checkpoint was saved. Plausible inference, not proven.

**B11.5 How many epochs were trained and how many test samples?**
29 epochs of 30 configured; 3,796 test sequences (from the classification report in metadata).

**B11.6 What recorded metrics should you quote?**
Accuracy 0.532, precision 0.535, recall 0.883, F1 0.666, ROC AUC 0.537, best val loss 0.6898 — with the caveat that these are historical metadata values.

**B11.7 Did the current training script produce the deployed model?**
No. The metadata schema and 80/20 split match the backup script; the current 70/15/15 refactor was not rerun.

**B11.8 Was there leakage in the original training?**
The matching backup script fitted the scaler before splitting — evidence of leakage. The current trainer fits on training rows only.

**B11.9 Does a Phase 3 checkpoint exist?**
No. Only design code for a 3-class + return model; no `phase3_*.keras` in `models/`.

**B11.10 How does the dashboard get accuracy?**
It reads `lstm_meta['metrics']['accuracy']` from metadata (`app.py` line 320); it does not recompute it.

**B11.11 What happens when data has a 45-day gap?**
`preflight_symbol()` blocks inference with a "data gap" error; smaller gaps produce warnings.

**B11.12 Why is LSTM live-mode only in the UI?**
A manual snapshot has one row, but the model requires a 60-session sequence.

**B11.13 What was verified in this read-only audit that you can quote confidently?**
Exact layer list/parameter count, input/output shapes and dtypes, stored optimizer state, scaler type/statistics, 21-feature order, recorded metrics, and that inference leaves weights unchanged.

---

## Final advice

- If you do not know a number, say **"NOT AVAILABLE in the repository"** rather than guessing.
- Always separate **general LSTM theory** from **this checkpoint's exact configuration**.
- When quoting accuracy, add: *"That is a recorded historical value from metadata; AUC 0.537 shows weak signal and I am not claiming predictive reliability."*

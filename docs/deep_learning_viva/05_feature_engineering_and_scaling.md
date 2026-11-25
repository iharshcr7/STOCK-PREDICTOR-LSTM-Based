# 05 — Feature Engineering and Scaling

**VERIFIED feature list** from `scripts/model_config.py` lines 28–50, `scripts/lstm_data_pipeline.py` lines 65–70 and `models/lstm_metadata.json` lines 5–27. **THEORETICAL** scaling concepts are marked where appropriate.

---

## 5.1 Why feature engineering is needed

Raw price and volume are non-stationary and hard to compare across stocks priced at $5 or $2,000. Technical indicators transform prices into *relative* signals (trend, momentum, volatility, volume pressure) that are more comparable and carry domain knowledge the LSTM can exploit.

**Project flow:** `scripts/preprocessing.py` → `scripts/feature_engineering.py` → `scripts/lstm_data_pipeline.py` → `scripts/predict_lstm.py` / `scripts/lstm_inference.py`. The canonical indicator definitions live in `compute_technical_features()` (`lstm_data_pipeline.py` lines 127–199) and are reused by training, live and Spark paths.

---

## 5.2 The exact 21 model inputs, in order

Let $C_t$ = Close at session $t$, $O_t,H_t,L_t,V_t$ = Open, High, Low, Volume. Rolling windows are trailing and use `min_periods = window`, so warm-up rows are NaN and dropped.

| # | Feature | Exact definition in code | Financial meaning / why useful | Quick example (HYPOTHETICAL) |
|---:|---|---|---|---|
| 1 | `Open` | raw Open | Session opening price; baseline for intraday change | $100.50 |
| 2 | `High` | raw High | Session maximum; captures buying pressure | $105.00 |
| 3 | `Low` | raw Low | Session minimum; captures selling pressure | $99.50 |
| 4 | `Close` | raw Close | Consensus price; anchor for returns | $104.00 |
| 5 | `Volume` | raw Volume | Participation/liquidity | 1,000,000 |
| 6 | `MA5` | `close.rolling(5, min_periods=5).mean()` | 5-day simple moving average; short trend | closes 101,102,101,103,104 → 102.2 |
| 7 | `MA10` | 10-day SMA | Medium-short trend | — |
| 8 | `MA20` | 20-day SMA | ~1-month trend | — |
| 9 | `EMA12` | `close.ewm(span=12, min_periods=12, adjust=False).mean()` | Recency-weighted trend; faster reaction than SMA | $\alpha=2/13\approx0.1538$ |
| 10 | `RSI` | Wilder RSI(14): gains/losses smoothed with `ewm(alpha=1/14, adjust=False)`, $RSI=100-100/(1+RS)$ | Momentum oscillator; overbought/oversold context | RSI 70 suggests strong recent gains |
| 11 | `MACD` | `EMA12 - EMA26` | Trend-following momentum difference | positive → short EMA above long EMA |
| 12 | `MACD_Signal` | `MACD.ewm(span=9, min_periods=9, adjust=False).mean()` | Smoothed MACD; crossover signal line | — |
| 13 | `Daily_Returns` | $(C_t/C_{t-1}-1)\times100$ | Daily percentage change | 103 → 104 gives 0.971% |
| 14 | `Volatility` | $H_t-L_t$ (**intraday price range, not std-dev of returns**) | Intraday uncertainty/range | 105 − 99.5 = 5.5 |
| 15 | `Price_Change_Pct` | $(C_t-O_t)/O_t\times100$ | Intraday body direction/strength | (104−100.5)/100.5 ≈ 3.48% |
| 16 | `Weekly_Momentum` | $(C_t/C_{t-5}-1)\times100$ | One-week return; short momentum | 99 → 104 gives 5.05% |
| 17 | `Trend_Strength` | $|MA5-MA20|/MA20\times100$ | Magnitude of short-vs-medium trend divergence | MA5 102.2, MA20 100 → 2.2% |
| 18 | `BB_Width` | $((MA20+2\sigma_{20})-(MA20-2\sigma_{20}))/MA20\times100 = 4\sigma_{20}/MA20\times100$ | Bollinger band width = volatility normalised by price | wider band → higher volatility |
| 19 | `Avg_5D_Volume_Trend` | $V_t/\text{MA}_5(V_t)$ | Volume relative to recent average; confirms moves | 1.4 means 40% above 5-day average |
| 20 | `Lag_1` | $C_{t-1}$ | Previous close; short-term autoregressive cue | $103 |
| 21 | `Lag_3` | $C_{t-3}$ | Three-session-ago close; very short memory | $101 |

**Sources:** `scripts/lstm_data_pipeline.py` lines 140–188. Spark reimplements the same 21 columns with per-symbol windows in `scripts/spark_live_processing.py` lines 258–333.

### Indicators explicitly discussed but NOT model inputs

| Indicator | Status |
|---|---|
| `BB_Position`, `MACD_Hist`, `Volume_Change_Pct`, `Avg_Volume_Trend`, `Rolling_Std_Returns` | Computed as **diagnostic** columns (`lstm_data_pipeline.py` lines 190–197) but excluded from `FEATURE_COLUMNS` |
| Log returns $\ln(C_t/C_{t-1})$ | **THEORETICAL / NOT USED**. Project uses simple percentage returns |
| Adjusted Close | Present in raw CSVs but removed by preprocessing (`scripts/preprocessing.py` lines 140–146) and not a model feature |
| Stochastic oscillator, OBV, ATR | Not implemented |

**Viva caution:** "Bollinger Bands" are represented in the model only through `BB_Width`, not the full upper/lower band or band position.

---

## 5.3 RSI special conventions (VERIFIED)

`lstm_data_pipeline.py` lines 152–164:

- Gains = positive price differences; losses = absolute negative differences.
- Smoothed with Wilder's method: `ewm(alpha=1/14, min_periods=14, adjust=False)`.
- $RS=\text{avg\_gain}/\text{avg\_loss}$, $RSI=100-100/(1+RS)$.
- If avg_loss = 0 and avg_gain > 0 → RSI = 100.
- If both are zero → RSI = 50 (neutral).

These conventions keep valid flat periods usable rather than producing NaN or infinity.

---

## 5.4 Why scaling is required (THEORY)

Neural networks use gradient-based optimisation. If one input has range $[10^6,10^9]$ (Volume) and another $[0,100]$ (RSI), the loss surface becomes extremely elongated: gradients along the large feature dominate, and learning is slow or unstable.

Feature scaling puts all inputs on a comparable range so:
- the loss surface is better conditioned,
- learning rates work uniformly across features,
- sigmoid/tanh activations are not immediately saturated,
- distances and dot products are meaningful.

---

## 5.5 MinMaxScaler (THEORY + VERIFIED)

MinMaxScaler performs linear rescaling to a target range, here $(0,1)$:

$$
x' = \frac{x - x_{\min}}{x_{\max}-x_{\min}}
$$

scikit-learn stores `scale_ = 1/(max-min)` and `min_` so that:

$$
x' = x\cdot \text{scale\_} + \text{min\_}
$$

**VERIFIED scaler attributes** (audit §10.5): `MinMaxScaler`, `n_features_in_=21`, `feature_range=(0,1)`, `data_mean_` absent. Example fitted boundaries: Open min 0.0946 / max 2173.07; Volume min 347,200 / max 1.855×10⁹; RSI min 13.03 / max 93.40.

**Worked example (HYPOTHETICAL):** if Close has fitted min 90 and max 110, then Close 105 → $(105-90)/20 = 0.75$. If a live Close is 130 → $2.0$, i.e. outside the training range.

### `fit`, `fit_transform` and `transform`

| Method | Action | When to use |
|---|---|---|
| `fit(X)` | Learns min/max, changes the scaler object | Only on training data, once |
| `fit_transform(X)` | Fit then transform | Only on training data |
| `transform(X)` | Apply stored min/max, no learning | Validation, test and live inference |

**VERIFIED:** the original scaler is loaded with `joblib.load()` in `scripts/predict_lstm.py` lines 110–129 and passed to `transform_sequences()`. No inference path calls `fit` or `fit_transform`; `scripts/lstm_inference.py` lines 243–259 even validates that the loaded object is a 21-feature MinMaxScaler with range (0,1).

### Why re-fitting on live data is wrong

1. Live data has few rows, so min/max would change arbitrarily.
2. The model's first-layer weights were learned on the original distribution; new scaling means the same raw price maps to a different input value → inconsistent predictions.
3. Re-fitting leaks test/future information into preprocessing.

**VERIFIED mitigation:** `transform_sequences()` (`lstm_data_pipeline.py` lines 412–420) flattens the 3-D window to 2-D, calls `scaler.transform(flat)`, clips to `[0,1]`, and reshapes back. The inference module warns when more than 25% of values fall outside `[0,1]` (`lstm_inference.py` lines 73–75, 438–445, 640–648) — a distribution-shift signal, never a trigger to refit.

### Leakage caveat from the original training run

The backup script that matches the shipped metadata (`phase2_backup_20261124/train_lstm_model.py` lines 394–398) fits `MinMaxScaler` on **all** rows before splitting. The current refactored trainer fixes this: `scripts/train_lstm_model.py` lines 409–423 fits only on training partitions. State this distinction clearly in the viva.

---

## 5.6 Data validation before features (VERIFIED)

`prepare_symbol_frame()` (`lstm_data_pipeline.py` lines 91–120):
- requires `Symbol`, `Date`, OHLCV;
- coerces dates and numerics, drops invalid rows;
- requires positive prices and non-negative volume;
- sorts ascending by date;
- duplicates by date keep the **last** occurrence;
- tags `Sequence_Block` when a calendar gap > 7 days occurs.

`scripts/preprocessing.py` lines 105–130 uses forward-fill within symbol then drops remaining NaNs for the historical dataset.

---

## 5.7 Viva questions and answers

**Q1. How many features does your model take and where is the list defined?**
21, defined in `scripts/model_config.py` lines 28–50 and mirrored in metadata and the pipeline.

**Q2. Name five non-price features.**
MA5, EMA12, RSI, MACD, BB_Width (also MACD_Signal, Daily_Returns, Weekly_Momentum, etc.).

**Q3. What is RSI and how is it computed here?**
Wilder's 14-period RSI using exponentially smoothed gains/losses with α=1/14; `RSI = 100 − 100/(1+RS)`.

**Q4. What is EMA12 and how does it differ from MA5?**
`ewm(span=12, adjust=False)` — an exponentially weighted average emphasising recent prices; MA5 is an unweighted 5-day rolling mean.

**Q5. What is MACD and why use it?**
EMA12 minus EMA26; a trend/momentum indicator. Its signal line is the 9-period EMA of MACD.

**Q6. What does BB_Width measure?**
Bollinger band width as a percentage of MA20: `4*std20/MA20*100`. It measures volatility normalised by price.

**Q7. Is Volatility the standard deviation of returns?**
No — in this project it is simply `High − Low`, the intraday price range.

**Q8. Are log returns used?**
No. Daily_Returns is simple percentage change.

**Q9. Is Adjusted Close a model input?**
No. It is dropped in preprocessing and is not in the 21-feature schema.

**Q10. What is MinMaxScaler's equation and output range?**
$x'=(x-\min)/(\max-\min)$, here scaled to (0,1) via stored `scale_` and `min_`.

**Q11. Why scale before feeding an LSTM?**
To equalise feature magnitudes, improve gradient conditioning and avoid saturation.

**Q12. What is the difference between `fit_transform` and `transform`?**
`fit_transform` learns min/max and applies them; `transform` only applies already-learned statistics.

**Q13. Why must you reuse the original fitted scaler?**
The weights were trained on that exact mapping. Refitting changes the input distribution and therefore changes predictions for the same raw data.

**Q14. What happens to out-of-range live values?**
`transform_sequences()` clips them to [0,1]; the inference module warns if >25% are out of range.

**Q15. What is data leakage and did it occur here?**
Using information from the future/test set during training. The original script fitted the scaler before splitting (evidence of leakage); the current trainer fits on training data only.

**Q16. Why does the scaler store `n_features_in_=21` matter?**
It enforces the exact feature count/order; a different schema is rejected.

**Q17. How does Spark keep feature parity?**
`spark_live_processing.py` computes the same 21 named columns with `Window.partitionBy("Symbol").orderBy("Date")`.

**Q18. (Tricky) If Volume is scaled to [0,1] using training min/max, what does a value of 1 mean?**
It means that session's volume reached the maximum observed in the scaler's fitting data — not "maximum possible volume".

**Q19. (Tricky) Why are warm-up rows dropped instead of zero-filled?**
Zero would be a fabricated indicator value inconsistent with the trained distribution; dropping keeps only genuinely computed features.

**Q20. (Tricky) Does scaling happen before or after sequence construction?**
In the current training code, sequences are built first, then the scaler is fitted on training sequences and applied to all partitions (`train_lstm_model.py` lines 409–424). At inference, features are computed, the 60-row window is taken, then `transform()` is applied.

**Q21. (Tricky) Could you use StandardScaler instead?**
Technically yes for a new model, but this checkpoint's scaler is MinMaxScaler and must be reused; swapping would invalidate the deployed model.

**Q22. What diagnostic features exist but are not fed to the model?**
`BB_Position`, `MACD_Hist`, `Volume_Change_Pct`, `Avg_Volume_Trend`, `Rolling_Std_Returns`.

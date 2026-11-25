# 06 — Binary Classification, Probability and Metrics

**THEORY** with a **VERIFIED** section on this project's recorded results.

---

## 6.1 Binary classification

The task maps each input to one of two labels:

$$
y \in \{0,1\},\qquad 0=\text{DOWN},\quad 1=\text{UP}
$$

**VERIFIED:** `models/lstm_metadata.json` lines 82–89 defines `labels: {"0":"DOWN","1":"UP"}`, `threshold: 0.5`, `confidence_source: "sigmoid_output_probability"`. The model has one output unit with sigmoid activation (`config.json`).

---

## 6.2 Sigmoid and why its output is a probability-like value

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

Properties: $\sigma(-\infty)=0$, $\sigma(0)=0.5$, $\sigma(+\infty)=1$, strictly increasing, derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$.

Because the range is $(0,1)$ and training used binary cross-entropy, the output can be interpreted as the model's **estimated** probability of the positive class:

$$
\hat p = P(y=1\mid x) \approx \sigma(z)
$$

**Numerical (HYPOTHETICAL):** $z=1.5 \Rightarrow \hat p=0.818$; $z=-0.7 \Rightarrow \hat p=0.332$.

**Caveat:** sigmoid output is *calibrated-ish*, not guaranteed. It is not a frequency count and does not guarantee real-world odds (section 6.8).

---

## 6.3 Classification threshold

$$
\hat y = \begin{cases}1\;(\text{UP}) & \hat p \ge \tau\\ 0\;(\text{DOWN}) & \hat p < \tau\end{cases}
\qquad \tau=0.5
$$

**VERIFIED:** `scripts/predict_lstm.py` line 256 uses `y_pred = 1 if y_proba >= 0.5 else 0`; `scripts/lstm_inference.py` line 541 uses the metadata threshold. **Exactly 0.5 is classified UP** because the comparison is `>=`.

The threshold is a decision policy, not a model parameter. Raising it makes UP predictions more conservative; lowering it increases UP predictions.

---

## 6.4 Probabilities and confidence reported by the project

**VERIFIED:** `scripts/lstm_inference.py` lines 539–563 and `scripts/predict_lstm.py` lines 258–283:

- $p_\text{UP} = \hat p$
- $p_\text{DOWN} = 1-\hat p$
- `confidence = max(p_UP, p_DOWN) × 100` (percent)
- `probabilities` dictionary reports both as percentages

**Example (HYPOTHETICAL):** if `raw_output = 0.62`, then UP = 62%, DOWN = 38%, confidence = 62%, label = UP.

Note that confidence here is simply the larger probability; it is **not** a calibrated confidence interval.

---

## 6.5 Confusion matrix

Rows = actual class, columns = predicted class.

|  | Predicted DOWN | Predicted UP |
|---|---:|---:|
| **Actual DOWN** | True Negative (TN) | False Positive (FP) |
| **Actual UP** | False Negative (FN) | True Positive (TP) |

**VERIFIED recorded matrix** (`models/lstm_metadata.json` lines 71–80):

|  | Pred DOWN | Pred UP | Row total |
|---|---:|---:|---:|
| Actual DOWN | 248 (TN) | 1542 (FP) | 1790 |
| Actual UP | 234 (FN) | 1772 (TP) | 2006 |
| Column total | 482 | 3314 | 3796 |

Interpretation: the model predicted UP for 3314/3796 = 87.3% of test sequences. It correctly found most UP days (1772/2006) but missed 86% of DOWN days. This is the classic signature of a majority-bias classifier on an imbalanced dataset.

---

## 6.6 Evaluation metrics (formulas)

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\text{Precision}=\frac{TP}{TP+FP}\quad\text{(of predicted UP, how many were UP)}
$$

$$
\text{Recall (TPR)}=\frac{TP}{TP+FN}\quad\text{(of actual UP, how many were found)}
$$

$$
\text{F1}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

$$
\text{Specificity}=\frac{TN}{TN+FP}
$$

**Hypothetical worked example:** TN=40, FP=10, FN=20, TP=30 → accuracy = 70/100 = 0.70; precision = 30/40 = 0.75; recall = 30/50 = 0.60; F1 = 0.667; specificity = 40/50 = 0.80.

---

## 6.7 ROC curve and AUC

A ROC curve plots True Positive Rate against False Positive Rate as the decision threshold sweeps from 1 to 0. AUC is the area under that curve:

- AUC = 0.5 → random ranking
- AUC = 1.0 → perfect separation
- AUC < 0.5 → worse than random (inverted)

AUC is threshold-independent and therefore more informative than accuracy when classes are imbalanced.

**VERIFIED recorded AUC:** 0.537280 (`models/lstm_metadata.json` line 62) — only marginally above chance.

---

## 6.8 Probability is not certainty

A prediction of $p=0.8$ does **not** mean "80% chance the stock goes up" unless the model is perfectly calibrated and the data-generating process is stable. Reasons:

1. **Calibration:** sigmoid outputs are optimised for cross-entropy ranking, not exact probability calibration.
2. **Distribution shift:** live data may lie outside the scaler's fitted range.
3. **Missing information:** news, macro events and order flow are not inputs.
4. **Binary label:** a ±1% threshold discards magnitude; "UP" may mean +1.01% or +15%.
5. **Weak signal:** recorded AUC ≈ 0.537.

**Viva formulation:** "It is the model's estimated probability under the training distribution, not a guarantee of market movement."

---

## 6.9 Class imbalance

**VERIFIED:** test set has 1790 DOWN vs 2006 UP (52.8% UP). A trivial "always UP" classifier would achieve 2006/3796 = 52.8% accuracy — almost exactly the recorded 53.2%. That explains why accuracy alone is misleading and why UP recall (0.883) is high while DOWN recall is only 0.139.

Remedies (THEORETICAL — not implemented here): class weights, resampling, threshold tuning, focal loss.

---

## 6.10 Why this model cannot produce 3 classes or returns

| Desired output | Required change |
|---|---|
| 3 classes (DOWN / FLAT / UP) | Target must define FLAT (current pipeline does: ±1%), output head must be `Dense(3, softmax)` trained with categorical cross-entropy, and a matching checkpoint must exist |
| Numerical next-session return | Target must be continuous `Next_Return_Pct`, output `Dense(1, linear)` trained with MSE/Huber |

**VERIFIED:** the deployed checkpoint has `Dense(1, sigmoid)` only. Phase 3 code (`scripts/phase3_model.py` lines 102–145, `scripts/phase3_config.py` lines 58–100) defines a two-head 3-class + Huber-regression model, but **no `phase3_*.keras` artifact exists** in `models/`. Inference explicitly reports `three_class: false` and `next_session_return: false` with an empty return field (`scripts/lstm_inference.py` lines 302–306, 556–560).

---

## 6.11 Recorded metrics table (VERIFIED — from metadata, not re-measured)

| Metric | Value | Source |
|---|---:|---|
| Accuracy | 0.532139 | `lstm_metadata.json` line 58 |
| Precision (UP) | 0.534701 | line 59 |
| Recall (UP) | 0.883350 | line 60 |
| F1 (UP) | 0.666165 | line 61 |
| ROC AUC | 0.537280 | line 62 |
| Mean / median / p90 confidence | 0.5233 / 0.5187 / 0.5534 | lines 63–67 |
| Best val accuracy / val loss | 0.532139 / 0.689764 | lines 68–69 |
| Test sequences | 3796 | classification report, line 81 |

These are historical values stored in the repository. They were **not independently recomputed** during this read-only audit, and no new performance claim is made.

---

## 6.12 Viva questions and answers

**Q1. What is binary classification?**
Mapping an input to one of two labels; here 0=DOWN, 1=UP.

**Q2. Why sigmoid at the output?**
It maps any real number to (0,1), giving a probability-like $P(\text{UP})$.

**Q3. What does a probability of 0.8 mean?**
The model's estimated P(UP) under its training distribution — not a guarantee and not necessarily calibrated.

**Q4. What happens at exactly 0.5?**
The code uses `>= 0.5`, so it is classified UP (1).

**Q5. What is the threshold and where is it stored?**
0.5, in `models/lstm_metadata.json` line 87; applied in `predict_lstm.py` line 256 and `lstm_inference.py` line 541.

**Q6. How is confidence calculated?**
`max(P_UP, P_DOWN) × 100`, e.g. P_UP=0.62 → 62%.

**Q7. Define precision and recall in this context.**
Precision = of predicted UP days, fraction truly UP; recall = of actual UP days, fraction found by the model.

**Q8. Why is accuracy misleading here?**
The test set is 52.8% UP; always predicting UP gives ≈52.8% accuracy, close to the recorded 53.2%.

**Q9. What is F1 and when is it useful?**
Harmonic mean of precision and recall; useful when you want a balance and classes are imbalanced.

**Q10. What is ROC AUC and what does 0.537 mean?**
Threshold-independent ranking quality; 0.5 is random, so 0.537 is only marginally above chance.

**Q11. What does the confusion matrix tell you about this model?**
It predicts UP 87% of the time, finding most UP days but missing 86% of DOWN days.

**Q12. Why not use accuracy for early stopping?**
The stored checkpoint used `val_loss` (binary cross-entropy), which is smooth and uses probability information.

**Q13. Does the model predict next-day price?**
No. It predicts a binary direction class for the session after the last input date.

**Q14. Can it predict a +2.3% return?**
No. That requires a regression head, which does not exist in the shipped checkpoint.

**Q15. Can it output FLAT?**
No. The deployed head has one sigmoid neuron; FLAT belongs to the unshipped Phase 3 design.

**Q16. What is class imbalance and how does it show up here?**
Unequal class counts; it produces high UP recall and low DOWN recall.

**Q17. How would you improve evaluation?**
Report per-class precision/recall/F1, AUC, confusion matrix, and use a chronological holdout; never rely on accuracy alone.

**Q18. (Tricky) If you changed the threshold to 0.7, would accuracy necessarily improve?**
No. It would reduce UP predictions; given the UP-heavy data it would likely reduce recall and possibly accuracy. Threshold tuning requires validation data.

**Q19. (Tricky) Is a 53% accuracy better than random for this dataset?**
Barely. The majority-class baseline is 52.8%, so the improvement is ~0.4 percentage points.

**Q20. (Tricky) Why might ROC AUC be more trustworthy than accuracy here?**
It evaluates ranking across all thresholds and is less sensitive to the class ratio.

**Q21. (Tricky) Does 0.62 probability mean 62 out of 100 similar days will rise?**
Only if the model is calibrated and the distribution is unchanged. Neither is guaranteed.

**Q22. Where does the dashboard's accuracy number come from?**
It reads `lstm_meta['metrics']['accuracy']` from `models/lstm_metadata.json` (`app.py` line 320); it is not computed live.

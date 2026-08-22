# Part 1 -- Return-Risk Scoring Pipeline: Report

This document contains the written analysis required alongside Part 1's code
(Tasks 2, 4, 5, 7, and 8 of the brief). All numbers below are reproducible by
running the task scripts in order: `generate_orders.py` -> `task4_baseline.py`
-> `task5_logreg.py` -> `task6_random_forest.py` -> `task7_feature_importance.py`
-> `task8_subgroup_analysis.py` -> `task9_save_artifact.py`.

---

## Task 2 -- Dataset Verification

**Shape:** 6,000 rows x 13 columns.
**Overall return rate:** 22.75%.
**Missing `rating_given`:** 13.05% of rows.

**Return rate by `product_category`:**

| Category    | Return rate | Count |
|-------------|------------:|------:|
| Apparel     | 26.4%       | 1,979 |
| Footwear    | 26.0%       | 1,071 |
| Beauty      | 20.0%       | 579   |
| Home        | 19.2%       | 1,055 |
| Electronics | 18.7%       | 1,316 |

**Return rate by `payment_method`:**

| Method       | Return rate | Count |
|--------------|------------:|------:|
| COD          | 30.8%       | 2,501 |
| Wallet       | 17.9%       | 594   |
| Prepaid_UPI  | 16.9%       | 1,448 |
| Prepaid_Card | 16.8%       | 1,457 |

**Missingness classification: MAR (Missing At Random)**

The missingness in `rating_given` is **MAR**, conditioned on the observed
`payment_method` column -- not MCAR and not MNAR.

- **Not MCAR**, because the missing rate is not uniform across rows: COD
  orders are missing `rating_given` at **22.83%**, versus **6.06%** for all
  non-COD orders combined -- a **16.77 percentage-point gap**. If missingness
  were MCAR, these two rates would be roughly equal.
- **Not MNAR**, because the missingness probability is a function of
  `payment_method` (an observed column), not a function of the unobserved
  `rating_given` value itself. If it were MNAR, specifically low ratings
  would be more likely to be missing regardless of payment method -- that is
  not how the generator produced this column.

Practically, this makes sense: COD customers haven't pre-paid and went
through a lighter-commitment checkout flow, so they may be less likely to be
prompted for, or to bother leaving, a rating compared to prepaid customers.

---

## Task 4 -- Baseline (DummyClassifier)

- **Accuracy:** 0.7725
- **F1-score (class `returned=1`):** 0.0

**Why high accuracy is misleading here:** With a 22.75% positive rate, a
classifier that *always* predicts "not returned" gets ~77% accuracy for
free, without learning anything about which orders are actually at risk.
This is the **"high accuracy, zero recall" trap** -- accuracy rewards
matching the majority class, and on an imbalanced target it can look
strong while being completely useless for the actual business problem
(the whole point of this model is to *catch* returns before they happen,
which requires recall on the minority class, not overall accuracy). Any
real model must be judged against this baseline and against F1/recall/
precision on the class that actually matters, not raw accuracy.

---

## Task 5 -- Logistic Regression

**At default 0.5 threshold (class `returned=1`):**

| Metric    | Value  |
|-----------|-------:|
| Accuracy  | ~0.70  |
| F1        | 0.3921 |
| Recall    | ~0.33  |
| Precision | ~0.49  |
| ROC-AUC   | 0.6253 |

**Threshold sweep result:** F1-maximizing threshold = **0.44**, giving a
**+17.95 percentage-point recall improvement** over the default threshold,
at a precision cost of only **1.63 percentage points**.

**Business trade-off:** Lowering the decision threshold trades precision for
recall -- it flags more orders as "risky," catching more of the true
returns (fewer missed at-risk orders), but also raises more false alarms
(orders flagged as risky that would not actually have been returned). For
Flipkart's use case, missing a genuine at-risk order is likely more
expensive (unrecovered return/logistics cost, poor customer experience if
caught late) than a false alarm (which just means a support agent gives an
order slightly more scrutiny than necessary). This justifies accepting the
17.95pp recall gain even at the cost of some extra false positives.

---

## Task 7 -- Feature Importance

**Top-5 by impurity-based `.feature_importances_`:**
1. `payment_method_COD`
2. `price_inr`
3. `customer_tenure_days`
4. `delivery_distance_km`
5. `discount_pct`

**Interpretation:**
- `payment_method_COD` -- COD orders lack the upfront-payment commitment of
  prepaid orders, plausibly correlating with lower purchase intent and
  higher return likelihood (confirmed directly in the Task 2 return-rate
  table: COD returns at 30.8% vs. 16-18% for prepaid methods).
- `price_inr` -- higher-value items likely carry more scrutiny after
  arrival (fit, quality, buyer's remorse), raising return odds.
- `customer_tenure_days` -- newer customers may be less familiar with
  sizing/product expectations, or more willing to test-and-return.
- `delivery_distance_km` -- appears high by impurity, but see below: this
  is a case of impurity importance overrating a feature with no real
  signal.
- `discount_pct` -- heavily discounted items may attract more impulsive,
  lower-commitment purchases.

**Permutation importance comparison:** Recomputing importance via
`sklearn.inspection.permutation_importance` on the held-out test split
shows `delivery_distance_km` drops sharply -- from rank 4 (impurity) to
around rank 15, with a **near-zero / slightly negative** permutation
importance (-0.0024), meaning shuffling this feature barely hurts (or
doesn't hurt) test performance. `customer_tenure_days` and `discount_pct`
also drop substantially in rank under permutation.

**Why this happens:** Impurity-based `.feature_importances_` is biased
toward high-cardinality continuous columns (like a gamma-distributed
distance in km) because such features offer many more possible split
points for trees to exploit, regardless of whether those splits actually
generalize to unseen data -- permutation importance measures the real
performance drop and is not subject to this bias.

---

## Task 8 -- Subgroup Analysis

**By `payment_method`, the weakest subgroup is `Prepaid_Card`, with a
recall of 0.0** for the `returned=1` class -- the model misses every
actual return within this payment method on the test set, despite
reasonable overall recall (~0.51).

**Proposed fix:** Rather than "collect more data," a concrete next step is
to add a **`payment_method` x `num_previous_returns` interaction feature**
(or, more simply, fit a **payment-method-specific decision threshold**
instead of one global threshold). Since Task 5/6 showed threshold choice
has a large effect on recall, and Prepaid_Card orders' predicted
probabilities may simply cluster below whatever global threshold is
chosen, a per-segment threshold calibrated on Prepaid_Card orders
specifically would let the model catch its true positives in that segment
without needing new data collection.

---

## Task 9 -- Saved Artifact

- **Model:** Tuned Random Forest pipeline (`GridSearchCV`-selected:
  `n_estimators=100, max_depth=6`) from Task 6 -- not the Logistic
  Regression.
- **File:** `models/return_risk_model.pkl`
- **`t*_rf` (F1-maximizing threshold on the RF's own `predict_proba`,
  test split):** `0.46`
- **Risk buckets (anchored to `t*_rf`, not fixed values):**
  - Low: probability `< 0.46`
  - Medium: `0.46 <= probability < 0.61`
  - High: probability `>= 0.61`
- Reload-verified: `joblib.load()` on the saved file reproduces identical
  `predict_proba` output to the in-memory model.

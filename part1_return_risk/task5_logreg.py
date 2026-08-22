"""
Task 5: Logistic Regression with class_weight="balanced", default-threshold evaluation,
and a threshold sweep to find the F1-maximising cut point.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
)

from preprocessing import build_preprocessor, FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

logreg_pipeline = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])

logreg_pipeline.fit(X_train, y_train)

# --- Default threshold (0.5) evaluation ---
y_pred_default = logreg_pipeline.predict(X_test)
y_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

acc_default = accuracy_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default, pos_label=1)
recall_default = recall_score(y_test, y_pred_default, pos_label=1)
precision_default = precision_score(y_test, y_pred_default, pos_label=1)
roc_auc = roc_auc_score(y_test, y_proba)

print("=== Logistic Regression @ default threshold (0.5) ===")
print(f"Accuracy:  {acc_default:.4f}")
print(f"F1:        {f1_default:.4f}")
print(f"Recall:    {recall_default:.4f}")
print(f"Precision: {precision_default:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# --- Threshold sweep ---
thresholds = np.arange(0.10, 0.90 + 1e-9, 0.02)
results = []
for t in thresholds:
    preds_t = (y_proba >= t).astype(int)
    f1_t = f1_score(y_test, preds_t, pos_label=1, zero_division=0)
    recall_t = recall_score(y_test, preds_t, pos_label=1, zero_division=0)
    precision_t = precision_score(y_test, preds_t, pos_label=1, zero_division=0)
    results.append({"threshold": round(t, 2), "f1": f1_t, "recall": recall_t, "precision": precision_t})

sweep_df = pd.DataFrame(results)
best_row = sweep_df.loc[sweep_df["f1"].idxmax()]

print("\n=== Threshold sweep (F1-maximising row) ===")
print(best_row)

print("\n=== Full sweep table ===")
print(sweep_df.to_string(index=False))

recall_gain_pp = (best_row["recall"] - recall_default) * 100
precision_drop_pp = (precision_default - best_row["precision"]) * 100

print(f"\nRecall gain vs default threshold: {recall_gain_pp:.2f} percentage points")
print(f"Precision drop vs default threshold: {precision_drop_pp:.2f} percentage points")

sweep_df.to_csv("logreg_threshold_sweep.csv", index=False)
print("\nSaved sweep table to logreg_threshold_sweep.csv")

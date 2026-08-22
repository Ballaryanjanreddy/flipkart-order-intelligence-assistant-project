"""
Task 9: Save the final chosen artifact (tuned Random Forest pipeline from Task 6) and
compute t*_rf -- the F1-maximising threshold on THIS model's own predict_proba output
on the test split (re-running Task 5's sweep procedure, but on the Random Forest,
not the Logistic Regression).

This t*_rf value is what Part 3's check_return_risk tool anchors its Low/Medium/High
risk buckets to.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

from preprocessing import FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Load the winning pipeline produced by Task 6's GridSearchCV
best_model = joblib.load("rf_best_model_temp.pkl")

# --- Re-run the threshold sweep, but on the Random Forest's own predict_proba ---
y_proba_rf = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.10, 0.90 + 1e-9, 0.02)
sweep_rows = []
for t in thresholds:
    preds_t = (y_proba_rf >= t).astype(int)
    f1_t = f1_score(y_test, preds_t, pos_label=1, zero_division=0)
    recall_t = recall_score(y_test, preds_t, pos_label=1, zero_division=0)
    precision_t = precision_score(y_test, preds_t, pos_label=1, zero_division=0)
    sweep_rows.append({"threshold": round(t, 2), "f1": f1_t, "recall": recall_t, "precision": precision_t})

rf_sweep_df = pd.DataFrame(sweep_rows)
best_rf_row = rf_sweep_df.loc[rf_sweep_df["f1"].idxmax()]

t_star_rf = float(best_rf_row["threshold"])

print("=== Random Forest threshold sweep (for t*_rf) ===")
print(rf_sweep_df.to_string(index=False))
print(f"\nt*_rf (F1-maximising threshold on RF's own predict_proba): {t_star_rf}")
print(f"At t*_rf -> F1: {best_rf_row['f1']:.4f}, Recall: {best_rf_row['recall']:.4f}, Precision: {best_rf_row['precision']:.4f}")

rf_sweep_df.to_csv("rf_threshold_sweep.csv", index=False)

# --- Define the risk buckets, anchored to t*_rf, exactly as Part 3's tool will use them ---
bucket_low_cut = t_star_rf
bucket_high_cut = t_star_rf + 0.15

print(f"\n=== Risk bucket cut points (anchored to t*_rf = {t_star_rf}) ===")
print(f"Low:    probability < {bucket_low_cut}")
print(f"Medium: {bucket_low_cut} <= probability < {bucket_high_cut}")
print(f"High:   probability >= {bucket_high_cut}")

# --- Save the final artifact ---
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/return_risk_model.pkl")
print("\nSaved final artifact to models/return_risk_model.pkl")

# Save t*_rf and bucket cut points to a small config file so Part 3's tool can load them
# without re-deriving them (avoids any drift between what Part 1 reports and what Part 3 uses).
config = {
    "t_star_rf": t_star_rf,
    "bucket_low_cut": bucket_low_cut,
    "bucket_high_cut": bucket_high_cut,
    "f1_at_t_star_rf": float(best_rf_row["f1"]),
    "recall_at_t_star_rf": float(best_rf_row["recall"]),
    "precision_at_t_star_rf": float(best_rf_row["precision"]),
}
import json
with open("models/risk_threshold_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved threshold config to models/risk_threshold_config.json")

# --- Sanity check: reload from disk and confirm predict_proba matches ---
reloaded = joblib.load("models/return_risk_model.pkl")
reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
assert np.allclose(reloaded_proba, y_proba_rf), "Reloaded model's predict_proba does not match!"
print("\nSanity check passed: reloaded model's predict_proba matches the original exactly.")

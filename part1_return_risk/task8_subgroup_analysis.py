"""
Task 8: Subgroup / root-cause analysis -- recall and precision by product_category
and by payment_method, for the winning Random Forest on the held-out test set.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

from preprocessing import FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

best_model = joblib.load("rf_best_model_temp.pkl")
y_pred = best_model.predict(X_test)

results_df = X_test.copy()
results_df["y_true"] = y_test.values
results_df["y_pred"] = y_pred

def subgroup_metrics(df_slice, group_col):
    rows = []
    for group_val, sub in df_slice.groupby(group_col):
        n = len(sub)
        n_positive = sub["y_true"].sum()
        if n_positive == 0:
            recall = float("nan")
        else:
            recall = recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        precision = precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        f1 = f1_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        rows.append({
            group_col: group_val, "n_orders": n, "n_actual_returns": int(n_positive),
            "recall": round(recall, 4) if recall == recall else None,
            "precision": round(precision, 4), "f1": round(f1, 4),
        })
    return pd.DataFrame(rows).sort_values(group_col)

overall_recall = recall_score(results_df["y_true"], results_df["y_pred"], pos_label=1)
overall_precision = precision_score(results_df["y_true"], results_df["y_pred"], pos_label=1)
print(f"=== Overall test-set recall: {overall_recall:.4f} | precision: {overall_precision:.4f} ===\n")

cat_table = subgroup_metrics(results_df, "product_category")
print("=== By product_category ===")
print(cat_table.to_string(index=False))

pay_table = subgroup_metrics(results_df, "payment_method")
print("\n=== By payment_method ===")
print(pay_table.to_string(index=False))

cat_table.to_csv("subgroup_by_category.csv", index=False)
pay_table.to_csv("subgroup_by_payment.csv", index=False)
print("\nSaved: subgroup_by_category.csv, subgroup_by_payment.csv")

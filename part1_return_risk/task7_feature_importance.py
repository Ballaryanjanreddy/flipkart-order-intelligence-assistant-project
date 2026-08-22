"""
Task 7: Feature importance -- impurity-based vs permutation-based, compared side by side.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from preprocessing import FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Load the winning pipeline from Task 6
best_model = joblib.load("rf_best_model_temp.pkl")

# --- Impurity-based feature importances ---
preprocessor = best_model.named_steps["preprocess"]
rf_clf = best_model.named_steps["clf"]

feature_names = preprocessor.get_feature_names_out()
importances = rf_clf.feature_importances_

impurity_df = pd.DataFrame({
    "feature": feature_names,
    "impurity_importance": importances,
}).sort_values("impurity_importance", ascending=False).reset_index(drop=True)

print("=== Impurity-based feature importances (.feature_importances_) -- top 10 ===")
print(impurity_df.head(10).to_string(index=False))

top5_impurity = impurity_df.head(5)["feature"].tolist()
print(f"\nTop 5 (impurity-based): {top5_impurity}")

# --- Permutation importance on the held-out test set ---
# Note: we run permutation_importance on the WHOLE pipeline (preprocess + clf) against the
# RAW X_test (pre-transformation) so that shuffling happens on the original, human-readable
# feature columns -- not on the one-hot-encoded matrix -- keeping the comparison apples-to-apples
# with the impurity_df feature names as closely as possible. Since one-hot columns come from
# categorical originals, we compute permutation importance on the transformed matrix instead,
# using the already-fitted preprocessor, so importances land on the exact same one-hot columns
# as the impurity-based ranking above.
X_test_transformed = preprocessor.transform(X_test)

perm_result = permutation_importance(
    rf_clf, X_test_transformed, y_test,
    n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": feature_names,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
}).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

print("\n=== Permutation importance (test split, roc_auc scoring) -- top 10 ===")
print(perm_df.head(10).to_string(index=False))

# --- Side-by-side comparison for the original top-5 impurity features ---
print("\n=== Side-by-side comparison for original top-5 impurity-based features ===")
comparison_rows = []
for feat in top5_impurity:
    impurity_rank = impurity_df.index[impurity_df["feature"] == feat][0] + 1
    impurity_val = impurity_df.loc[impurity_df["feature"] == feat, "impurity_importance"].values[0]
    perm_rank = perm_df.index[perm_df["feature"] == feat][0] + 1
    perm_val = perm_df.loc[perm_df["feature"] == feat, "perm_importance_mean"].values[0]
    comparison_rows.append({
        "feature": feat,
        "impurity_rank": impurity_rank, "impurity_importance": round(impurity_val, 4),
        "permutation_rank": perm_rank, "permutation_importance": round(perm_val, 4),
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

comparison_df.to_csv("feature_importance_comparison.csv", index=False)
impurity_df.to_csv("feature_importance_impurity_full.csv", index=False)
perm_df.to_csv("feature_importance_permutation_full.csv", index=False)
print("\nSaved: feature_importance_comparison.csv, feature_importance_impurity_full.csv, feature_importance_permutation_full.csv")

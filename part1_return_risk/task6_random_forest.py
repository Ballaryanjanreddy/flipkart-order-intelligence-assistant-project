"""
Task 6: Random Forest with class_weight="balanced", tuned via GridSearchCV
(5-fold StratifiedKFold, scored on roc_auc).
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

from preprocessing import build_preprocessor, FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")
X = df[FEATURE_COLUMNS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

rf_pipeline = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [6, 10, None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

print("Running GridSearchCV (5-fold, 6 param combos = 30 fits)...")
grid_search.fit(X_train, y_train)

print("\n=== GridSearchCV results ===")
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Evaluate the winning model on the held-out test set
best_model = grid_search.best_estimator_
y_proba_test = best_model.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_proba_test)

print(f"Held-out test ROC-AUC: {test_roc_auc:.4f}")
print(f"Gap (CV - test): {abs(grid_search.best_score_ - test_roc_auc):.4f}")

# Save full CV results table for the record
cv_results_df = pd.DataFrame(grid_search.cv_results_)[
    ["param_clf__n_estimators", "param_clf__max_depth", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")
cv_results_df.to_csv("rf_gridsearch_results.csv", index=False)
print("\nFull grid results (sorted by rank):")
print(cv_results_df.to_string(index=False))

# Save the winning fitted pipeline -- this IS the artifact Task 9 will persist,
# but we save it here too so downstream Task 7/8 scripts can just load it instead
# of re-running the grid search every time.
joblib.dump(best_model, "rf_best_model_temp.pkl")
print("\nSaved winning pipeline to rf_best_model_temp.pkl (temp, Task 9 will finalize this as models/return_risk_model.pkl)")

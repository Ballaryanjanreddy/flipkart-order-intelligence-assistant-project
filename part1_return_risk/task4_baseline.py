"""
Task 4: DummyClassifier baseline.

Purpose: establish the floor every real model must beat. A model that can't outperform
"always predict the majority class" isn't learning anything about return risk.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing import build_preprocessor, FEATURE_COLUMNS, TARGET

RANDOM_STATE = 42

df = pd.read_csv("orders_dataset.csv")

X = df[FEATURE_COLUMNS]
y = df[TARGET]

# Stratified split so both train and test preserve the ~22.75% return rate.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train return rate: {y_train.mean():.4f} | Test return rate: {y_test.mean():.4f}")

# Pipeline: preprocessing + dummy classifier. Fitting the whole pipeline on X_train only
# means the imputer/scaler/encoder are fit on training data only, exactly as Task 3 requires.
dummy_pipeline = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
])

dummy_pipeline.fit(X_train, y_train)
y_pred_dummy = dummy_pipeline.predict(X_test)

dummy_acc = accuracy_score(y_test, y_pred_dummy)
dummy_f1 = f1_score(y_test, y_pred_dummy, pos_label=1)

print("\n=== DummyClassifier (most_frequent) baseline ===")
print(f"Accuracy: {dummy_acc:.4f}")
print(f"F1 (returned=1): {dummy_f1:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred_dummy, target_names=["not_returned", "returned"]))

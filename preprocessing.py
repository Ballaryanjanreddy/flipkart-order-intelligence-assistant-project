# """
# Task 3: Leakage-free preprocessing pipeline for the Flipkart return-risk dataset.

# Design notes:
# - Numeric features: median imputation (robust to the skewed distributions we generated,
#   e.g. delivery_distance_km via gamma, customer_tenure_days via exponential) + standard scaling.
# - Categorical features: mode imputation + one-hot encoding.
# - rating_given is numeric but MAR-missing (see Task 2 analysis) -- median imputation is a
#   reasonable, simple choice here since we are not trying to model the missingness mechanism
#   itself, just prevent NaNs from breaking the estimator.
# - The ColumnTransformer + Pipeline combo means we NEVER call .fit() or .fit_transform() on
#   test data anywhere in this project -- only .transform(). Fitting happens exclusively on
#   the training split, inside cross-validation folds, or inside GridSearchCV's internal folds.
# """

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# NUMERIC_FEATURES = [
#     "price_inr",
#     "discount_pct",
#     "customer_tenure_days",
#     "num_previous_orders",
#     "num_previous_returns",
#     "delivery_distance_km",
#     "delivery_days",
#     "is_weekend_order",
#     "rating_given",
# ]

# CATEGORICAL_FEATURES = [
#     "product_category",
#     "payment_method",
# ]

# TARGET = "returned"

# # Columns the model is actually allowed to see. order_id is an identifier, not a feature,
# # and must never be fed to the model (it carries no real signal and including it would be
# # a subtle leakage/overfitting risk since it's just a row index).
# FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# def build_preprocessor() -> ColumnTransformer:
#     """Returns an unfitted ColumnTransformer. Caller is responsible for fitting
#     it only on training data (typically inside a Pipeline alongside an estimator,
#     so GridSearchCV/cross_val_score handle the fit/transform split correctly per fold)."""

#     numeric_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore")),
#     ])

#     preprocessor = ColumnTransformer(transformers=[
#         ("num", numeric_transformer, NUMERIC_FEATURES),
#         ("cat", categorical_transformer, CATEGORICAL_FEATURES),
#     ])

#     return preprocessor


# def get_feature_names(preprocessor: ColumnTransformer):
#     """Utility to recover human-readable feature names after the ColumnTransformer
#     has been fitted (needed later for feature_importances_ / permutation_importance
#     reporting in Task 7)."""
#     return preprocessor.get_feature_names_out()



"""
Task 3: Leakage-free preprocessing pipeline for the Flipkart return-risk dataset.

Design notes:
- Numeric features: median imputation (robust to the skewed distributions we generated,
  e.g. delivery_distance_km via gamma, customer_tenure_days via exponential) + standard scaling.
- Categorical features: mode imputation + one-hot encoding.
- rating_given is numeric but MAR-missing (see Task 2 analysis) -- median imputation is a
  reasonable, simple choice here since we are not trying to model the missingness mechanism
  itself, just prevent NaNs from breaking the estimator.
- The ColumnTransformer + Pipeline combo means we NEVER call .fit() or .fit_transform() on
  test data anywhere in this project -- only .transform(). Fitting happens exclusively on
  the training split, inside cross-validation folds, or inside GridSearchCV's internal folds.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

CATEGORICAL_FEATURES = [
    "product_category",
    "payment_method",
]

TARGET = "returned"

# Columns the model is actually allowed to see. order_id is an identifier, not a feature,
# and must never be fed to the model (it carries no real signal and including it would be
# a subtle leakage/overfitting risk since it's just a row index).
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer. Caller is responsible for fitting
    it only on training data (typically inside a Pipeline alongside an estimator,
    so GridSearchCV/cross_val_score handle the fit/transform split correctly per fold)."""

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer):
    """Utility to recover human-readable feature names after the ColumnTransformer
    has been fitted (needed later for feature_importances_ / permutation_importance
    reporting in Task 7)."""
    return preprocessor.get_feature_names_out()


if __name__ == "__main__":
    # Self-test: proves the preprocessor fits on train only and transforms both
    # splits without error. Run directly with `py preprocessing.py` to sanity-check
    # this module in isolation before any task script imports it.
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("orders_dataset.csv")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = build_preprocessor()
    pre.fit(X_train)          # fit on TRAIN ONLY
    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)   # transform only, never fit, on test

    print(f"Train: {X_train.shape} -> {X_train_t.shape}")
    print(f"Test:  {X_test.shape} -> {X_test_t.shape}")
    print(f"Train return rate: {y_train.mean():.4f} | Test return rate: {y_test.mean():.4f}")
    print("\nFeature names after preprocessing:")
    print(list(get_feature_names(pre)))
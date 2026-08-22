"""
Tasks 3 & 4 -- check_return_risk and classify_product_image.

Both tools load the REAL saved artifacts from Part 1 and Part 2 -- nothing
here is a hardcoded stand-in. Adjust PART1_DIR / PART2_DIR below if your
folder layout differs from the sibling-directory structure used here
(part1_return_risk/, part2_image_classifier/, part3_agent/ all under the
same repo root).
"""

import os
import sys
import json
import joblib
import pandas as pd

# --- Resolve sibling-part paths -------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
PART1_DIR = os.path.join(REPO_ROOT, "part1_return_risk")
PART2_DIR = os.path.join(REPO_ROOT, "part2_image_classifier")

sys.path.insert(0, PART2_DIR)  # so we can import model_utils / task7_save_artifact from Part 2


# ============================================================================
# Task 3 -- check_return_risk
# ============================================================================

_RF_MODEL = None
_THRESHOLD_CONFIG = None


def _load_return_risk_artifacts():
    global _RF_MODEL, _THRESHOLD_CONFIG
    if _RF_MODEL is None:
        model_path = os.path.join(PART1_DIR, "models", "return_risk_model.pkl")
        config_path = os.path.join(PART1_DIR, "models", "risk_threshold_config.json")
        _RF_MODEL = joblib.load(model_path)
        with open(config_path) as f:
            _THRESHOLD_CONFIG = json.load(f)
    return _RF_MODEL, _THRESHOLD_CONFIG


def check_return_risk(order_features: dict) -> dict:
    """Loads Part 1's tuned Random Forest pipeline and returns a predicted
    return probability + risk bucket. Bucket cut points are anchored to
    t*_rf (the F1-maximizing threshold computed on the Random Forest's OWN
    predict_proba output in Part 1 Task 9) -- NOT fixed values, since a
    fixed 0.3/0.6 split isn't guaranteed to mean anything for this
    specific model's probability distribution.

    order_features must contain exactly the raw columns Part 1's
    preprocessing pipeline expects: price_inr, discount_pct,
    customer_tenure_days, num_previous_orders, num_previous_returns,
    delivery_distance_km, delivery_days, is_weekend_order, rating_given,
    product_category, payment_method.
    """
    model, config = _load_return_risk_artifacts()
    t_star = config["t_star_rf"]

    required_cols = [
        "price_inr", "discount_pct", "customer_tenure_days",
        "num_previous_orders", "num_previous_returns",
        "delivery_distance_km", "delivery_days", "is_weekend_order",
        "rating_given", "product_category", "payment_method",
    ]
    missing = [c for c in required_cols if c not in order_features]
    if missing:
        raise ValueError(f"order_features missing required columns: {missing}")

    X = pd.DataFrame([order_features])[required_cols]
    probability = float(model.predict_proba(X)[0, 1])

    # Bucket cut points anchored to t*_rf, per Part 1 Task 9 / Part 3 Task 3.
    low_cut = t_star
    high_cut = t_star + 0.15
    if probability < low_cut:
        bucket = "Low"
    elif probability >= high_cut:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(probability, 4),
        "risk_bucket": bucket,
        "t_star_rf": t_star,
        "cut_points": {"low_below": low_cut, "high_at_or_above": high_cut},
    }


# ============================================================================
# Task 4 -- classify_product_image
# ============================================================================

_IMAGE_MODEL = None
_IMAGE_DEVICE = None


def _load_image_model():
    global _IMAGE_MODEL, _IMAGE_DEVICE
    if _IMAGE_MODEL is None:
        # Imports from Part 2's own module, so we reuse the EXACT same
        # architecture-reconstruction logic used when the model was trained
        # and saved -- no duplicated/divergent model-building code.
        from task7_save_artifact import load_model
        from model_utils import get_device

        _IMAGE_DEVICE = get_device()
        artifact_path = os.path.join(PART2_DIR, "models", "product_classifier.pt")
        _IMAGE_MODEL = load_model(artifact_path=artifact_path, device=_IMAGE_DEVICE)
    return _IMAGE_MODEL, _IMAGE_DEVICE


def classify_product_image(image_path: str) -> dict:
    """Loads Part 2's saved classifier and returns the predicted category
    label + confidence for a real .png file (expected to be one of the
    files exported to part2_image_classifier/data/sample_images/ in Part 2
    Task 8). Accepts an absolute path or a path relative to part3_agent/.
    """
    from task7_save_artifact import predict_image  # Part 2's own prediction function

    model, device = _load_image_model()

    if not os.path.isabs(image_path):
        # try relative to part3_agent first, then relative to part2's sample dir
        candidate = os.path.join(THIS_DIR, image_path)
        if not os.path.exists(candidate):
            candidate = os.path.join(PART2_DIR, image_path)
        image_path = candidate

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = predict_image(image_path, model=model, device=device)
    return {
        "predicted_category": result["predicted_category"],
        "confidence": round(result["confidence"], 4),
    }


if __name__ == "__main__":
    # Self-test (run this AFTER Part 1 and Part 2 are both fully trained
    # and their artifacts saved).
    print("--- check_return_risk self-test ---")
    example_order = {
        "price_inr": 1800, "discount_pct": 35, "customer_tenure_days": 40,
        "num_previous_orders": 2, "num_previous_returns": 1,
        "delivery_distance_km": 300, "delivery_days": 5, "is_weekend_order": 1,
        "rating_given": None, "product_category": "Apparel", "payment_method": "COD",
    }
    print(check_return_risk(example_order))

    print("\n--- classify_product_image self-test ---")
    sample_dir = os.path.join(PART2_DIR, "data", "sample_images")
    if os.path.isdir(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
        if samples:
            print(classify_product_image(os.path.join(sample_dir, samples[0])))
        else:
            print("No sample images found -- run Part 2 Task 8 first.")

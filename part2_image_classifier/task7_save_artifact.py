"""
Task 7 -- Save the final artifact.

Persists the best available checkpoint (fine-tuned if it exists, otherwise
feature-extraction-only) to models/product_classifier.pt as a clean, final
artifact name -- distinct from the internal `_checkpoint_*.pt` working
files task3/task4 produce. Also provides `load_model()` and
`predict_image()`, the exact functions Part 3's `classify_product_image`
tool imports and calls.
"""

import os
import torch
from PIL import Image

from model_utils import build_model, unfreeze_layer4, get_transforms, get_device, CLASS_NAMES

FINETUNED_CKPT = "models/_checkpoint_finetuned.pt"
FEATURE_ONLY_CKPT = "models/_checkpoint_feature_extraction.pt"
FINAL_ARTIFACT = "models/product_classifier.pt"


def save_final_artifact():
    """Copies whichever checkpoint is best (fine-tuned > feature-extraction-only)
    to the final artifact path, plus a small metadata sidecar recording which
    architecture state (whether layer4 was unfrozen) it corresponds to --
    load_model() needs this to reconstruct the exact same architecture."""
    if os.path.exists(FINETUNED_CKPT):
        state_dict = torch.load(FINETUNED_CKPT, map_location="cpu")
        layer4_unfrozen = True
        source = "fine-tuned"
    elif os.path.exists(FEATURE_ONLY_CKPT):
        state_dict = torch.load(FEATURE_ONLY_CKPT, map_location="cpu")
        layer4_unfrozen = False
        source = "feature-extraction-only"
    else:
        raise FileNotFoundError(
            "No checkpoint found -- run task3_feature_extraction.py "
            "(and task4_finetune.py if needed) first."
        )

    os.makedirs("models", exist_ok=True)
    torch.save({
        "state_dict": state_dict,
        "layer4_unfrozen": layer4_unfrozen,
        "class_names": CLASS_NAMES,
    }, FINAL_ARTIFACT)
    print(f"Saved final artifact ({source}) to {FINAL_ARTIFACT}")


def load_model(artifact_path: str = FINAL_ARTIFACT, device=None):
    """Loads the saved artifact and reconstructs the exact model architecture
    it was trained with. This is the function Part 3's classify_product_image
    tool calls once at startup."""
    device = device or get_device()
    checkpoint = torch.load(artifact_path, map_location=device)

    model = build_model(num_classes=10, pretrained=False)  # weights come from checkpoint
    if checkpoint["layer4_unfrozen"]:
        model = unfreeze_layer4(model)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_image(image_path: str, model=None, device=None) -> dict:
    """Single-image prediction. This exact function is what Part 3's
    classify_product_image(image_path) tool wraps.

    Returns: {"predicted_category": str, "confidence": float, "all_probabilities": dict}
    """
    device = device or get_device()
    if model is None:
        model = load_model(device=device)

    transform = get_transforms()
    image = Image.open(image_path).convert("L")  # ensure grayscale in, like Fashion-MNIST
    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(probs.argmax())
    return {
        "predicted_category": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


if __name__ == "__main__":
    save_final_artifact()

    # Quick self-test if sample images already exist (run task8 first for this
    # to have something to test against).
    sample_dir = "data/sample_images"
    if os.path.isdir(sample_dir):
        samples = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
        if samples:
            device = get_device()
            model = load_model(device=device)
            test_file = os.path.join(sample_dir, samples[0])
            result = predict_image(test_file, model=model, device=device)
            print(f"\nSelf-test on {test_file}:")
            print(f"  Predicted: {result['predicted_category']} "
                  f"(confidence {result['confidence']:.4f})")

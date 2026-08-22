"""
Task 8 -- Export real sample images as actual .png files.

torchvision.datasets.FashionMNIST stores data as raw IDX binary, not
individual image files. Part 3's classify_product_image(image_path) tool
needs real file paths to point at, so this script picks real test-split
images (one per class, for full coverage) and writes each out as a
labeled .png file via PIL.Image.fromarray().
"""

import os
import numpy as np
from PIL import Image

from task1_load_data import load_splits
from model_utils import CLASS_NAMES

OUT_DIR = "data/sample_images"


def export_samples():
    _, _, _, test_set = load_splits()

    os.makedirs(OUT_DIR, exist_ok=True)

    # One real sample per class (10 total, comfortably over the "at least 5"
    # requirement, and gives Part 3 full category coverage to demo against).
    labels = np.array(test_set.targets)
    exported = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # first test-split index matching this class
        idx = int(np.where(labels == class_idx)[0][0])
        image, label = test_set[idx]  # PIL Image, int -- raw, un-preprocessed
        assert label == class_idx

        safe_name = class_name.replace("/", "-").replace(" ", "_").lower()
        filename = f"{class_idx:02d}_{safe_name}.png"
        path = os.path.join(OUT_DIR, filename)
        image.save(path)  # raw 28x28 grayscale PNG, real data, not simulated
        exported.append((path, class_name))
        print(f"Exported {path}  (true label: {class_name})")

    print(f"\n{len(exported)} real sample images exported to {OUT_DIR}/")
    return exported


if __name__ == "__main__":
    export_samples()

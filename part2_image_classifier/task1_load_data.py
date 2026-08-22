"""
Task 1 -- Load Fashion-MNIST and carve a stratified validation split.

Standard split: 60,000 train images / 10,000 test images (from torchvision's
built-in train=True/False flag). We carve a stratified validation split of
at least 5,000 images out of the training set for model selection, leaving
the 10,000-image test split completely untouched until final evaluation
(Task 5).

Run directly with `py task1_load_data.py` to download the dataset (first run
only -- it's cached locally after that) and confirm split sizes.
"""

import numpy as np
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split

DATA_ROOT = "./data/fashion_mnist"
VAL_SIZE = 5000       # >= 5,000 as required by the brief
RANDOM_STATE = 42


def load_splits():
    """Returns (train_dataset, val_indices, test_dataset) where val_indices
    indexes into train_dataset. Downloads Fashion-MNIST to DATA_ROOT if not
    already cached there (no login/API key required -- pulled from the
    canonical zalandoresearch/fashion-mnist source via torchvision)."""

    full_train = FashionMNIST(root=DATA_ROOT, train=True, download=True)
    test_set = FashionMNIST(root=DATA_ROOT, train=False, download=True)

    # Stratified split of the 60,000 training images: carve out VAL_SIZE
    # images for validation, stratified on the label so all 10 classes stay
    # proportionally represented in both the training-subset and validation set.
    labels = np.array(full_train.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=VAL_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    return full_train, train_idx, val_idx, test_set


if __name__ == "__main__":
    full_train, train_idx, val_idx, test_set = load_splits()
    print(f"Full training pool: {len(full_train)} images")
    print(f"  -> train subset:      {len(train_idx)} images")
    print(f"  -> validation subset: {len(val_idx)} images")
    print(f"Test split (untouched until Task 5): {len(test_set)} images")

    # Sanity check: confirm stratification kept class balance close to uniform
    # (Fashion-MNIST is exactly balanced -- 6,000 images/class in the full
    # 60k training pool -- so both subsets should be close to 10% per class).
    labels = np.array(full_train.targets)
    print("\nClass distribution in validation subset:")
    for c in range(10):
        pct = (labels[val_idx] == c).mean() * 100
        print(f"  class {c}: {pct:.1f}%")

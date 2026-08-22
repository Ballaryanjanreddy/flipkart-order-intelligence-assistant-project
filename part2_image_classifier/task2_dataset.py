"""
Task 2 -- Preprocessing for a pretrained backbone.

Fashion-MNIST images are 28x28 single-channel (grayscale). ResNet-18 expects
3-channel 224x224 input normalized with ImageNet statistics. This module
wraps the raw torchvision dataset with that exact transform pipeline
(defined once, in model_utils.get_transforms(), so training/val/test/Part3
all use identical preprocessing).
"""

from torch.utils.data import Dataset
from model_utils import get_transforms


class FashionMNISTSubset(Dataset):
    """Wraps a torchvision FashionMNIST dataset (+ optional index subset)
    and applies the ImageNet-style transform pipeline on __getitem__."""

    def __init__(self, base_dataset, indices=None, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else range(len(base_dataset))
        self.transform = transform if transform is not None else get_transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        image, label = self.base_dataset[real_idx]  # PIL Image (mode 'L'), int
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # Self-test: confirm the transform pipeline produces the expected shape.
    from task1_load_data import load_splits

    full_train, train_idx, val_idx, test_set = load_splits()
    wrapped = FashionMNISTSubset(full_train, train_idx[:5])
    img, label = wrapped[0]
    print(f"Transformed image shape: {tuple(img.shape)}  (expect: (3, 224, 224))")
    print(f"Label: {label} -> {label}")
    print(f"Pixel value range after normalization: [{img.min():.3f}, {img.max():.3f}]")

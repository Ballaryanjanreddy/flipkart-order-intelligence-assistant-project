"""
Shared utilities for Part 2 -- Product Image Categoriser via Transfer Learning.

Every task script imports from here so the class list, preprocessing
transforms, and model architecture are guaranteed identical across
training, evaluation, and the final saved-artifact loading snippet used
by Part 3's classify_product_image tool.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

# Fashion-MNIST's 10 classes, in the dataset's canonical label order (0-9).
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ResNet-18 was pretrained on ImageNet at 224x224 with these exact
# normalization stats. We replicate the single Fashion-MNIST grayscale
# channel to 3 channels and resize to match what the backbone expects.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224  # documented input size for ResNet-18


def get_transforms():
    """Preprocessing pipeline (Task 2): grayscale -> 3-channel, resize to
    224x224, normalize with ImageNet mean/std."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Task 3: ResNet-18 backbone with early/middle layers frozen and a
    new classifier head for 10 classes.

    Freezing strategy: ALL backbone layers (early, middle, and late --
    including layer4) are frozen initially, and only the new `fc` head is
    trainable. This is the "feature extraction" stage (Task 3). If Task 4's
    validation accuracy check shows fine-tuning is needed, call
    `unfreeze_layer4()` afterward to unfreeze just the last residual block
    for gradual fine-tuning, per the brief's guidance to keep early/middle
    layers frozen even during fine-tuning.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head -- this is always trainable.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layer4(model: nn.Module):
    """Task 4: gradual unfreezing. Only unfreezes the backbone's last
    residual block (layer4), keeping earlier (more generic) layers frozen,
    per the standard fine-tuning strategy."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

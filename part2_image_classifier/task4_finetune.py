"""
Task 4 -- Fine-tune if needed.

Only run this AFTER task3_feature_extraction.py. If that script's reported
validation accuracy was already >= 80%, you can skip this step entirely --
just document "feature extraction alone was sufficient" with the val_acc
number from Task 3's output.

If accuracy was below 80%, this script unfreezes layer4 (the backbone's
last residual block -- early/middle layers stay frozen) and continues
training end-to-end at a lower learning rate, per the standard gradual-
unfreezing fine-tuning strategy. Because layer4 is now trainable, we can no
longer use Task 3's cached features (those were computed by the fully-frozen
backbone) -- we go back to running real images through the full network.

Hyperparameters:
    Optimizer:          Adam
    Backbone LR:        1e-5  (small -- avoid destroying pretrained weights)
    Head LR:            1e-4
    Batch size:         32   (smaller than Task 3 since full forward+backward
                               through the backbone is heavier than a linear head)
    Epochs:             5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_utils import build_model, unfreeze_layer4, get_device
from task1_load_data import load_splits
from task2_dataset import FashionMNISTSubset

BATCH_SIZE = 32
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
EPOCHS = 5
CHECKPOINT_IN = "models/_checkpoint_feature_extraction.pt"
CHECKPOINT_OUT = "models/_checkpoint_finetuned.pt"


def run_finetune():
    device = get_device()
    print(f"Device: {device}")

    full_train, train_idx, val_idx, test_set = load_splits()
    train_ds = FashionMNISTSubset(full_train, train_idx)
    val_ds = FashionMNISTSubset(full_train, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_classes=10, pretrained=True)
    model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=device))
    model = unfreeze_layer4(model).to(device)

    # Report before-fine-tune val accuracy for the "before/after" comparison
    # the brief requires.
    before_acc = evaluate(model, val_loader, device)
    print(f"Validation accuracy BEFORE fine-tuning: {before_acc:.4f}")

    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
        {"params": model.fc.parameters(), "lr": HEAD_LR},
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={total_loss / n_seen:.4f} | val_acc={val_acc:.4f}")

    after_acc = evaluate(model, val_loader, device)
    print(f"\nValidation accuracy AFTER fine-tuning: {after_acc:.4f}")
    print(f"Improvement: {after_acc - before_acc:+.4f}")

    torch.save(model.state_dict(), CHECKPOINT_OUT)
    print(f"Saved fine-tuned checkpoint: {CHECKPOINT_OUT}")
    return before_acc, after_acc


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    run_finetune()

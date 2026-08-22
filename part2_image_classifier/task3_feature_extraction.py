"""
Task 3 -- Build the transfer-learning model and train the head via cached
frozen-backbone features (the brief's CPU speed trick).

Since the ResNet-18 backbone is entirely frozen at this stage, running it
over every image every epoch is wasted work -- the backbone's output for a
given image never changes. Instead we run the frozen backbone ONCE over
every image, cache the resulting 512-dim feature vectors to disk, and then
train only the small linear head on those cached vectors. This is
mathematically identical to re-running the frozen backbone each epoch, but
turns an hours-long CPU loop into a few-minutes feature pass + a
near-instant head-training step.

Hyperparameters (documented per the brief's requirement):
    Optimizer:      Adam
    Learning rate:  1e-3 (head only -- backbone is frozen, no LR needed for it)
    Batch size:     64
    Epochs:         15 (head-only training converges fast on cached features)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_utils import build_model, get_device, CLASS_NAMES
from task1_load_data import load_splits
from task2_dataset import FashionMNISTSubset

CACHE_DIR = "./data/feature_cache"
BATCH_SIZE = 64
HEAD_LR = 1e-3
HEAD_EPOCHS = 15


def extract_and_cache_features(model, loader, device, split_name):
    """Runs the frozen backbone (everything except fc) over a loader once,
    caches the resulting feature vectors + labels to CACHE_DIR."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    feat_path = os.path.join(CACHE_DIR, f"{split_name}_features.pt")
    label_path = os.path.join(CACHE_DIR, f"{split_name}_labels.pt")

    if os.path.exists(feat_path) and os.path.exists(label_path):
        print(f"[{split_name}] cached features found, skipping extraction.")
        return torch.load(feat_path), torch.load(label_path)

    # Temporarily swap fc for Identity so forward() returns the 512-dim
    # pooled feature vector instead of class logits.
    original_fc = model.fc
    model.fc = nn.Identity()
    model.eval()

    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)  # (B, 512)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    model.fc = original_fc  # restore

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save(features, feat_path)
    torch.save(labels, label_path)
    print(f"[{split_name}] extracted and cached: {features.shape}")
    return features, labels


def train_head(model, train_feats, train_labels, val_feats, val_labels, device):
    """Trains only model.fc on cached feature vectors."""
    model.fc = model.fc.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LR)
    criterion = nn.CrossEntropyLoss()

    train_feats, train_labels = train_feats.to(device), train_labels.to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.to(device)

    n = train_feats.size(0)
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.fc.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch_feats, batch_labels = train_feats[idx], train_labels[idx]

            optimizer.zero_grad()
            logits = model.fc(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.fc.eval()
        with torch.no_grad():
            val_logits = model.fc(val_feats)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        print(f"Epoch {epoch:2d}/{HEAD_EPOCHS} | train_loss={total_loss / n:.4f} | val_acc={val_acc:.4f}")

    return val_acc


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    full_train, train_idx, val_idx, test_set = load_splits()

    train_ds = FashionMNISTSubset(full_train, train_idx)
    val_ds = FashionMNISTSubset(full_train, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_classes=10, pretrained=True).to(device)

    print("\n--- Extracting + caching frozen-backbone features ---")
    train_feats, train_labels = extract_and_cache_features(model, train_loader, device, "train")
    val_feats, val_labels = extract_and_cache_features(model, val_loader, device, "val")

    print("\n--- Training classifier head on cached features ---")
    final_val_acc = train_head(model, train_feats, train_labels, val_feats, val_labels, device)

    print(f"\nFeature-extraction-only validation accuracy: {final_val_acc:.4f}")
    if final_val_acc >= 0.80:
        print("=> Meets the 80% bar. Fine-tuning (Task 4) is NOT required, "
              "but you may still run it to see if it improves results further.")
    else:
        print("=> Below the 80% bar. Proceed to task4_finetune.py to unfreeze "
              "layer4 and fine-tune.")

    # Save the head-only model as a checkpoint so task4 can load it if needed.
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/_checkpoint_feature_extraction.pt")
    print("Saved checkpoint: models/_checkpoint_feature_extraction.pt")

"""
Task 5 -- Final evaluation on the held-out test split (touched here for the
first time -- never used for any training or model-selection decision).

Automatically picks whichever checkpoint exists: the fine-tuned one if
task4 was run, otherwise the feature-extraction-only one. Reports test
accuracy, a full 10x10 confusion matrix, and per-class precision/recall.
Saves outputs to CSV so Task 6's confusion-pattern analysis and the README
can reference exact numbers.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model_utils import build_model, unfreeze_layer4, get_device, CLASS_NAMES
from task1_load_data import load_splits
from task2_dataset import FashionMNISTSubset

BATCH_SIZE = 64
FINETUNED_CKPT = "models/_checkpoint_finetuned.pt"
FEATURE_ONLY_CKPT = "models/_checkpoint_feature_extraction.pt"


def load_best_available_model(device):
    model = build_model(num_classes=10, pretrained=True)
    if os.path.exists(FINETUNED_CKPT):
        print(f"Loading fine-tuned checkpoint: {FINETUNED_CKPT}")
        model = unfreeze_layer4(model)  # match the architecture state used when saved
        model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
        source = "fine-tuned"
    elif os.path.exists(FEATURE_ONLY_CKPT):
        print(f"Loading feature-extraction-only checkpoint: {FEATURE_ONLY_CKPT}")
        model.load_state_dict(torch.load(FEATURE_ONLY_CKPT, map_location=device))
        source = "feature-extraction-only"
    else:
        raise FileNotFoundError(
            "No checkpoint found. Run task3_feature_extraction.py first "
            "(and task4_finetune.py if accuracy was below 80%)."
        )
    return model.to(device), source


def run_evaluation():
    device = get_device()
    model, source = load_best_available_model(device)
    model.eval()

    _, _, _, test_set = load_splits()
    test_ds = FashionMNISTSubset(test_set)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n=== Model source: {source} ===")
    print(f"Test accuracy: {acc:.4f}")
    if acc >= 0.80:
        print("=> Meets the required 80% accuracy bar.")
    else:
        print("=> Below the 80% bar -- report honestly, do not fabricate. "
              "See confusion matrix below for diagnosis.")

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv("confusion_matrix.csv")
    print("\nConfusion matrix saved to confusion_matrix.csv")
    print(cm_df)

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")
    print("\nPer-class precision/recall saved to classification_report.csv")
    print(report_df.round(4))

    return acc, cm_df, report_df


if __name__ == "__main__":
    run_evaluation()

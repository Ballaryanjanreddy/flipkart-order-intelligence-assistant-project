# Part 2 -- Product Image Categoriser: Report

**IMPORTANT:** This template has placeholders (`[[FILL IN]]`) for numbers that
can only come from actually running the pipeline on your machine -- I do not
have a working torch/GPU environment here to generate real results, so I
will not fabricate numbers. Run the scripts in order, then fill in every
placeholder below from your own console output / CSV files before treating
this as complete.

Run order:
```
py task1_load_data.py
py task2_dataset.py
py task3_feature_extraction.py
py task4_finetune.py          # only if task3's val accuracy was below 80%
py task5_evaluate.py
py task6_confusion_analysis.py
py task7_save_artifact.py
py task8_export_samples.py
```

---

## Task 1 -- Dataset splits

- Training pool: 60,000 images
- Train subset: `[[FILL IN]]` images
- Validation subset: `[[FILL IN]]` images (must be >= 5,000)
- Test split (untouched until Task 5): 10,000 images

## Task 2 -- Preprocessing

- Input size used: 224x224 (ResNet-18's documented pretrained input size)
- Grayscale replicated to 3 channels: yes
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

## Task 3/4 -- Training

- Optimizer: Adam
- Batch size (feature extraction / fine-tune): 64 / 32
- Learning rate (head / backbone during fine-tune): 1e-3 / 1e-5
- Epochs (feature extraction / fine-tune): 15 / 5

**Feature-extraction-only validation accuracy:** `[[FILL IN]]`

**Was fine-tuning required (< 80% val accuracy)?** `[[FILL IN: yes/no]]`

If yes:
- Validation accuracy BEFORE fine-tuning: `[[FILL IN]]`
- Validation accuracy AFTER fine-tuning: `[[FILL IN]]`

## Task 5 -- Final test-set evaluation

- **Test accuracy:** `[[FILL IN]]`
- Meets 80% bar: `[[FILL IN: yes/no]]`
- Full confusion matrix: see `confusion_matrix.csv`
- Per-class precision/recall: see `classification_report.csv`

## Task 6 -- Confusion pattern analysis

Run `task6_confusion_analysis.py` and paste your actual top-confused pairs
and their explanations here (the script prints pre-written explanations for
the common Fashion-MNIST confusions, e.g. Shirt/T-shirt/Pullover/Coat and
Sneaker/Sandal/Ankle boot -- but only use them if they match YOUR actual
top pairs; if your run confuses a different pair, write a fresh explanation
based on the real visual similarity).

`[[FILL IN: top confused pairs + paragraph explanations from the script output]]`

## Task 7/8 -- Saved artifact

- `models/product_classifier.pt`: `[[CONFIRM EXISTS, size in MB]]`
- `data/sample_images/`: 10 real exported `.png` files (one per class)

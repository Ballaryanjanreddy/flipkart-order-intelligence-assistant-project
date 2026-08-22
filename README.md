# Flipkart Support Assistant

This project brings together three connected machine-learning components that work as one support system:

1. **Return-Risk Scoring** – predicts the likelihood of an order being returned.
2. **Product Image Classification** – identifies product categories using transfer learning.
3. **Flipkart Support Agent** – a LangGraph-based agent that uses the ML models as tools and answers policy-related questions using a retrieval-augmented knowledge base.

## Project Structure

```text
flipkart-support-assistant/
│
├── part1_return_risk/          # Return-risk prediction pipeline
├── part2_image_classifier/     # Product image classification
├── part3_agent/                # LangGraph support agent
└── README.md
```

---

# Part 1 – Return-Risk Scoring Pipeline

**Status: Complete and verified end-to-end.**

Part 1 builds a return-risk prediction model using Python and scikit-learn. The pipeline covers data generation, preprocessing, baseline modelling, logistic regression, random-forest tuning, feature importance, subgroup analysis, and final model export.

### Running Part 1

```powershell
cd part1_return_risk

py -m pip install -r requirements.txt

py generate_orders.py
py preprocessing.py
py task4_baseline.py
py task5_logreg.py
py task6_random_forest.py
py task7_feature_importance.py
py task8_subgroup_analysis.py
py task9_save_artifact.py
```

`generate_orders.py` creates the `orders_dataset.csv` file using a seeded process, so the generated dataset is deterministic.

The final trained model is saved as:

```text
part1_return_risk/models/return_risk_model.pkl
```

The threshold configuration is saved alongside it.

### Key Results

* Dataset size: **6,000 rows**
* Return rate: **22.75%**
* `rating_given` was treated as MAR and analysed against `payment_method`.
* There was a **16.77 percentage-point difference** in missingness between COD and non-COD orders.
* Tuned Random Forest:

  * Cross-validation ROC-AUC: **0.6178**
  * Test ROC-AUC: **0.6143**
* Final Random Forest threshold: **0.46**

A detailed discussion of the analysis and results is available in:

```text
part1_return_risk/PART1_REPORT.md
```

---

# Part 2 – Product Image Classifier

**Status: Code complete.**

Part 2 uses PyTorch and transfer learning to classify product images into 10 categories.

The implementation uses a pretrained **ResNet-18** backbone. Initially, the backbone is frozen and a new classification head is trained. Features can be cached so that the model does not need to repeatedly run the full backbone during head training.

If the initial validation accuracy is below the required threshold, the training pipeline can unfreeze the `layer4` section of ResNet-18 and perform a short fine-tuning stage with a lower learning rate.

### Running Part 2

```powershell
cd part2_image_classifier

py -m pip install -r requirements.txt

py task1_load_data.py
py task3_feature_extraction.py
py task4_finetune.py
py task5_evaluate.py
py task6_confusion_analysis.py
py task7_save_artifact.py
py task8_export_samples.py
```

If the feature-extraction stage already reaches the required validation accuracy, the fine-tuning step can be skipped as described in `PART2_REPORT.md`.

The final model is saved as:

```text
part2_image_classifier/models/product_classifier.pt
```

The confusion-matrix analysis is produced by:

```text
part2_image_classifier/task6_confusion_analysis.py
```

The detailed Part 2 documentation is available in:

```text
part2_image_classifier/PART2_REPORT.md
```

---

# Part 3 – Flipkart Support Agent

Part 3 connects the previous ML components with a LangGraph-based support agent.

The agent uses:

* A policy knowledge base
* Retrieval-augmented generation
* FAISS vector search
* LangGraph
* Two ML tools
* Guardrails
* Conversation state
* Retrieval evaluation

The two ML tools use the actual saved models from Parts 1 and 2 rather than hardcoded predictions:

```text
check_return_risk
classify_product_image
```

### Running Part 3

```powershell
cd part3_agent

py -m pip install -r requirements.txt

py build_index.py
py run_transcripts.py
py evaluate_retrieval.py
```

`build_index.py` creates the vector index from the policy knowledge base.

`run_transcripts.py` generates the required test conversations.

`evaluate_retrieval.py` evaluates retrieval using document-level **Precision@3** and **Recall@3**.

### Agent Flow

The LangGraph workflow consists of five main stages:

```text
Guardrail
    ↓
Intent Detection
    ↓
Retrieval / Tool
    ↓
Response
```

The graph also contains conditional routing depending on the user's request.

The project runs in `MOCK_LLM` mode by default, so the basic workflow can be tested without API keys or external network calls.

### Test Conversations

The transcript suite covers at least eight scenarios:

1. Policy question using retrieval
2. Another policy question using retrieval
3. Return-risk prediction using `check_return_risk`
4. Product-category prediction using `classify_product_image`
5. Multi-turn conversation with state carried between turns
6. Fresh conversation where previous state is not available
7. Prompt-injection attempt that is blocked
8. Unsupported/ungrounded policy question that is refused

Generated conversations are stored under:

```text
part3_agent/transcripts/
```

---

# Retrieval Evaluation

The retrieval evaluation compares the documents returned for each query against the expected relevant documents.

The evaluation reports:

```text
Precision@3
Recall@3
```

The calculation is performed at the document level rather than counting individual chunks as separate documents.

Run:

```powershell
cd part3_agent
py evaluate_retrieval.py
```

The script prints the per-query results along with the average Precision@3 and Recall@3.

---

# Git Workflow

Development is done using feature branches rather than making changes directly on `main`.

A typical workflow is:

```powershell
git checkout main
git pull origin main

git checkout -b feature/<branch-name>

git add .
git commit -m "Add Flipkart support assistant updates"

git push -u origin feature/<branch-name>
```

After pushing the feature branch, a Pull Request can be opened on GitHub and reviewed before merging into `main`.

To inspect the repository history:

```powershell
git log --graph --all
```

---

# Final Repository Contents

The repository brings all three parts together:

```text
flipkart-support-assistant/
│
├── README.md
│
├── part1_return_risk/
│   ├── generate_orders.py
│   ├── orders_dataset.csv
│   ├── training and evaluation scripts
│   ├── PART1_REPORT.md
│   └── models/
│       └── return_risk_model.pkl
│
├── part2_image_classifier/
│   ├── training scripts
│   ├── evaluation scripts
│   ├── confusion-matrix analysis
│   ├── PART2_REPORT.md
│   └── models/
│       └── product_classifier.pt
│
└── part3_agent/
    ├── knowledge base
    ├── vector-index build code
    ├── tools
    ├── LangGraph agent
    ├── transcripts/
    ├── retrieval evaluation
    └── PART3_NOTES.md
```

The three components are designed to work together as a single Flipkart support workflow: the return-risk model handles order risk, the image classifier identifies products, and the LangGraph agent provides the orchestration and policy-aware support layer.

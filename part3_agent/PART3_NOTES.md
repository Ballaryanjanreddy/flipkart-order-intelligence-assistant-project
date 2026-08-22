# Part 3 -- Run Order & Verification Notes

**IMPORTANT:** This code was written but NOT executed by me (no torch/
langgraph/faiss/network access in the environment I built it in). You must
run it yourself, exactly like we did for Part 1's preprocessing.py, and
report back what happens at each step so we can fix anything that breaks.

## Prerequisites
- Part 1 must be complete with `part1_return_risk/models/return_risk_model.pkl`
  and `risk_threshold_config.json` present.
- Part 2 must be complete with `part2_image_classifier/models/product_classifier.pt`
  and at least 5 real `.png` files in `part2_image_classifier/data/sample_images/`.
- Folder layout must be sibling directories under one repo root:
  ```
  repo_root/
    part1_return_risk/
    part2_image_classifier/
    part3_agent/
  ```

## Run order

```
py -m pip install -r requirements.txt
py knowledge_base.py        # sanity-check chunking (14 docs -> N chunks)
py build_index.py           # embeds chunks, builds FAISS index (downloads
                             # all-MiniLM-L6-v2 once, ~90MB, then cached)
py guardrails.py            # sanity-check injection + groundedness functions
py tools.py                 # self-test: calls BOTH real Part 1 + Part 2 artifacts
py agent_graph.py           # sanity-check: runs one policy question through the full graph
py run_transcripts.py       # generates all 8 required transcripts into transcripts/
py evaluate_retrieval.py    # Precision@3 / Recall@3 with per-query arithmetic
```

## Spot-check requirement (brief's acceptance criteria)

The brief requires proving the tools call the REAL saved models, not a
hardcoded stand-in. To verify:

```python
# In a Python shell, from part3_agent/:
from tools import check_return_risk
import joblib
model = joblib.load("../part1_return_risk/models/return_risk_model.pkl")

order = {...}  # same dict used in tools.py's self-test
direct_prob = model.predict_proba(...)[0, 1]   # run the model directly
tool_prob = check_return_risk(order)["return_probability"]
assert abs(direct_prob - tool_prob) < 1e-9
```

Do this once and note the matching numbers in your README as proof.

## Things likely to need small fixes once you actually run this

- `faiss-cpu` install can be finicky on Windows -- if `pip install faiss-cpu`
  fails, try `pip install faiss-cpu --only-binary :all:`.
- `sentence-transformers` will download the `all-MiniLM-L6-v2` model on
  first run (needs network once; fully offline after that).
- If `langgraph`'s API has changed since this was written, `StateGraph`
  usage (`add_conditional_edges`, `set_entry_point`, `.compile()`) may need
  minor adjustment -- check `pip show langgraph` version and their docs if
  you hit an import/API error.

"""
Task 6 -- Document confusion patterns.

Reads the confusion_matrix.csv produced by task5_evaluate.py (run that
first) and identifies the top off-diagonal (i.e., true-label != predicted-
label) cell values -- the pairs the model actually confuses most often, read
directly off the real matrix rather than guessed.

This script only surfaces the DATA (which pairs, how many misclassifications
each way). The written one-paragraph-per-pair visual-similarity explanation
required by the brief still needs a human sentence or two -- fill in the
`EXPLANATIONS` dict below once you see which pairs your run actually
produces (Fashion-MNIST's well-known hard pairs are Shirt/T-shirt/Pullover/
Coat, and Sneaker/Sandal/Ankle boot, but don't assume -- confirm against
your own numbers first).
"""

import pandas as pd

CM_PATH = "confusion_matrix.csv"


def top_confused_pairs(cm_df: pd.DataFrame, top_n: int = 5):
    """Returns the top_n (true_class, predicted_class, count) triples with
    the highest off-diagonal confusion counts."""
    pairs = []
    for true_label in cm_df.index:
        for pred_label in cm_df.columns:
            if true_label == pred_label:
                continue
            count = cm_df.loc[true_label, pred_label]
            if count > 0:
                pairs.append((true_label, pred_label, int(count)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


# Fill these in once you've seen your actual top-confused pairs from the run.
# Each explanation should reference the real visual/silhouette similarity
# between the two garment types.
EXPLANATION_TEMPLATES = {
    frozenset({"Shirt", "T-shirt/top"}): (
        "Both are upper-body garments with a similar boxy silhouette at "
        "28x28 resolution -- the collar and button details that visually "
        "distinguish a shirt from a T-shirt are only a few pixels wide and "
        "are easily lost at this resolution, especially in a side-on or "
        "flat-lay product photo pose."
    ),
    frozenset({"Shirt", "Pullover"}): (
        "Shirts and pullovers share the same general torso-plus-sleeves "
        "outline; the difference (collar/button placket vs. crew neckline) "
        "is a small, localized visual feature that a low-resolution "
        "grayscale image can easily wash out."
    ),
    frozenset({"Shirt", "Coat"}): (
        "At small scale, an open or loosely-draped shirt can present a "
        "silhouette very close to a coat's, particularly since Fashion-MNIST "
        "images are cropped/centered product shots without strong scale cues."
    ),
    frozenset({"Sneaker", "Ankle boot"}): (
        "Both are enclosed, low-to-mid-height footwear with a similar sole "
        "and profile silhouette from a side angle -- the main visual "
        "difference (ankle-boot height, sneaker's mesh/lace texture) is a "
        "fine detail that a small grayscale image doesn't preserve well."
    ),
    frozenset({"Sneaker", "Sandal"}): (
        "Both are low-profile footwear silhouettes; an open-toe sandal photographed "
        "at an angle that hides its straps can present an outline close enough "
        "to a sneaker's for the model to conflate the two."
    ),
    frozenset({"Pullover", "Coat"}): (
        "Both are outer upper-body garments with long sleeves and a similar "
        "boxy torso silhouette; the layering/thickness cues that distinguish "
        "a coat from a pullover are subtle and resolution-sensitive."
    ),
}


def explain(true_label, pred_label):
    key = frozenset({true_label, pred_label})
    return EXPLANATION_TEMPLATES.get(
        key,
        "(No pre-written explanation for this pair -- write one based on "
        "the two garments' visual/silhouette similarity.)",
    )


if __name__ == "__main__":
    cm_df = pd.read_csv(CM_PATH, index_col=0)
    top_pairs = top_confused_pairs(cm_df, top_n=5)

    print("Top confused category pairs (true -> predicted, count):\n")
    for true_label, pred_label, count in top_pairs:
        print(f"  {true_label:15s} -> {pred_label:15s} : {count} misclassifications")

    print("\n--- Written explanations (first 2+, per brief requirement) ---\n")
    seen_pairs = set()
    explained = 0
    for true_label, pred_label, count in top_pairs:
        key = frozenset({true_label, pred_label})
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        print(f"### {true_label} <-> {pred_label} ({count} misclassifications)")
        print(explain(true_label, pred_label))
        print()
        explained += 1
        if explained >= 2:
            break

"""
Task 10 -- Evaluate retrieval.

Computes Precision@3 and Recall@3 for each query in
knowledge_base.RETRIEVAL_ANSWER_KEY, scored at the DOCUMENT level: each of
the top-3 retrieved CHUNKS is mapped back to its parent doc_id and
de-duplicated before comparing against the answer key's relevant_doc_ids.
Shows per-query arithmetic, not just the final averages, as required.
"""

from knowledge_base import RETRIEVAL_ANSWER_KEY
from build_index import load_index_and_model, retrieve


def evaluate_retrieval():
    index, chunks, model = load_index_and_model()

    precisions, recalls = [], []

    print(f"{'Query':60s} | {'Retrieved docs':25s} | {'Relevant docs':20s} | P@3    | R@3")
    print("-" * 130)

    for item in RETRIEVAL_ANSWER_KEY:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        results = retrieve(query, index, chunks, model, top_k=3)
        # De-duplicate chunk-level results down to document-level hits,
        # preserving order of first appearance (highest similarity first).
        retrieved_docs = []
        for r in results:
            if r["doc_id"] not in retrieved_docs:
                retrieved_docs.append(r["doc_id"])

        retrieved_set = set(retrieved_docs)
        true_positives = retrieved_set & relevant

        precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(true_positives) / len(relevant) if relevant else 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(f"{query[:58]:60s} | {','.join(retrieved_docs):25s} | {','.join(relevant):20s} "
              f"| {len(true_positives)}/{len(retrieved_set)}={precision:.3f} | "
              f"{len(true_positives)}/{len(relevant)}={recall:.3f}")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print("-" * 130)
    print(f"Average Precision@3: {avg_precision:.4f}")
    print(f"Average Recall@3:    {avg_recall:.4f}")

    return avg_precision, avg_recall, list(zip(RETRIEVAL_ANSWER_KEY, precisions, recalls))


if __name__ == "__main__":
    evaluate_retrieval()

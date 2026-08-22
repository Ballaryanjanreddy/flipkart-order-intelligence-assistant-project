"""
Task 2 -- Embed and index.

Embeds every chunk from knowledge_base.py with a free, local
sentence-transformer (all-MiniLM-L6-v2) and builds a FAISS index over them.
Both are free, local, and require no account or API key.

Saves the index + chunk metadata to disk so the agent doesn't need to
re-embed on every run.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from knowledge_base import build_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "kb_index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "chunks_metadata.json")


def build_and_save_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    chunks = build_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model: {MODEL_NAME} (downloads once, then cached locally)")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    # Normalized embeddings + inner product = cosine similarity search.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")


def load_index_and_model():
    """Loads the saved index + metadata + embedding model, for use by the
    RAG retrieval node in agent_graph.py."""
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        chunks = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, chunks, model


def retrieve(query: str, index, chunks, model, top_k: int = 3):
    """Returns top_k chunks with their cosine-similarity scores, sorted
    descending. Each result includes the parent doc_id for document-level
    scoring/dedup downstream."""
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "doc_title": chunk["doc_title"],
            "text": chunk["text"],
            "similarity": float(score),
        })
    return results


if __name__ == "__main__":
    build_and_save_index()

    # Quick self-test
    index, chunks, model = load_index_and_model()
    test_query = "How long do I have to return shoes?"
    results = retrieve(test_query, index, chunks, model, top_k=3)
    print(f"\nTest query: {test_query!r}")
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r['doc_id']} ({r['doc_title']}): {r['text']}")

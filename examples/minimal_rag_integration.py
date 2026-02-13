from __future__ import annotations

import numpy as np

from vector_engine import VectorArray, VectorIndex


def fake_embed(texts: list[str], dim: int = 64) -> np.ndarray:
    """Replace this with your real embedder output."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(len(texts), dim)).astype(np.float32)


def build_retriever(doc_texts: list[str]) -> VectorIndex:
    doc_ids = [f"doc-{i}" for i in range(len(doc_texts))]
    metadata = [{"text": t} for t in doc_texts]
    xb = fake_embed(doc_texts)
    vectors = VectorArray.from_numpy(xb, ids=doc_ids, metadata=metadata, normalize=True)
    return VectorIndex.create(vectors, metric="cosine", backend="bruteforce")


def retrieve_context(index: VectorIndex, query_text: str, k: int = 3) -> list[dict]:
    xq = VectorArray.from_numpy(fake_embed([query_text]), normalize=True)
    res = index.search(xq, k=k, return_metadata=True)
    return [
        {"doc_id": doc_id, "score": float(score), "metadata": md}
        for doc_id, score, md in zip(res.ids[0], res.scores[0], res.metadata[0])
    ]


if __name__ == "__main__":
    docs = [
        "Vector search retrieves context for generation.",
        "RAG pipelines use nearest neighbors over embeddings.",
        "Evaluation metrics include recall at k and ndcg.",
    ]
    index = build_retriever(docs)
    context = retrieve_context(index, "How can I retrieve context for a generator?", k=2)
    for row in context:
        print(row)

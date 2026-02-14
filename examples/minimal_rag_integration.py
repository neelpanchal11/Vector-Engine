from __future__ import annotations

import re
from hashlib import blake2b

import numpy as np

from vector_engine import VectorArray, VectorIndex
from vector_engine.eval import retrieval_report


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_vec(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def fake_embed(texts: list[str], dim: int = 64) -> np.ndarray:
    """Replace this with your real embedder output."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = _tokenize(text)
        if not toks:
            continue
        for token in toks:
            out[i] += _token_vec(token, dim)
    return out


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
    query_text = "How can I retrieve context for a generator?"
    context = retrieve_context(index, query_text, k=2)
    for row in context:
        print(row)
    gt = np.array([["doc-0", "doc-1"]], dtype=object)
    retrieved_ids = np.array([[c["doc_id"] for c in context]], dtype=object)
    print("metrics", retrieval_report(retrieved_ids, gt, ks=(1, 2)))

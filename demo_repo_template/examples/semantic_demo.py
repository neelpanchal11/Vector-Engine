from __future__ import annotations

import os
import sys

import numpy as np

try:
    from vector_engine import VectorArray, VectorIndex
    from vector_engine.eval import retrieval_report
except ModuleNotFoundError:
    # Fallback for monorepo usage before package installation.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vector_engine import VectorArray, VectorIndex
    from vector_engine.eval import retrieval_report


def main() -> None:
    docs = [
        "RAG retrieves context for generation.",
        "Faiss is optimized for nearest-neighbor search.",
        "Vector Engine wraps backends with a unified API.",
    ]
    ids = [f"doc-{i}" for i in range(len(docs))]
    rng = np.random.default_rng(42)
    xb = rng.normal(size=(len(docs), 32)).astype(np.float32)
    xq = xb[[0]] + 0.01 * rng.normal(size=(1, 32)).astype(np.float32)

    base = VectorArray.from_numpy(xb, ids=ids, metadata=[{"text": t} for t in docs], normalize=True)
    query = VectorArray.from_numpy(xq, ids=["q0"], normalize=True)
    index = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    res = index.search(query, k=2, return_metadata=True)
    gt = np.array([[ids[0]]], dtype=object)
    print("metrics", retrieval_report(res.ids, gt, ks=(1, 2)))
    print("top hits", res.ids.tolist())


if __name__ == "__main__":
    main()

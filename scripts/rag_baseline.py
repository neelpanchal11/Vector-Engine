from __future__ import annotations

import argparse
import json
import os
import re
from hashlib import blake2b

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from vector_engine import VectorArray, VectorIndex
from vector_engine.eval import retrieval_report

ARTIFACT_VERSION = "v1"

FIXTURE_DOCS = [
    {"id": "doc_python", "text": "Python is used for machine learning and data analysis."},
    {"id": "doc_faiss", "text": "Faiss enables fast nearest neighbor search for dense vectors."},
    {"id": "doc_rag", "text": "RAG retrieves relevant context before text generation."},
    {"id": "doc_knn", "text": "k nearest neighbors is a simple non parametric method."},
    {"id": "doc_eval", "text": "Recall at k and NDCG evaluate ranking quality."},
    {"id": "doc_torch", "text": "PyTorch tensors can represent embeddings for retrieval models."},
]

FIXTURE_QUERIES = [
    {"text": "How do we evaluate ranking quality?", "relevant": {"doc_eval"}},
    {"text": "What is RAG retrieval?", "relevant": {"doc_rag"}},
    {"text": "Dense vector nearest neighbor search library", "relevant": {"doc_faiss"}},
    {"text": "Non parametric neighbors method", "relevant": {"doc_knn"}},
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_vec(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def embed_texts(texts: list[str], dim: int = 128) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = _tokenize(text)
        if not toks:
            continue
        for token in toks:
            out[i] += _token_vec(token, dim)
    return out


def run_baseline(output_dir: str, k: int = 3) -> dict[str, object]:
    docs = FIXTURE_DOCS
    queries = FIXTURE_QUERIES
    doc_ids = [d["id"] for d in docs]
    xb = embed_texts([d["text"] for d in docs], dim=128)
    xq = embed_texts([q["text"] for q in queries], dim=128)

    base = VectorArray.from_numpy(xb, ids=doc_ids, metadata=docs, normalize=True)
    query = VectorArray.from_numpy(xq, ids=[f"q{i}" for i in range(len(queries))], normalize=True)
    index = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    result = index.search(query, k=k, return_metadata=True)

    gt = np.array([[x for x in q["relevant"]] for q in queries], dtype=object)
    report = retrieval_report(result.ids, gt, ks=(1, 3))

    payload: dict[str, object] = {
        "artifact_version": ARTIFACT_VERSION,
        "k": k,
        "num_docs": len(docs),
        "num_queries": len(queries),
        "metrics": report,
        "topk_ids": result.ids.tolist(),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rag_baseline_metrics.{ARTIFACT_VERSION}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible RAG baseline for Vector Engine.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to save metrics artifact.")
    parser.add_argument("--k", default=3, type=int, help="Top-k to retrieve for baseline report.")
    args = parser.parse_args()
    payload = run_baseline(args.output_dir, k=args.k)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote artifact to {os.path.join(args.output_dir, f'rag_baseline_metrics.{ARTIFACT_VERSION}.json')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

try:
    from vector_engine import VectorArray, VectorIndex
except ModuleNotFoundError:
    # Fallback for monorepo usage before package installation.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vector_engine import VectorArray, VectorIndex


def _timed_search(index: VectorIndex, query: VectorArray, *, k: int, warmup: int, loops: int) -> tuple[dict[str, float], np.ndarray]:
    for _ in range(warmup):
        index.search(query, k=k, return_metadata=False)
    latencies: list[float] = []
    last_ids: np.ndarray | None = None
    t0 = time.perf_counter()
    for _ in range(loops):
        t = time.perf_counter()
        res = index.search(query, k=k, return_metadata=False)
        latencies.append((time.perf_counter() - t) * 1000.0)
        last_ids = res.ids
    elapsed = time.perf_counter() - t0
    qps = (query.shape[0] * loops) / elapsed
    return {
        "qps": float(qps),
        "latency_p50_ms": float(np.percentile(np.asarray(latencies, dtype=np.float64), 50)),
        "latency_p95_ms": float(np.percentile(np.asarray(latencies, dtype=np.float64), 95)),
    }, last_ids


def _overlap_at_k(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    vals = []
    for i in range(reference.shape[0]):
        a = set(reference[i, :k].tolist())
        b = set(candidate[i, :k].tolist())
        vals.append(len(a & b) / float(k))
    return float(np.mean(vals))


def main() -> None:
    rng = np.random.default_rng(7)
    xb = rng.normal(size=(5000, 64)).astype(np.float32)
    xq = rng.normal(size=(100, 64)).astype(np.float32)
    base = VectorArray.from_numpy(xb, ids=np.arange(xb.shape[0]), normalize=True)
    query = VectorArray.from_numpy(xq, normalize=True)

    k = 10
    warmup = 1
    loops = 3
    rows: list[dict[str, float | str]] = []

    bf = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    bf_stats, bf_ids = _timed_search(bf, query, k=k, warmup=warmup, loops=loops)
    rows.append(
        {
            "backend": "bruteforce",
            "qps": bf_stats["qps"],
            "latency_p50_ms": bf_stats["latency_p50_ms"],
            "latency_p95_ms": bf_stats["latency_p95_ms"],
            "overlap_vs_bruteforce": 1.0,
        }
    )

    try:
        ff = VectorIndex.create(
            base,
            metric="cosine",
            backend="faiss",
            backend_config={"index_factory": "Flat"},
        )
        ff_stats, ff_ids = _timed_search(ff, query, k=k, warmup=warmup, loops=loops)
        rows.append(
            {
                "backend": "faiss_flat",
                "qps": ff_stats["qps"],
                "latency_p50_ms": ff_stats["latency_p50_ms"],
                "latency_p95_ms": ff_stats["latency_p95_ms"],
                "overlap_vs_bruteforce": _overlap_at_k(bf_ids, ff_ids, k),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "backend": "faiss_flat",
                "qps": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "overlap_vs_bruteforce": 0.0,
                "note": f"faiss unavailable: {exc}",
            }
        )

    print(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()

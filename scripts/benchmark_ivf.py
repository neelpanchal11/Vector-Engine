"""Standalone recall/latency/QPS sweep for the IVF backend vs. bruteforce.

Kept separate from scripts/benchmark_matrix.py (which is wired specifically to
benchmarks/compare_bruteforce_vs_faiss.py and its contract-validated schema).
This script answers a narrower question: for a given nprobe, what recall and
speed does IVF give up relative to exact search on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from vector_engine import VectorArray, VectorIndex


def _time_search(index: VectorIndex, xq: VectorArray, k: int, loops: int) -> dict[str, float]:
    latencies_ms = []
    for _ in range(loops):
        start = time.perf_counter()
        index.search(xq, k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    total_queries = xq.shape[0] * loops
    total_seconds = float(np.sum(arr)) / 1000.0
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "qps": float(total_queries / total_seconds) if total_seconds > 0 else 0.0,
    }


def _recall_at_k(exact_ids: np.ndarray, other_ids: np.ndarray) -> float:
    hits = 0
    total = 0
    for row_exact, row_other in zip(exact_ids, other_ids):
        hits += len(set(row_exact.tolist()) & set(row_other.tolist()))
        total += len(row_exact)
    return hits / total if total > 0 else 0.0


def run(
    *,
    n: int,
    d: int,
    nq: int,
    k: int,
    n_clusters: int,
    nprobe_options: list[int],
    loops: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    xb = VectorArray.from_numpy(rng.standard_normal((n, d)).astype(np.float32), ids=np.arange(n))
    xq = VectorArray.from_numpy(rng.standard_normal((nq, d)).astype(np.float32), ids=np.arange(nq))

    exact_index = VectorIndex.create(xb, metric="l2", backend="bruteforce")
    exact_result = exact_index.search(xq, k=k)
    exact_timing = _time_search(exact_index, xq, k, loops)

    sweep = []
    for nprobe in nprobe_options:
        ivf_index = VectorIndex.create(
            xb,
            metric="l2",
            backend="ivf",
            backend_config={"n_clusters": n_clusters, "nprobe": nprobe, "random_state": seed},
        )
        ivf_result = ivf_index.search(xq, k=k)
        timing = _time_search(ivf_index, xq, k, loops)
        sweep.append(
            {
                "nprobe": nprobe,
                "recall_at_k": _recall_at_k(exact_result.ids, ivf_result.ids),
                **timing,
            }
        )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"n": n, "d": d, "nq": nq, "k": k, "n_clusters": n_clusters, "loops": loops, "seed": seed},
        "bruteforce": exact_timing,
        "ivf_sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IVF backend recall/latency vs bruteforce.")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--nq", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--nprobe-options", default="1,4,8,16,32,100")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/ivf_benchmark/ivf_benchmark.json")
    args = parser.parse_args()

    nprobe_options = [int(x) for x in args.nprobe_options.split(",")]
    result = run(
        n=args.n,
        d=args.d,
        nq=args.nq,
        k=args.k,
        n_clusters=args.n_clusters,
        nprobe_options=nprobe_options,
        loops=args.loops,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

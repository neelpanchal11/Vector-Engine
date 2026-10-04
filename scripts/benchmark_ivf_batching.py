"""Compare IVF cluster-batched search with the previous per-query scoring loop."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from vector_engine import VectorArray, VectorIndex
from vector_engine.backends.ivf import _pairwise_scores


def _previous_search(backend, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    probe_scores = _pairwise_scores(queries, backend.centroids, backend.metric)
    probe_clusters = np.argsort(probe_scores, axis=1)[:, : backend.nprobe]
    k_eff = min(k, backend.xb.shape[0])
    all_scores = np.full((len(queries), k_eff), np.nan, dtype=np.float32)
    all_ids = np.full((len(queries), k_eff), -1, dtype=np.int64)

    for row, query in enumerate(queries):
        candidate_ids = np.flatnonzero(np.isin(backend.labels, probe_clusters[row]))
        if candidate_ids.size == 0:
            candidate_ids = np.arange(backend.xb.shape[0])
        scores = _pairwise_scores(query[None, :], backend.xb[candidate_ids], backend.metric)[0]
        count = min(k_eff, len(candidate_ids))
        top = np.argpartition(scores, kth=count - 1)[:count]
        top = top[np.argsort(scores[top])]
        all_scores[row, :count] = scores[top]
        all_ids[row, :count] = candidate_ids[top]

    return all_scores, all_ids


def _measure(search, warmup: int, loops: int) -> dict[str, float]:
    for _ in range(warmup):
        search()
    samples_ms = []
    for _ in range(loops):
        start = time.perf_counter()
        search()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "samples_ms": samples_ms,
    }


def run(*, n: int, d: int, nq: int, k: int, n_clusters: int, nprobe_options: list[int],
        loops: int, warmup: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vectors = VectorArray.from_numpy(
        rng.standard_normal((n, d)).astype(np.float32), ids=np.arange(n)
    )
    queries = VectorArray.from_numpy(
        rng.standard_normal((nq, d)).astype(np.float32), ids=np.arange(nq)
    )
    index = VectorIndex.create(
        vectors,
        metric="l2",
        backend="ivf",
        backend_config={
            "n_clusters": n_clusters,
            "nprobe": max(nprobe_options),
            "random_state": seed,
        },
    )
    backend = index._backend
    rows = []

    for nprobe in nprobe_options:
        backend.nprobe = min(nprobe, n_clusters)
        query_values = queries.values
        previous = _previous_search(backend, query_values, k)
        batched = backend.search(query_values, k)
        if not np.array_equal(previous[1], batched[1]) or not np.allclose(
            previous[0], batched[0], equal_nan=True
        ):
            raise AssertionError(f"search results differ at nprobe={nprobe}")

        previous_timing = _measure(
            lambda: _previous_search(backend, query_values, k), warmup, loops
        )
        batched_timing = _measure(
            lambda: backend.search(query_values, k), warmup, loops
        )
        rows.append(
            {
                "nprobe": nprobe,
                "results_identical": True,
                "previous_loop": previous_timing,
                "cluster_batched": batched_timing,
                "speedup": previous_timing["median_ms"] / batched_timing["median_ms"],
            }
        )

    return {
        "benchmark": "ivf_cluster_batching",
        "protocol": {
            "n": n,
            "d": d,
            "nq": nq,
            "k": k,
            "n_clusters": n_clusters,
            "nprobe_options": nprobe_options,
            "loops": loops,
            "warmup": warmup,
            "seed": seed,
            "metric": "l2",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--nq", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--nprobe-options", default="1,4,16,32")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--output", default="artifacts/ivf_benchmark/ivf_batching_comparison.json")
    args = parser.parse_args()
    options = [int(value) for value in args.nprobe_options.split(",")]
    if not options or any(value <= 0 for value in options):
        parser.error("--nprobe-options must contain positive integers")
    if args.loops <= 0 or args.warmup < 0:
        parser.error("--loops must be positive and --warmup must be non-negative")

    result = run(
        n=args.n,
        d=args.d,
        nq=args.nq,
        k=args.k,
        n_clusters=args.n_clusters,
        nprobe_options=options,
        loops=args.loops,
        warmup=args.warmup,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

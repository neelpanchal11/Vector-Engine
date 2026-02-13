import time
import os
import sys
import argparse
import json

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from vector_engine import VectorArray, VectorIndex


def overlap_at_k(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    vals = []
    for i in range(reference.shape[0]):
        a = set(reference[i, :k].tolist())
        b = set(candidate[i, :k].tolist())
        vals.append(len(a & b) / float(k))
    return float(np.mean(vals))


def percentile_ms(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def estimate_memory_mb(xb: np.ndarray, xq: np.ndarray) -> float:
    # Coarse in-process estimate for benchmark comparability.
    return float((xb.nbytes + xq.nbytes) / (1024**2))


def timed_search(
    index: VectorIndex,
    queries: VectorArray,
    *,
    k: int,
    warmup: int,
    loops: int,
) -> tuple[dict[str, float], np.ndarray]:
    for _ in range(warmup):
        index.search(queries, k=k, return_metadata=False)

    latencies = []
    last_ids = None
    t0 = time.perf_counter()
    for _ in range(loops):
        t_loop = time.perf_counter()
        res = index.search(queries, k=k, return_metadata=False)
        latencies.append((time.perf_counter() - t_loop) * 1000.0)
        last_ids = res.ids
    elapsed = time.perf_counter() - t0
    qps = (queries.shape[0] * loops) / elapsed
    stats = {
        "qps": float(qps),
        "latency_p50_ms": percentile_ms(latencies, 50.0),
        "latency_p95_ms": percentile_ms(latencies, 95.0),
    }
    return stats, last_ids


def maybe_build_faiss(base: VectorArray, mode: str) -> list[tuple[str, VectorIndex]]:
    indexes: list[tuple[str, VectorIndex]] = []
    try:
        indexes.append(
            (
                "faiss_flat",
                VectorIndex.create(
                    base,
                    metric="cosine",
                    backend="faiss",
                    backend_config={"index_factory": "Flat"},
                ),
            )
        )
    except Exception as exc:
        print(f"faiss unavailable: {exc}")
        return indexes

    if mode in {"ann", "all"}:
        try:
            indexes.append(
                (
                    "faiss_ivf",
                    VectorIndex.create(
                        base,
                        metric="cosine",
                        backend="faiss",
                        backend_config={"index_factory": "IVF128,Flat", "nprobe": 8},
                    ),
                )
            )
        except Exception as exc:
            print(f"faiss IVF unavailable: {exc}")
    return indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bruteforce vs faiss backends.")
    parser.add_argument("--n", type=int, default=10000, help="Number of database vectors.")
    parser.add_argument("--d", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--nq", type=int, default=200, help="Number of query vectors per loop.")
    parser.add_argument("--k", type=int, default=10, help="Top-k neighbors.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup search loops.")
    parser.add_argument("--loops", type=int, default=8, help="Timed search loops.")
    parser.add_argument(
        "--mode",
        choices=("exact", "ann", "all"),
        default="exact",
        help="Benchmark mode: exact=bruteforce+faiss Flat, ann=bruteforce+faiss Flat+IVF.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    xb = rng.normal(size=(args.n, args.d)).astype(np.float32)
    xq = rng.normal(size=(args.nq, args.d)).astype(np.float32)

    base = VectorArray.from_numpy(xb, ids=np.arange(args.n), normalize=True)
    query = VectorArray.from_numpy(xq, normalize=True)
    mem_mb = estimate_memory_mb(xb, xq)

    rows: list[dict[str, float | str | None]] = []
    bf = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    bf_stats, ids_bf = timed_search(bf, query, k=args.k, warmup=args.warmup, loops=args.loops)
    rows.append(
        {
            "backend": "bruteforce",
            "qps": bf_stats["qps"],
            "latency_p50_ms": bf_stats["latency_p50_ms"],
            "latency_p95_ms": bf_stats["latency_p95_ms"],
            "overlap_vs_bruteforce": 1.0,
            "memory_mb_estimate": mem_mb,
        }
    )

    for name, idx in maybe_build_faiss(base, args.mode):
        stats, ids = timed_search(idx, query, k=args.k, warmup=args.warmup, loops=args.loops)
        rows.append(
            {
                "backend": name,
                "qps": stats["qps"],
                "latency_p50_ms": stats["latency_p50_ms"],
                "latency_p95_ms": stats["latency_p95_ms"],
                "overlap_vs_bruteforce": overlap_at_k(ids_bf, ids, args.k),
                "memory_mb_estimate": mem_mb,
            }
        )

    summary = {
        "config": {
            "seed": args.seed,
            "n": args.n,
            "d": args.d,
            "nq": args.nq,
            "k": args.k,
            "warmup": args.warmup,
            "loops": args.loops,
            "mode": args.mode,
        },
        "results": rows,
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

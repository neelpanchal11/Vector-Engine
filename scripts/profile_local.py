from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.benchmark_matrix import run_matrix
from scripts.rag_real_corpus_eval import _parse_ks
from scripts.stability_runs import run_stability_study


def run_local_profile(
    *,
    embeddings: str,
    query_embeddings: str,
    ids: str,
    ground_truth: str,
    metadata: str | None,
    output_dir: str,
    matrix: list[dict[str, int]] | None = None,
    stability_run_count: int = 50,
    matrix_max_memory_mb: float = 6144.0,
    matrix_min_flat_overlap: float | None = None,
) -> dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    benchmark_dir = os.path.join(output_dir, "benchmark_matrix")
    testing_dir = os.path.join(output_dir, "testing_runs")
    matrix_cfg = matrix or [
        {"name": "m3_medium_a", "n": 10000, "d": 128, "nq": 200, "k": 10},
        {"name": "m3_medium_b", "n": 25000, "d": 256, "nq": 300, "k": 20},
    ]
    matrix_summary = run_matrix(
        matrix=matrix_cfg,
        mode="exact",
        warmup=2,
        loops=8,
        seed=7,
        min_flat_overlap=matrix_min_flat_overlap,
        out_dir=benchmark_dir,
        profile="medium",
        max_memory_mb=matrix_max_memory_mb,
    )
    stability_summary = run_stability_study(
        embeddings_path=embeddings,
        query_embeddings_path=query_embeddings,
        ids_path=ids,
        ground_truth_path=ground_truth,
        metadata_path=metadata,
        backend="bruteforce",
        k=10,
        ks=_parse_ks("1,5,10"),
        loops=5,
        run_count=stability_run_count,
        out_runs_path=os.path.join(testing_dir, f"stability_runs_bruteforce_{stability_run_count}.jsonl"),
        out_summary_path=os.path.join(testing_dir, f"stability_summary_bruteforce_{stability_run_count}.json"),
        threshold_recall=None,
        threshold_ndcg=None,
        threshold_p95_ms=None,
    )
    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_matrix_summary_path": os.path.join(benchmark_dir, "matrix_summary.json"),
        "stability_summary_path": os.path.join(testing_dir, f"stability_summary_bruteforce_{stability_run_count}.json"),
        "benchmark_backend_summary": matrix_summary["backend_summary"],
        "stability_performance_summary": stability_summary["performance_summary"],
    }
    output_path = os.path.join(output_dir, "profile_local_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run medium-scale M3 Pro local profiling baseline.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--query-embeddings", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output-dir", default="artifacts/local_profile")
    parser.add_argument("--stability-run-count", type=int, default=50)
    parser.add_argument("--matrix-max-memory-mb", type=float, default=6144.0)
    parser.add_argument(
        "--matrix-min-flat-overlap",
        type=float,
        default=None,
        help="Optional faiss_flat overlap gate for exact mode matrix runs.",
    )
    args = parser.parse_args()
    payload = run_local_profile(
        embeddings=args.embeddings,
        query_embeddings=args.query_embeddings,
        ids=args.ids,
        ground_truth=args.ground_truth,
        metadata=args.metadata,
        output_dir=args.output_dir,
        stability_run_count=args.stability_run_count,
        matrix_max_memory_mb=args.matrix_max_memory_mb,
        matrix_min_flat_overlap=args.matrix_min_flat_overlap,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

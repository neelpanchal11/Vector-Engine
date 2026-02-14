from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.rag_real_corpus_eval import evaluate_real_corpus, _parse_ks


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("input_error: cannot summarize empty values")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "std": std,
        "cv": float(std / mean) if mean != 0.0 else 0.0,
        "p02_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_stability_study(
    *,
    embeddings_path: str,
    query_embeddings_path: str,
    ids_path: str,
    ground_truth_path: str,
    metadata_path: str | None,
    backend: str,
    k: int,
    ks: tuple[int, ...],
    loops: int,
    run_count: int,
    out_runs_path: str,
    out_summary_path: str,
    threshold_recall: float | None,
    threshold_ndcg: float | None,
    threshold_p95_ms: float | None,
) -> dict[str, Any]:
    if run_count <= 0:
        raise ValueError("input_error: run_count must be > 0")
    os.makedirs(os.path.dirname(out_runs_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_summary_path) or ".", exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    tmp_single_run_path = os.path.join(os.path.dirname(out_summary_path) or ".", "_tmp_single_run.json")
    try:
        with open(out_runs_path, "w", encoding="utf-8") as f:
            for run_idx in range(1, run_count + 1):
                payload = evaluate_real_corpus(
                    embeddings_path=embeddings_path,
                    query_embeddings_path=query_embeddings_path,
                    ids_path=ids_path,
                    ground_truth_path=ground_truth_path,
                    metadata_path=metadata_path,
                    output_path=tmp_single_run_path,
                    backend=backend,
                    k=k,
                    ks=ks,
                    loops=loops,
                    threshold_recall=threshold_recall,
                    threshold_ndcg=threshold_ndcg,
                    threshold_p95_ms=threshold_p95_ms,
                )
                row = {
                    "run_index": run_idx,
                    "metrics": payload["metrics"],
                    "performance": payload["performance"],
                    "checks": payload["checks"],
                    "runtime_seconds": payload.get("runtime_seconds"),
                }
                run_rows.append(row)
                f.write(json.dumps(row) + "\n")
    finally:
        if os.path.exists(tmp_single_run_path):
            os.remove(tmp_single_run_path)

    perf_p50 = [float(r["performance"]["latency_p50_ms"]) for r in run_rows]
    perf_p95 = [float(r["performance"]["latency_p95_ms"]) for r in run_rows]
    perf_qps = [float(r["performance"]["qps"]) for r in run_rows]

    metric_keys = sorted(run_rows[0]["metrics"].keys())
    metric_stats: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        metric_stats[key] = _stats([float(r["metrics"][key]) for r in run_rows])

    check_keys = sorted(run_rows[0]["checks"].keys()) if run_rows[0]["checks"] else []
    check_pass_rate = {
        key: float(np.mean(np.asarray([1.0 if r["checks"].get(key, False) else 0.0 for r in run_rows], dtype=np.float64)))
        for key in check_keys
    }

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": run_count,
        "backend": backend,
        "config": {
            "k": k,
            "ks": list(ks),
            "loops": loops,
            "threshold_recall": threshold_recall,
            "threshold_ndcg": threshold_ndcg,
            "threshold_p95_ms": threshold_p95_ms,
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "performance_summary": {
            "latency_p50_ms": _stats(perf_p50),
            "latency_p95_ms": _stats(perf_p95),
            "qps": _stats(perf_qps),
        },
        "metric_summary": metric_stats,
        "check_pass_rate": check_pass_rate,
        "input_files": {
            "embeddings_path": embeddings_path,
            "query_embeddings_path": query_embeddings_path,
            "ids_path": ids_path,
            "ground_truth_path": ground_truth_path,
            "metadata_path": metadata_path,
        },
        "runs_path": out_runs_path,
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated retrieval-eval stability study and save stats.")
    parser.add_argument("--embeddings", required=True, help="Path to corpus embeddings .npy")
    parser.add_argument("--query-embeddings", required=True, help="Path to query embeddings .npy")
    parser.add_argument("--ids", required=True, help="Path to JSON list of corpus IDs")
    parser.add_argument("--ground-truth", required=True, help="Path to JSON list of relevant IDs per query")
    parser.add_argument("--metadata", default=None, help="Optional JSON list of metadata")
    parser.add_argument("--backend", default="bruteforce", choices=("bruteforce", "faiss"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--run-count", type=int, default=200)
    parser.add_argument("--output-dir", default="artifacts/testing_runs")
    parser.add_argument("--threshold-recall", type=float, default=None)
    parser.add_argument("--threshold-ndcg", type=float, default=None)
    parser.add_argument("--threshold-p95-ms", type=float, default=None)
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    out_runs_path = os.path.join(args.output_dir, f"stability_runs_{args.backend}_{args.run_count}.jsonl")
    out_summary_path = os.path.join(args.output_dir, f"stability_summary_{args.backend}_{args.run_count}.json")

    summary = run_stability_study(
        embeddings_path=args.embeddings,
        query_embeddings_path=args.query_embeddings,
        ids_path=args.ids,
        ground_truth_path=args.ground_truth,
        metadata_path=args.metadata,
        backend=args.backend,
        k=args.k,
        ks=ks,
        loops=args.loops,
        run_count=args.run_count,
        out_runs_path=out_runs_path,
        out_summary_path=out_summary_path,
        threshold_recall=args.threshold_recall,
        threshold_ndcg=args.threshold_ndcg,
        threshold_p95_ms=args.threshold_p95_ms,
    )
    print(json.dumps(summary["performance_summary"], indent=2))
    print(json.dumps(summary["metric_summary"], indent=2))
    print(f"wrote runs: {out_runs_path}")
    print(f"wrote summary: {out_summary_path}")


if __name__ == "__main__":
    main()

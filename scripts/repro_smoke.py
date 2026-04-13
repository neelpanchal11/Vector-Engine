from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.artifact_contracts import (
    validate_benchmark_report,
    validate_matrix_summary,
    validate_publishable_summary,
    validate_real_corpus_payload,
    validate_stability_summary,
)
from scripts.benchmark_matrix import run_matrix
from scripts.publishable_results import build_publishable_summary
from scripts.performance_gates import run_performance_gates
from scripts.rag_baseline import run_baseline
from scripts.rag_real_corpus_eval import evaluate_real_corpus
from scripts.stability_runs import run_stability_study


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _materialize_synthetic_inputs(root: Path, *, n: int = 64, d: int = 32, nq: int = 12) -> dict[str, str]:
    rng = np.random.default_rng(7)
    xb = rng.normal(size=(n, d)).astype(np.float32)
    xq = xb[:nq] + 0.01 * rng.normal(size=(nq, d)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(n)]
    gt = [[ids[i]] for i in range(nq)]
    md = [{"id": ids[i], "source": "synthetic"} for i in range(n)]

    emb_path = root / "real_corpus_inputs" / "embeddings.npy"
    q_path = root / "real_corpus_inputs" / "query_embeddings.npy"
    ids_path = root / "real_corpus_inputs" / "ids.json"
    gt_path = root / "real_corpus_inputs" / "ground_truth.json"
    md_path = root / "real_corpus_inputs" / "metadata.json"

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, xb)
    np.save(q_path, xq)
    _write_json(ids_path, ids)
    _write_json(gt_path, gt)
    _write_json(md_path, md)

    return {
        "embeddings": str(emb_path),
        "query_embeddings": str(q_path),
        "ids": str(ids_path),
        "ground_truth": str(gt_path),
        "metadata": str(md_path),
    }


def run_smoke(output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    inputs = _materialize_synthetic_inputs(root)

    baseline = run_baseline(str(root))
    validate_real_corpus_payload(
        evaluate_real_corpus(
            embeddings_path=inputs["embeddings"],
            query_embeddings_path=inputs["query_embeddings"],
            ids_path=inputs["ids"],
            ground_truth_path=inputs["ground_truth"],
            metadata_path=inputs["metadata"],
            output_path=str(root / "real_corpus_runs" / "run_1.json"),
            backend="bruteforce",
            k=6,
            ks=(1, 3, 6),
            loops=2,
            threshold_recall=0.50,
            threshold_ndcg=0.50,
            threshold_p95_ms=500.0,
        )
    )
    stability = run_stability_study(
        embeddings_path=inputs["embeddings"],
        query_embeddings_path=inputs["query_embeddings"],
        ids_path=inputs["ids"],
        ground_truth_path=inputs["ground_truth"],
        metadata_path=inputs["metadata"],
        backend="bruteforce",
        k=6,
        ks=(1, 3, 6),
        loops=2,
        run_count=10,
        out_runs_path=str(root / "testing_runs" / "stability_runs_bruteforce_10.jsonl"),
        out_summary_path=str(root / "testing_runs" / "stability_summary_bruteforce_10.json"),
        threshold_recall=0.50,
        threshold_ndcg=0.50,
        threshold_p95_ms=500.0,
    )
    validate_stability_summary(stability)
    matrix = run_matrix(
        matrix=[{"name": "smoke", "n": 1000, "d": 32, "nq": 50, "k": 5}],
        mode="exact",
        warmup=1,
        loops=2,
        seed=7,
        min_flat_overlap=None,
        out_dir=str(root / "benchmark_matrix"),
        profile="dev",
        max_memory_mb=512.0,
    )
    validate_matrix_summary(matrix)

    benchmark_path = root / "benchmark_matrix" / "smoke.json"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        validate_benchmark_report(json.load(f))

    publishable = build_publishable_summary(
        matrix_summary_path=str(root / "benchmark_matrix" / "matrix_summary.json"),
        stability_summary_path=str(root / "testing_runs" / "stability_summary_bruteforce_10.json"),
        out_path=str(root / "benchmark_matrix" / "publishable_results.v1.json"),
    )
    validate_publishable_summary(publishable)
    gate_report = run_performance_gates(
        matrix_summary_path=str(root / "benchmark_matrix" / "matrix_summary.json"),
        stability_summary_path=str(root / "testing_runs" / "stability_summary_bruteforce_10.json"),
        output_path=str(root / "release_gates" / "performance_gate_report.v1.json"),
        min_recall=0.50,
        min_ndcg=0.50,
        max_latency_p95_ms=500.0,
        min_qps=0.0,
        min_flat_overlap=0.95,
    )

    return {
        "baseline_artifact": str(root / "rag_baseline_metrics.v1.json"),
        "real_corpus_artifact": str(root / "real_corpus_runs" / "run_1.json"),
        "stability_artifact": str(root / "testing_runs" / "stability_summary_bruteforce_10.json"),
        "matrix_artifact": str(root / "benchmark_matrix" / "matrix_summary.json"),
        "publishable_artifact": str(root / "benchmark_matrix" / "publishable_results.v1.json"),
        "performance_gate_artifact": str(root / "release_gates" / "performance_gate_report.v1.json"),
        "performance_gate_status": str(gate_report["status"]),
        "baseline_metrics_keys": ",".join(sorted(baseline["metrics"].keys())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-environment reproducibility smoke flow on synthetic inputs.")
    parser.add_argument("--output-dir", default="artifacts/repro_smoke")
    args = parser.parse_args()
    payload = run_smoke(args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

import pytest

from scripts.artifact_contracts import (
    validate_benchmark_report,
    validate_matrix_summary,
    validate_publishable_summary,
    validate_real_corpus_payload,
    validate_stability_summary,
)


def test_validate_real_corpus_payload_accepts_valid_shape():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "backend": "bruteforce",
        "k": 10,
        "ks": [1, 5, 10],
        "metrics": {"recall@10": 1.0},
        "performance": {"latency_p50_ms": 1.0, "latency_p95_ms": 2.0, "qps": 100.0},
        "topk_ids": [["a", "b"]],
        "runtime_seconds": 0.01,
        "checks": {"recall_gate": True},
        "environment": {"platform": "x", "python_version": "3.11", "machine": "x", "processor": "x"},
        "inputs": {"embeddings_path": "a", "query_embeddings_path": "b", "ids_path": "c", "ground_truth_path": "d"},
        "artifact_contract_version": "1.0",
    }
    validate_real_corpus_payload(payload)


def test_validate_benchmark_report_rejects_missing_results():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "config": {},
        "environment": {},
        "results": [],
        "artifact_contract_version": "1.0",
    }
    with pytest.raises(ValueError, match="contract_error"):
        validate_benchmark_report(payload)


def test_validate_matrix_stability_publishable_contracts():
    matrix_summary = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "protocol": {},
        "environment": {},
        "matrix": [{"name": "s"}],
        "backend_summary": {"bruteforce": {}},
        "runs_dir": "artifacts/benchmark_matrix",
        "artifact_contract_version": "1.0",
    }
    validate_matrix_summary(matrix_summary)

    stability_summary = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "run_count": 3,
        "backend": "bruteforce",
        "config": {},
        "environment": {},
        "performance_summary": {
            "latency_p50_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
            "latency_p95_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
            "qps": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
        },
        "metric_summary": {},
        "check_pass_rate": {},
        "input_files": {},
        "runs_path": "artifacts/testing_runs/runs.jsonl",
        "artifact_contract_version": "1.0",
    }
    validate_stability_summary(stability_summary)

    publishable = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "sources": {"a": "b"},
        "matrix_backend_summary": {},
        "stability_performance_summary": {},
        "stability_metric_summary": {},
        "protocol": {},
        "environment": {},
        "artifact_contract_version": "1.0",
    }
    validate_publishable_summary(publishable)

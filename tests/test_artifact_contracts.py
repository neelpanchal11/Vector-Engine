import pytest

from scripts.artifact_contracts import (
    validate_benchmark_report,
    validate_ingest_manifest,
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


def test_validate_real_corpus_payload_rejects_out_of_range_metrics():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "backend": "bruteforce",
        "k": 10,
        "ks": [1, 5, 10],
        "metrics": {"recall@10": 1.2},
        "performance": {"latency_p50_ms": 1.0, "latency_p95_ms": 2.0, "qps": 100.0},
        "topk_ids": [["a", "b"]],
        "runtime_seconds": 0.01,
        "checks": {"recall_gate": True},
        "environment": {"platform": "x", "python_version": "3.11", "machine": "x", "processor": "x"},
        "inputs": {"embeddings_path": "a", "query_embeddings_path": "b", "ids_path": "c", "ground_truth_path": "d"},
        "artifact_contract_version": "1.0",
    }
    with pytest.raises(ValueError, match="contract_error"):
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


def test_validate_benchmark_report_rejects_unknown_backend():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "config": {"mode": "exact", "k": 10, "min_flat_overlap": None},
        "environment": {},
        "results": [
            {
                "backend": "custom_backend",
                "qps": 1.0,
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
                "overlap_vs_bruteforce": 1.0,
                "memory_mb_estimate": 1.0,
            }
        ],
        "artifact_contract_version": "1.0",
    }
    with pytest.raises(ValueError, match="contract_error"):
        validate_benchmark_report(payload)


def test_validate_matrix_stability_publishable_contracts():
    matrix_summary = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "protocol": {
            "profile": "dev",
            "mode": "exact",
            "warmup": 2,
            "loops": 8,
            "seed": 7,
            "min_flat_overlap": None,
            "max_memory_mb": 1024.0,
            "matrix_size": 1,
        },
        "environment": {},
        "matrix": [{"name": "s", "n": 1000, "d": 32, "nq": 64, "k": 5}],
        "backend_summary": {
            "bruteforce": {
                "latency_p50_ms": {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0},
                "latency_p95_ms": {"mean": 2.0, "median": 2.0, "min": 2.0, "max": 2.0},
                "qps": {"mean": 100.0, "median": 100.0, "min": 90.0, "max": 110.0},
                "overlap_vs_bruteforce": {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0},
            }
        },
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
        "metric_summary": {"recall@1": {"mean": 1.0}},
        "check_pass_rate": {},
        "input_files": {},
        "runs_path": "artifacts/testing_runs/runs.jsonl",
        "artifact_contract_version": "1.0",
    }
    validate_stability_summary(stability_summary)

    publishable = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "sources": {"matrix_summary_path": "artifacts/benchmark_matrix/matrix_summary.json"},
        "matrix_backend_summary": {"bruteforce": {"qps": {"mean": 1.0}}},
        "stability_performance_summary": {"qps": {"mean": 1.0}},
        "stability_metric_summary": {"recall@1": {"mean": 1.0}},
        "protocol": {"matrix_protocol": {"mode": "exact"}, "stability_config": {"run_count": 3}},
        "environment": {"matrix": {"platform": "x"}, "stability": {"platform": "x"}},
        "artifact_contract_version": "1.0",
    }
    validate_publishable_summary(publishable)


def test_validate_ingest_manifest_accepts_valid_shape():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "input_jsonl": "data/source.jsonl",
        "output_dir": "artifacts/ingest",
        "record_count": 10,
        "embedding_dim": 64,
        "provider": "hash",
        "seed": 7,
        "fields": {
            "id_field": "id",
            "text_field": "text",
            "label_field": "label",
            "split_field": "split",
            "query_group_field": "query_group",
            "ground_truth_field": "ground_truth",
        },
        "artifacts": {
            "embeddings_path": "artifacts/ingest/embeddings.npy",
            "ids_path": "artifacts/ingest/ids.json",
            "metadata_path": "artifacts/ingest/metadata.json",
            "labels_path": "artifacts/ingest/labels.json",
            "splits_path": "artifacts/ingest/splits.json",
            "query_groups_path": "artifacts/ingest/query_groups.json",
            "ground_truth_path": "artifacts/ingest/ground_truth.json",
        },
        "environment": {"platform": "x", "python_version": "3.12", "machine": "arm64", "processor": "arm"},
        "artifact_contract_version": "1.0",
    }
    validate_ingest_manifest(payload)


def test_validate_ingest_manifest_rejects_invalid_record_count():
    payload = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "input_jsonl": "data/source.jsonl",
        "output_dir": "artifacts/ingest",
        "record_count": 0,
        "embedding_dim": 64,
        "provider": "hash",
        "seed": 7,
        "fields": {
            "id_field": "id",
            "text_field": "text",
            "label_field": None,
            "split_field": None,
            "query_group_field": None,
            "ground_truth_field": None,
        },
        "artifacts": {
            "embeddings_path": "artifacts/ingest/embeddings.npy",
            "ids_path": "artifacts/ingest/ids.json",
            "metadata_path": "artifacts/ingest/metadata.json",
            "labels_path": None,
            "splits_path": None,
            "query_groups_path": None,
            "ground_truth_path": None,
        },
        "environment": {"platform": "x", "python_version": "3.12", "machine": "arm64", "processor": "arm"},
        "artifact_contract_version": "1.0",
    }
    with pytest.raises(ValueError, match="contract_error"):
        validate_ingest_manifest(payload)

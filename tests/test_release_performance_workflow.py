import json

from scripts.performance_gates import run_performance_gates
from scripts.publishable_results import build_publishable_summary


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_performance_gates_passes_with_reasonable_metrics(tmp_path):
    matrix = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "protocol": {"mode": "exact", "warmup": 2, "loops": 5, "seed": 7, "matrix_size": 1, "min_flat_overlap": 0.99},
        "environment": {},
        "matrix": [{"name": "cfg", "n": 1000, "d": 32, "nq": 64, "k": 5}],
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
    stability = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "run_count": 3,
        "backend": "bruteforce",
        "config": {},
        "environment": {},
        "performance_summary": {
            "latency_p50_ms": {"mean": 1.0},
            "latency_p95_ms": {"mean": 2.0},
            "qps": {"mean": 150.0},
        },
        "metric_summary": {"recall@10": {"mean": 0.95}, "ndcg@10": {"mean": 0.92}},
        "check_pass_rate": {},
        "input_files": {},
        "runs_path": "runs.jsonl",
        "artifact_contract_version": "1.0",
    }
    matrix_path = tmp_path / "matrix.json"
    stability_path = tmp_path / "stability.json"
    out_path = tmp_path / "gates.json"
    _write(matrix_path, matrix)
    _write(stability_path, stability)
    report = run_performance_gates(
        matrix_summary_path=str(matrix_path),
        stability_summary_path=str(stability_path),
        output_path=str(out_path),
        min_recall=0.8,
        min_ndcg=0.8,
        max_latency_p95_ms=50.0,
        min_qps=50.0,
    )
    assert report["status"] == "pass"
    assert out_path.exists()


def test_build_publishable_summary_includes_release_delta(tmp_path):
    matrix = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "protocol": {"mode": "exact", "warmup": 2, "loops": 8, "seed": 7, "matrix_size": 1},
        "environment": {},
        "matrix": [{"name": "cfg", "n": 1000, "d": 32, "nq": 64, "k": 5}],
        "backend_summary": {
            "bruteforce": {
                "latency_p50_ms": {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0},
                "latency_p95_ms": {"mean": 2.0, "median": 2.0, "min": 2.0, "max": 2.0},
                "qps": {"mean": 100.0, "median": 100.0, "min": 95.0, "max": 105.0},
                "overlap_vs_bruteforce": {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0},
            }
        },
        "runs_dir": "artifacts/benchmark_matrix",
        "artifact_contract_version": "1.0",
    }
    stability = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "run_count": 2,
        "backend": "bruteforce",
        "config": {"run_count": 2},
        "environment": {},
        "performance_summary": {
            "latency_p50_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
            "latency_p95_ms": {"mean": 2.0, "median": 2.0, "std": 0.0, "cv": 0.0, "p02_5": 2.0, "p97_5": 2.0, "min": 2.0, "max": 2.0},
            "qps": {"mean": 120.0, "median": 120.0, "std": 0.0, "cv": 0.0, "p02_5": 120.0, "p97_5": 120.0, "min": 120.0, "max": 120.0},
        },
        "metric_summary": {"recall@1": {"mean": 0.9}},
        "check_pass_rate": {},
        "input_files": {},
        "runs_path": "runs.jsonl",
        "artifact_contract_version": "1.0",
    }
    previous = {
        "matrix_backend_summary": {
            "bruteforce": {
                "latency_p95_ms": {"mean": 3.0},
                "qps": {"mean": 100.0},
                "overlap_vs_bruteforce": {"mean": 1.0},
            }
        },
        "stability_performance_summary": {
            "latency_p95_ms": {"mean": 2.5},
            "qps": {"mean": 100.0},
        },
    }
    matrix_path = tmp_path / "matrix.json"
    stability_path = tmp_path / "stability.json"
    previous_path = tmp_path / "previous.json"
    out_path = tmp_path / "publishable.json"
    _write(matrix_path, matrix)
    _write(stability_path, stability)
    _write(previous_path, previous)
    payload = build_publishable_summary(
        matrix_summary_path=str(matrix_path),
        stability_summary_path=str(stability_path),
        out_path=str(out_path),
        previous_summary_path=str(previous_path),
    )
    assert "release_over_release_delta" in payload
    delta = payload["release_over_release_delta"]["stability_performance_delta"]
    assert delta["latency_p95_ms_mean_delta"] < 0.0

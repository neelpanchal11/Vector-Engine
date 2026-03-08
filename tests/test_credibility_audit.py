import json

from scripts.credibility_audit import run_audit


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_credibility_audit_runs(tmp_path):
    matrix = {
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
        "matrix": [{"name": "smoke", "n": 1000, "d": 32, "nq": 64, "k": 10}],
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
        "config": {"backend": "bruteforce", "k": 10, "loops": 5, "run_count": 2},
        "environment": {},
        "performance_summary": {
            "latency_p50_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
            "latency_p95_ms": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
            "qps": {"mean": 1.0, "median": 1.0, "std": 0.0, "cv": 0.0, "p02_5": 1.0, "p97_5": 1.0, "min": 1.0, "max": 1.0},
        },
        "metric_summary": {"recall@1": {"mean": 1.0}, "ndcg@1": {"mean": 1.0}},
        "check_pass_rate": {},
        "input_files": {"ids_path": "artifacts/mock/ids.json"},
        "runs_path": "artifacts/testing_runs/runs.jsonl",
        "artifact_contract_version": "1.0",
    }
    publishable = {
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "sources": {
            "matrix_summary_path": "artifacts/benchmark_matrix/matrix_summary.json",
            "stability_summary_path": "artifacts/testing_runs/stability_summary_bruteforce_200.json",
        },
        "matrix_backend_summary": matrix["backend_summary"],
        "stability_performance_summary": stability["performance_summary"],
        "stability_metric_summary": stability["metric_summary"],
        "protocol": {
            "matrix_protocol": matrix["protocol"],
            "stability_config": stability["config"],
        },
        "environment": {"matrix": {}, "stability": {}},
        "artifact_contract_version": "1.0",
    }
    real = {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "backend": "bruteforce",
        "k": 3,
        "ks": [1, 3],
        "metrics": {"recall@1": 1.0},
        "performance": {"latency_p50_ms": 1.0, "latency_p95_ms": 1.0, "qps": 1.0},
        "topk_ids": [["doc-0"]],
        "runtime_seconds": 0.1,
        "checks": {},
        "environment": {},
        "inputs": {"ids_path": "artifacts/mock/ids.json"},
        "artifact_contract_version": "1.0",
    }
    matrix_path = tmp_path / "matrix.json"
    stability_path = tmp_path / "stability.json"
    publishable_path = tmp_path / "publishable.json"
    real_path = tmp_path / "real.json"
    out_path = tmp_path / "audit.json"

    _write(matrix_path, matrix)
    _write(stability_path, stability)
    _write(publishable_path, publishable)
    _write(real_path, real)

    report = run_audit(
        matrix_summary_path=str(matrix_path),
        stability_summary_path=str(stability_path),
        publishable_summary_path=str(publishable_path),
        real_corpus_report_path=str(real_path),
        output_path=str(out_path),
    )
    assert report["status"] == "pass"
    optional = [c for c in report["checks"] if c["check"] == "optional_faiss_disclosure"]
    assert optional and optional[0]["status"] == "pass"
    assert out_path.exists()

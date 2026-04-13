import json

from scripts.matrix_profile_advisor import recommend_configs


def test_recommend_configs_returns_sorted_recommendations(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    matrix_summary = {
        "expanded_runs": ["cfg_a", "cfg_b"],
        "runs_dir": str(run_dir),
    }
    (tmp_path / "matrix_summary.json").write_text(json.dumps(matrix_summary), encoding="utf-8")
    for name, qps, p95, overlap in [("cfg_a", 1200.0, 10.0, 0.90), ("cfg_b", 1000.0, 8.0, 0.98)]:
        payload = {
            "results": [
                {"backend": "bruteforce", "qps": 500.0, "latency_p95_ms": 12.0, "overlap_vs_bruteforce": 1.0},
                {"backend": "faiss_ivf", "qps": qps, "latency_p95_ms": p95, "overlap_vs_bruteforce": overlap},
            ]
        }
        (run_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    out = recommend_configs(str(tmp_path / "matrix_summary.json"), top_n=1)
    assert len(out["recommendations"]) == 1
    assert out["recommendations"][0]["run"] in {"cfg_a", "cfg_b"}

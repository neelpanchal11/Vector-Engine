import json

import numpy as np

from scripts.profile_local import run_local_profile


def test_run_local_profile_tiny_config(tmp_path):
    rng = np.random.default_rng(7)
    xb = rng.normal(size=(64, 16)).astype(np.float32)
    xq = xb[:8] + 0.01 * rng.normal(size=(8, 16)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(64)]
    gt = [[ids[i]] for i in range(8)]
    metadata = [{"i": i} for i in range(64)]

    emb_path = tmp_path / "emb.npy"
    q_path = tmp_path / "q.npy"
    ids_path = tmp_path / "ids.json"
    gt_path = tmp_path / "gt.json"
    md_path = tmp_path / "md.json"
    np.save(emb_path, xb)
    np.save(q_path, xq)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    md_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = run_local_profile(
        embeddings=str(emb_path),
        query_embeddings=str(q_path),
        ids=str(ids_path),
        ground_truth=str(gt_path),
        metadata=str(md_path),
        output_dir=str(tmp_path / "out"),
        matrix=[{"name": "tiny", "n": 2000, "d": 32, "nq": 64, "k": 5}],
        stability_run_count=5,
        matrix_max_memory_mb=512.0,
    )
    assert "benchmark_backend_summary" in payload
    assert "stability_performance_summary" in payload
    assert payload["benchmark_backend_summary"].get("bruteforce") is not None
    assert payload["stability_summary_path"].endswith("stability_summary_bruteforce_5.json")
    matrix_summary = json.loads((tmp_path / "out" / "benchmark_matrix" / "matrix_summary.json").read_text(encoding="utf-8"))
    assert matrix_summary["protocol"]["min_flat_overlap"] is None
    assert (tmp_path / "out" / "benchmark_matrix" / "matrix_summary.json").exists()
    assert (tmp_path / "out" / "testing_runs" / "stability_summary_bruteforce_5.json").exists()
    assert (tmp_path / "out" / "profile_local_summary.json").exists()

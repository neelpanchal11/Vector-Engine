import json
import subprocess

import numpy as np
import pytest

from scripts.benchmark_matrix import _apply_memory_limit, _parse_matrix, run_matrix
from scripts.datasets import load_jsonl_bundle, load_numpy_bundle, with_deterministic_splits


def test_parse_matrix_profile_defaults_to_dev():
    matrix = _parse_matrix(None, profile="dev")
    assert len(matrix) >= 1
    assert {"name", "n", "d", "nq", "k"}.issubset(set(matrix[0].keys()))


def test_apply_memory_limit_filters_large_configs():
    matrix = [
        {"name": "a", "n": 1000, "d": 32, "nq": 64, "k": 5},
        {"name": "b", "n": 1000000, "d": 512, "nq": 1024, "k": 10},
    ]
    filtered = _apply_memory_limit(matrix, max_memory_mb=10.0)
    assert len(filtered) == 1
    assert filtered[0]["name"] == "a"
    with pytest.raises(ValueError, match="input_error"):
        _apply_memory_limit(matrix, max_memory_mb=0.0001)


def test_load_numpy_bundle(tmp_path):
    xb = np.random.randn(8, 16).astype(np.float32)
    emb = tmp_path / "emb.npy"
    ids = tmp_path / "ids.json"
    md = tmp_path / "md.json"
    np.save(emb, xb)
    ids.write_text(json.dumps([f"doc-{i}" for i in range(8)]), encoding="utf-8")
    md.write_text(json.dumps([{"i": i} for i in range(8)]), encoding="utf-8")
    bundle = load_numpy_bundle(str(emb), str(ids), str(md))
    assert bundle.embeddings.shape == (8, 16)
    assert len(bundle.ids) == 8
    assert bundle.metadata is not None


def test_load_numpy_bundle_with_extended_fields(tmp_path):
    xb = np.random.randn(4, 8).astype(np.float32)
    emb = tmp_path / "emb.npy"
    ids = tmp_path / "ids.json"
    labels = tmp_path / "labels.json"
    splits = tmp_path / "splits.json"
    gt = tmp_path / "gt.json"
    qg = tmp_path / "qg.json"
    np.save(emb, xb)
    ids.write_text(json.dumps([f"doc-{i}" for i in range(4)]), encoding="utf-8")
    labels.write_text(json.dumps(["a", "b", "a", "b"]), encoding="utf-8")
    splits.write_text(json.dumps(["train", "train", "val", "test"]), encoding="utf-8")
    gt.write_text(json.dumps([["doc-1"], ["doc-2"], ["doc-3"], ["doc-0"]]), encoding="utf-8")
    qg.write_text(json.dumps(["faq", "faq", "long_tail", "long_tail"]), encoding="utf-8")
    bundle = load_numpy_bundle(
        str(emb),
        str(ids),
        labels_path=str(labels),
        splits_path=str(splits),
        ground_truth_path=str(gt),
        query_groups_path=str(qg),
    )
    assert bundle.labels == ["a", "b", "a", "b"]
    assert bundle.splits == ["train", "train", "val", "test"]
    assert bundle.ground_truth is not None and bundle.ground_truth[0] == ["doc-1"]
    assert bundle.query_groups == ["faq", "faq", "long_tail", "long_tail"]


def test_load_jsonl_bundle(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": "a", "embedding": [0.1, 0.2], "group": "x"},
        {"id": "b", "embedding": [0.3, 0.4], "group": "y"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bundle = load_jsonl_bundle(str(path))
    assert bundle.embeddings.shape == (2, 2)
    assert bundle.ids == ["a", "b"]


def test_load_jsonl_bundle_with_extended_fields(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": "q1", "embedding": [0.1, 0.2], "label": 1, "split": "train", "ground_truth": ["d1"], "query_group": "faq"},
        {"id": "q2", "embedding": [0.3, 0.4], "label": 0, "split": "test", "ground_truth": ["d2"], "query_group": "tail"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bundle = load_jsonl_bundle(
        str(path),
        label_field="label",
        split_field="split",
        ground_truth_field="ground_truth",
        query_group_field="query_group",
    )
    assert bundle.labels == [1, 0]
    assert bundle.splits == ["train", "test"]
    assert bundle.ground_truth == [["d1"], ["d2"]]
    assert bundle.query_groups == ["faq", "tail"]


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"id": "dup", "embedding": [0.1, 0.2]},
        {"id": "dup", "embedding": [0.3, 0.4]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="ids must be unique"):
        load_jsonl_bundle(str(path))


def test_loader_rejects_non_finite_values(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "a", "embedding": [0.1, float("inf")]}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="finite values"):
        load_jsonl_bundle(str(path))


def test_with_deterministic_splits_reproducible(tmp_path):
    xb = np.random.randn(20, 8).astype(np.float32)
    emb = tmp_path / "emb.npy"
    ids = tmp_path / "ids.json"
    np.save(emb, xb)
    ids.write_text(json.dumps([f"doc-{i}" for i in range(20)]), encoding="utf-8")
    bundle = load_numpy_bundle(str(emb), str(ids))
    split_a = with_deterministic_splits(bundle, seed=42)
    split_b = with_deterministic_splits(bundle, seed=42)
    assert split_a.splits == split_b.splits
    assert split_a.splits is not None
    assert set(split_a.splits).issubset({"train", "val", "test"})


def test_run_matrix_succeeds_without_faiss_when_overlap_gate_disabled(tmp_path, monkeypatch):
    def fake_check_call(cmd):
        out_path = cmd[cmd.index("--output") + 1]
        payload = {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "config": {"mode": "exact", "k": 5, "min_flat_overlap": None},
            "environment": {"platform": "darwin", "python_version": "3.12", "machine": "arm64", "processor": "arm"},
            "results": [
                {
                    "backend": "bruteforce",
                    "qps": 1000.0,
                    "latency_p50_ms": 1.0,
                    "latency_p95_ms": 2.0,
                    "overlap_vs_bruteforce": 1.0,
                    "memory_mb_estimate": 10.0,
                }
            ],
            "artifact_contract_version": "1.0",
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    summary = run_matrix(
        matrix=[{"name": "tiny", "n": 2000, "d": 32, "nq": 64, "k": 5}],
        mode="exact",
        warmup=1,
        loops=1,
        seed=7,
        min_flat_overlap=None,
        out_dir=str(tmp_path / "out"),
        profile="dev",
        max_memory_mb=1024.0,
    )
    assert "bruteforce" in summary["backend_summary"]
    assert summary["protocol"]["min_flat_overlap"] is None


def test_run_matrix_fails_when_overlap_gate_enabled_without_faiss(tmp_path, monkeypatch):
    def fake_check_call(cmd):
        if "--min-flat-overlap" in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    with pytest.raises(subprocess.CalledProcessError):
        run_matrix(
            matrix=[{"name": "tiny", "n": 2000, "d": 32, "nq": 64, "k": 5}],
            mode="exact",
            warmup=1,
            loops=1,
            seed=7,
            min_flat_overlap=0.99,
            out_dir=str(tmp_path / "out"),
            profile="dev",
            max_memory_mb=1024.0,
        )

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.artifact_contracts import validate_ingest_manifest
from scripts.ingest_dataset import run_ingest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_run_ingest_emits_manifest_and_bundle(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "d1", "text": "hello world", "label": "faq", "split": "train", "query_group": "head", "ground_truth": ["d1"]},
        {"id": "d2", "text": "vector search", "label": "tail", "split": "test", "query_group": "tail", "ground_truth": ["d2"]},
    ]
    _write_jsonl(source, rows)
    out_dir = tmp_path / "ingest"
    manifest = run_ingest(
        input_jsonl=str(source),
        output_dir=str(out_dir),
        id_field="id",
        text_field="text",
        embedding_dim=16,
        seed=7,
        label_field="label",
        split_field="split",
        query_group_field="query_group",
        ground_truth_field="ground_truth",
    )
    validate_ingest_manifest(manifest)
    assert (out_dir / "embeddings.npy").exists()
    assert (out_dir / "ids.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "ingest_manifest.v1.json").exists()
    emb = np.load(out_dir / "embeddings.npy")
    assert emb.shape == (2, 16)


def test_run_ingest_is_deterministic_given_same_seed(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"id": "d1", "text": "same"}, {"id": "d2", "text": "rows"}]
    _write_jsonl(source, rows)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_ingest(
        input_jsonl=str(source),
        output_dir=str(out_a),
        id_field="id",
        text_field="text",
        embedding_dim=32,
        seed=99,
    )
    run_ingest(
        input_jsonl=str(source),
        output_dir=str(out_b),
        id_field="id",
        text_field="text",
        embedding_dim=32,
        seed=99,
    )
    emb_a = np.load(out_a / "embeddings.npy")
    emb_b = np.load(out_b / "embeddings.npy")
    assert np.array_equal(emb_a, emb_b)


def test_run_ingest_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"id": "dup", "text": "a"}, {"id": "dup", "text": "b"}]
    _write_jsonl(source, rows)
    with pytest.raises(ValueError, match="ids must be unique"):
        run_ingest(
            input_jsonl=str(source),
            output_dir=str(tmp_path / "out"),
            id_field="id",
            text_field="text",
            embedding_dim=8,
            seed=7,
        )


def test_run_ingest_honors_memory_cap(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"id": f"d{i}", "text": f"text-{i}"} for i in range(10)]
    _write_jsonl(source, rows)
    with pytest.raises(ValueError, match="max_memory_mb"):
        run_ingest(
            input_jsonl=str(source),
            output_dir=str(tmp_path / "out"),
            id_field="id",
            text_field="text",
            embedding_dim=64,
            seed=7,
            max_memory_mb=0.0001,
        )

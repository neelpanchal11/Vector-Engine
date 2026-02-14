import json

import numpy as np

from scripts.rag_real_corpus_eval import evaluate_real_corpus


def test_real_corpus_eval_writes_report(tmp_path):
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(8, 16)).astype(np.float32)
    xq = xb[:3] + 0.01 * rng.normal(size=(3, 16)).astype(np.float32)
    ids = [f"doc-{i}" for i in range(8)]
    gt = [[ids[i]] for i in range(3)]

    emb_path = tmp_path / "emb.npy"
    q_path = tmp_path / "q.npy"
    ids_path = tmp_path / "ids.json"
    gt_path = tmp_path / "gt.json"
    out_path = tmp_path / "report.json"
    np.save(emb_path, xb)
    np.save(q_path, xq)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    payload = evaluate_real_corpus(
        embeddings_path=str(emb_path),
        query_embeddings_path=str(q_path),
        ids_path=str(ids_path),
        ground_truth_path=str(gt_path),
        metadata_path=None,
        output_path=str(out_path),
        backend="bruteforce",
        k=3,
        ks=(1, 3),
        loops=2,
        threshold_recall=0.5,
        threshold_ndcg=0.5,
        threshold_p95_ms=50.0,
    )
    assert out_path.exists()
    assert payload["metrics"]["recall@1"] >= 0.66
    assert len(payload["topk_ids"]) == 3
    assert payload["runtime_seconds"] >= 0.0


def test_real_corpus_eval_validates_alignment(tmp_path):
    rng = np.random.default_rng(1)
    xb = rng.normal(size=(8, 16)).astype(np.float32)
    xq = rng.normal(size=(3, 16)).astype(np.float32)
    emb_path = tmp_path / "emb.npy"
    q_path = tmp_path / "q.npy"
    ids_path = tmp_path / "ids.json"
    gt_path = tmp_path / "gt.json"
    out_path = tmp_path / "report.json"
    np.save(emb_path, xb)
    np.save(q_path, xq)
    ids_path.write_text(json.dumps([f"doc-{i}" for i in range(7)]), encoding="utf-8")
    gt_path.write_text(json.dumps([["doc-0"], ["doc-1"], ["doc-2"]]), encoding="utf-8")

    try:
        evaluate_real_corpus(
            embeddings_path=str(emb_path),
            query_embeddings_path=str(q_path),
            ids_path=str(ids_path),
            ground_truth_path=str(gt_path),
            metadata_path=None,
            output_path=str(out_path),
            backend="bruteforce",
            k=3,
            ks=(1, 3),
            loops=1,
            threshold_recall=None,
            threshold_ndcg=None,
            threshold_p95_ms=None,
        )
        assert False, "expected alignment validation error"
    except ValueError as exc:
        assert "input_error" in str(exc)

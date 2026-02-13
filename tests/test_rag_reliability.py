import json

import numpy as np

from scripts.rag_baseline import ARTIFACT_VERSION, run_baseline
from vector_engine import VectorArray, VectorIndex
from vector_engine.training import mine_hard_negatives


def test_rag_baseline_metrics_and_artifact(tmp_path):
    payload = run_baseline(str(tmp_path), k=3)
    metrics = payload["metrics"]
    assert metrics["recall@1"] >= 0.50
    assert metrics["recall@3"] >= 0.75
    assert metrics["ndcg@3"] >= 0.70
    out_path = tmp_path / f"rag_baseline_metrics.{ARTIFACT_VERSION}.json"
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["artifact_version"] == ARTIFACT_VERSION


def test_hard_negative_smoke():
    xb = VectorArray.from_numpy(np.random.randn(20, 8).astype(np.float32), ids=np.arange(20), normalize=True)
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    anchors = xb.subset([0, 1, 2, 3])
    positives = np.asarray([0, 1, 2, 3], dtype=object)
    triplets = mine_hard_negatives(index, anchors, positives=positives, k=5)
    assert len(triplets.anchors) == 4
    assert len(triplets.negatives) == 4
    for p, n in zip(triplets.positives, triplets.negatives):
        assert p != n

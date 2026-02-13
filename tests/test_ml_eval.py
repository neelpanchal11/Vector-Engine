import numpy as np

from vector_engine import VectorArray, VectorIndex
from vector_engine.eval import ndcg_at_k, precision_at_k, recall_at_k, retrieval_report
from vector_engine.ml import knn_classify, knn_regress


def _index_for_knn():
    x = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [0.9, 0.8]], dtype=np.float32)
    return VectorIndex.create(VectorArray.from_numpy(x, ids=np.arange(4)), metric="l2", backend="bruteforce")


def test_knn_classify_and_regress():
    index = _index_for_knn()
    q = VectorArray.from_numpy(np.array([[0.1, 0.1], [0.95, 0.9]], dtype=np.float32))
    labels = np.array([0, 0, 1, 1])
    y_reg = np.array([0.0, 0.1, 1.0, 0.9], dtype=np.float32)
    pred_c = knn_classify(index, q, y_train=labels, k=3)
    pred_r = knn_regress(index, q, y_train=y_reg, k=3)
    assert pred_c.tolist() == [0, 1]
    assert pred_r[0] < 0.4
    assert pred_r[1] > 0.6


def test_retrieval_metrics():
    retrieved = np.array([["a", "b", "c"], ["m", "n", "o"]], dtype=object)
    gt = np.array([["a", "z"], ["x", "n"]], dtype=object)
    p = precision_at_k(retrieved, gt, k=2)
    r = recall_at_k(retrieved, gt, k=2)
    n = ndcg_at_k(retrieved, gt, k=3)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= n <= 1.0
    report = retrieval_report(retrieved, gt, ks=(1, 2, 3))
    assert "precision@1" in report
    assert "recall@2" in report
    assert "ndcg@3" in report

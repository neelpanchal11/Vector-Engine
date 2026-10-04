import numpy as np
import pytest

from vector_engine import Metric, VectorArray, VectorIndex
from vector_engine.backends import ivf as ivf_backend
from vector_engine.backends.ivf import IVFBackend


def _make_data(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    ids = np.arange(n)
    return VectorArray.from_numpy(x, ids=ids)


def test_ivf_search_shapes_and_self_hit():
    xb = _make_data()
    index = VectorIndex.create(xb, metric="cosine", backend="ivf", backend_config={"n_clusters": 8, "nprobe": 8})
    xq = xb.subset([0, 1, 2])
    result = index.search(xq, k=5)
    assert result.ids.shape == (3, 5)
    assert result.scores.shape == (3, 5)
    # nprobe == n_clusters means we scan everything, so self-match must be top-1.
    assert result.ids[0][0] == 0
    assert result.ids[1][0] == 1
    assert result.ids[2][0] == 2


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_ivf_batched_search_matches_probe_candidates(metric):
    xb = _make_data(n=300, d=12, seed=11)
    xq = _make_data(n=24, d=12, seed=12)
    index = VectorIndex.create(
        xb,
        metric=metric,
        backend="ivf",
        backend_config={"n_clusters": 12, "nprobe": 3, "random_state": 13},
    )
    result = index.search(xq, k=7)
    backend = index._backend
    assert backend is not None

    queries = xq.values
    if metric == "cosine":
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    if metric == "l2":
        probe_scores = np.sum((queries[:, None, :] - backend.centroids[None, :, :]) ** 2, axis=2)
        probe_clusters = np.argsort(probe_scores, axis=1)[:, : backend.nprobe]
    else:
        probe_scores = queries @ backend.centroids.T
        probe_clusters = np.argsort(-probe_scores, axis=1)[:, : backend.nprobe]

    expected_ids = np.full((len(queries), 7), -1, dtype=np.int64)
    expected_scores = np.full((len(queries), 7), np.nan, dtype=np.float32)
    for row, query in enumerate(queries):
        candidate_ids = np.flatnonzero(np.isin(backend.labels, probe_clusters[row]))
        if candidate_ids.size == 0:
            candidate_ids = np.arange(len(backend.xb))
        candidates = backend.xb[candidate_ids]
        if metric == "l2":
            scores = np.sum((query[None, :] - candidates) ** 2, axis=1)
            order = np.argsort(scores)
        else:
            scores = query @ candidates.T
            order = np.argsort(-scores)
        count = min(7, len(candidate_ids))
        selected = order[:count]
        expected_ids[row, :count] = candidate_ids[selected]
        expected_scores[row, :count] = scores[selected]

    assert np.array_equal(result.ids, expected_ids)
    assert np.allclose(result.scores, expected_scores, equal_nan=True)


def test_ivf_custom_metric_supports_batched_search():
    metric = Metric.custom(
        "negative_squared_l2",
        lambda a, b: -np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2),
        higher_is_better=True,
    )
    xb = _make_data(n=120, d=8, seed=14)
    xq = _make_data(n=10, d=8, seed=15)
    index = VectorIndex.create(
        xb,
        metric=metric,
        backend="ivf",
        backend_config={"n_clusters": 6, "nprobe": 2, "random_state": 16},
    )
    result = index.search(xq, k=5)
    backend = index._backend
    assert backend is not None

    probe_scores = metric.fn(xq.values, backend.centroids)
    probe_clusters = np.argsort(-probe_scores, axis=1)[:, : backend.nprobe]
    expected_ids = np.full((len(xq.values), 5), -1, dtype=np.int64)
    expected_scores = np.full((len(xq.values), 5), np.nan, dtype=np.float32)
    for row, query in enumerate(xq.values):
        candidate_ids = np.flatnonzero(np.isin(backend.labels, probe_clusters[row]))
        if candidate_ids.size == 0:
            candidate_ids = np.arange(len(backend.xb))
        scores = metric.fn(query[None, :], backend.xb[candidate_ids])[0]
        order = np.argsort(-scores)
        count = min(5, len(candidate_ids))
        selected = order[:count]
        expected_ids[row, :count] = candidate_ids[selected]
        expected_scores[row, :count] = scores[selected]

    assert np.array_equal(result.ids, expected_ids)
    assert np.allclose(result.scores, expected_scores, equal_nan=True)


def test_ivf_search_batches_scoring_by_cluster(monkeypatch):
    xb = _make_data(n=240, d=10, seed=17)
    xq = _make_data(n=48, d=10, seed=18)
    index = VectorIndex.create(
        xb,
        metric="l2",
        backend="ivf",
        backend_config={"n_clusters": 8, "nprobe": 3, "random_state": 19},
    )
    original = ivf_backend._pairwise_scores
    query_batch_sizes = []

    def observe_batches(a, b, metric):
        query_batch_sizes.append(a.shape[0])
        return original(a, b, metric)

    monkeypatch.setattr(ivf_backend, "_pairwise_scores", observe_batches)
    index.search(xq, k=5)

    assert max(query_batch_sizes) > 1
    assert len(query_batch_sizes) <= 9


def test_ivf_search_preserves_empty_probe_fallback_and_padding():
    backend = IVFBackend(
        xb=np.array([[0, 0], [1, 0], [10, 0]], dtype=np.float32),
        metric=Metric.l2(),
        centroids=np.array([[0, 0], [1, 0], [100, 0]], dtype=np.float32),
        labels=np.array([0, 1, 1], dtype=np.int64),
        n_clusters=3,
        nprobe=1,
    )

    scores, ids = backend.search(np.array([[100, 0], [0, 0]], dtype=np.float32), k=3)

    assert np.array_equal(ids[0, :3], np.array([2, 1, 0]))
    assert np.array_equal(ids[1, :1], np.array([0]))
    assert np.array_equal(ids[1, 1:], -np.ones(2, dtype=np.int64))
    assert np.isnan(scores[1, 1:]).all()


def test_ivf_recall_improves_with_higher_nprobe():
    xb = _make_data(n=500, d=32, seed=1)
    xq = _make_data(n=20, d=32, seed=2)

    exact = VectorIndex.create(xb, metric="l2", backend="bruteforce")
    exact_result = exact.search(xq, k=10)

    low_probe = VectorIndex.create(xb, metric="l2", backend="ivf", backend_config={"n_clusters": 20, "nprobe": 1, "random_state": 3})
    high_probe = VectorIndex.create(xb, metric="l2", backend="ivf", backend_config={"n_clusters": 20, "nprobe": 20, "random_state": 3})

    def recall_at_10(ivf_index):
        result = ivf_index.search(xq, k=10)
        hits = 0
        total = 0
        for row_exact, row_ivf in zip(exact_result.ids, result.ids):
            hits += len(set(row_exact.tolist()) & set(row_ivf.tolist()))
            total += len(row_exact)
        return hits / total

    low_recall = recall_at_10(low_probe)
    high_recall = recall_at_10(high_probe)
    assert high_recall >= low_recall
    assert high_recall == pytest.approx(1.0, abs=1e-6)


def test_ivf_add_and_save_load(tmp_path):
    xb = _make_data(n=100, d=8, seed=4)
    index = VectorIndex.create(xb, metric="cosine", backend="ivf", backend_config={"n_clusters": 5, "nprobe": 5})

    extra = VectorArray.from_numpy(
        np.random.default_rng(5).standard_normal((10, 8)).astype(np.float32),
        ids=np.arange(100, 110),
    )
    index.add(extra)
    assert index.runtime_stats()["count"] == 110

    path = str(tmp_path / "ivf_index")
    index.save(path)
    loaded = VectorIndex.load(path)

    xq = xb.subset([0])
    original_result = index.search(xq, k=3)
    loaded_result = loaded.search(xq, k=3)
    assert np.array_equal(original_result.ids, loaded_result.ids)
    assert np.allclose(original_result.scores, loaded_result.scores)


def test_ivf_rejects_invalid_config():
    xb = _make_data(n=10, d=4)
    with pytest.raises(ValueError, match="index_error"):
        VectorIndex.create(xb, metric="cosine", backend="ivf", backend_config={"n_clusters": 0})
    with pytest.raises(ValueError, match="index_error"):
        VectorIndex.create(xb, metric="cosine", backend="ivf", backend_config={"n_clusters": 100})

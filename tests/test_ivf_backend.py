import numpy as np
import pytest

from vector_engine import VectorArray, VectorIndex


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

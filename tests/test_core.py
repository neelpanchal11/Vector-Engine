import numpy as np

from vector_engine import VectorArray, VectorIndex


def test_vector_array_subset_by_ids():
    x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    va = VectorArray.from_numpy(x, ids=["a", "b", "c"], metadata=[{"v": 1}, {"v": 2}, {"v": 3}])
    sub = va.subset(["c", "a"])
    assert sub.shape == (2, 2)
    assert sub.ids.tolist() == ["c", "a"]
    assert sub.metadata[0]["v"] == 3


def test_index_search_and_metadata():
    xb = VectorArray.from_numpy(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32),
        ids=["x", "y", "z"],
        metadata=[{"label": "x"}, {"label": "y"}, {"label": "z"}],
        normalize=True,
    )
    xq = VectorArray.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32), normalize=True)
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    res = index.search(xq, k=2)
    assert res.ids.shape == (1, 2)
    assert res.ids[0, 0] in ("x", "z")
    assert isinstance(res.metadata[0][0], dict)


def test_index_save_and_load_roundtrip(tmp_path):
    xb = VectorArray.from_numpy(np.random.randn(20, 8).astype(np.float32), ids=np.arange(20))
    xq = VectorArray.from_numpy(np.random.randn(3, 8).astype(np.float32))
    index = VectorIndex.create(xb, metric="l2", backend="bruteforce")
    first = index.search(xq, k=4)
    path = tmp_path / "idx"
    index.save(str(path))
    loaded = VectorIndex.load(str(path))
    second = loaded.search(xq, k=4)
    assert np.array_equal(first.ids, second.ids)


def test_index_backend_capabilities_and_runtime_stats():
    xb = VectorArray.from_numpy(np.random.randn(16, 4).astype(np.float32), ids=np.arange(16))
    xq = VectorArray.from_numpy(np.random.randn(2, 4).astype(np.float32))
    index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
    caps = index.backend_capabilities()
    assert caps["supports_add"] is True
    assert caps["supports_ann_tuning"] is False
    index.search(xq, k=3)
    stats = index.runtime_stats()
    assert stats["search_calls"] == 1
    assert stats["last_search_k"] == 3
    assert stats["backend_stats"]["backend"] == "bruteforce"

import numpy as np
import pytest

from vector_engine import Metric, VectorArray, VectorIndex


def test_vector_array_rejects_empty_values():
    with pytest.raises(ValueError, match="vector_array_error"):
        VectorArray.from_numpy(np.empty((0, 8), dtype=np.float32))


def test_vector_array_rejects_non_string_int_ids():
    x = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="vector_array_error"):
        VectorArray.from_numpy(x, ids=[{"x": 1}, "ok"])


def test_subset_unknown_id_raises_clear_error():
    va = VectorArray.from_numpy(np.ones((3, 4), dtype=np.float32), ids=["a", "b", "c"])
    with pytest.raises(KeyError, match="vector_array_error"):
        va.subset(["missing"])


def test_index_search_requires_valid_k():
    xb = VectorArray.from_numpy(np.random.randn(10, 4).astype(np.float32), ids=np.arange(10))
    xq = VectorArray.from_numpy(np.random.randn(2, 4).astype(np.float32))
    index = VectorIndex.create(xb, metric="l2", backend="bruteforce")
    with pytest.raises(TypeError, match="index_error"):
        index.search(xq, k=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index_error"):
        index.search(xq, k=0)


def test_metric_rejects_invalid_name():
    with pytest.raises(ValueError, match="metric_error"):
        Metric.from_value("does-not-exist")

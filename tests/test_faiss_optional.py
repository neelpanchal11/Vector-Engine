import numpy as np
import pytest

from vector_engine import VectorArray, VectorIndex


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_faiss(), reason="faiss is not installed")
def test_faiss_backend_basic():
    xb = VectorArray.from_numpy(np.random.randn(100, 16).astype(np.float32), normalize=True)
    xq = VectorArray.from_numpy(np.random.randn(5, 16).astype(np.float32), normalize=True)
    index = VectorIndex.create(
        xb,
        metric="cosine",
        backend="faiss",
        backend_config={"index_factory": "Flat"},
    )
    res = index.search(xq, k=4)
    assert res.ids.shape == (5, 4)
    assert res.scores.shape == (5, 4)

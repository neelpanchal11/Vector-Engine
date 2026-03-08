import numpy as np

from benchmarks.compare_bruteforce_vs_faiss import timed_search
from vector_engine import VectorArray, VectorIndex


def test_perf_smoke_bruteforce_tiny():
    rng = np.random.default_rng(7)
    xb = rng.normal(size=(512, 64)).astype(np.float32)
    xq = rng.normal(size=(32, 64)).astype(np.float32)
    base = VectorArray.from_numpy(xb, ids=np.arange(xb.shape[0]), normalize=True)
    query = VectorArray.from_numpy(xq, normalize=True)
    idx = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    stats, ids = timed_search(idx, query, k=5, warmup=1, loops=2)
    assert ids.shape == (32, 5)
    assert stats["qps"] > 0.0
    assert stats["latency_p95_ms"] < 5000.0

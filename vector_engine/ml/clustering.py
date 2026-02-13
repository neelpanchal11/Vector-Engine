from __future__ import annotations

import numpy as np

from vector_engine.array import VectorArray


def kmeans(vectors: VectorArray, n_clusters: int, *, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simple wrapper around sklearn KMeans for VectorArray."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(vectors.values)
    centers = model.cluster_centers_.astype(np.float32)
    return labels.astype(np.int64), centers

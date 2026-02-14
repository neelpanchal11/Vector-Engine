from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_engine.array import VectorArray


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


def kmeans(
    vectors: VectorArray,
    n_clusters: int,
    *,
    random_state: int = 0,
    max_iter: int = 300,
) -> KMeansResult:
    """Simple wrapper around sklearn KMeans for VectorArray."""
    from sklearn.cluster import KMeans

    if not isinstance(random_state, int):
        raise TypeError("ml_error: random_state must be an int")
    if vectors.values.ndim != 2:
        raise ValueError("ml_error: vectors must be a 2D matrix")
    if vectors.values.shape[1] == 0:
        raise ValueError("ml_error: vectors must have at least one feature column")
    if not np.all(np.isfinite(vectors.values)):
        raise ValueError("ml_error: vectors must contain only finite values")
    if n_clusters <= 1:
        raise ValueError("ml_error: n_clusters must be > 1")
    if n_clusters > vectors.values.shape[0]:
        raise ValueError("ml_error: n_clusters cannot exceed number of rows")
    if max_iter <= 0:
        raise ValueError("ml_error: max_iter must be > 0")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
        max_iter=max_iter,
    )
    labels = model.fit_predict(vectors.values)
    centers = model.cluster_centers_.astype(np.float32)
    return KMeansResult(
        labels=labels.astype(np.int64),
        centers=centers,
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )

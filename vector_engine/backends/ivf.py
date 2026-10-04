from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from vector_engine.metric import Metric


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _pairwise_scores(a: np.ndarray, b: np.ndarray, metric: Metric) -> np.ndarray:
    if metric.fn is not None:
        return metric.fn(a, b)
    if metric.name in ("ip", "cosine"):
        return a @ b.T
    if metric.name == "l2":
        diff = a[:, None, :] - b[None, :, :]
        return np.sum(diff * diff, axis=2)
    raise ValueError(f"unsupported metric for ivf: {metric.name}")


def _top_k_indices(scores: np.ndarray, k: int, higher_is_better: bool) -> np.ndarray:
    ranked_scores = -scores if higher_is_better else scores
    top = np.argpartition(ranked_scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(ranked_scores, top, axis=1)
    order = np.argsort(top_scores, axis=1)
    return np.take_along_axis(top, order, axis=1)


def _kmeans_lloyd(x: np.ndarray, n_clusters: int, *, max_iter: int, seed: int) -> np.ndarray:
    """Minimal Lloyd's-algorithm k-means, pure numpy. Returns centroids (n_clusters, d)."""
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(x.shape[0], size=n_clusters, replace=False)
    centroids = x[init_idx].copy()

    for _ in range(max_iter):
        diff = x[:, None, :] - centroids[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        new_centroids = centroids.copy()
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = x[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


@dataclass
class IVFBackend:
    """Inverted-file ANN backend: coarse-quantize with k-means, probe nearest clusters at search time."""

    name: str = "ivf"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "supports_delete": False,
            "supports_custom_metric": True,
            "supports_persistence": True,
            "supports_add": True,
            "supports_ann_tuning": True,
        }
    )
    xb: np.ndarray | None = None
    metric: Metric | None = None
    centroids: np.ndarray | None = None
    labels: np.ndarray | None = None
    n_clusters: int = 0
    nprobe: int = 1

    def build(self, xb: np.ndarray, metric: Metric, config: dict) -> None:
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        if metric.name == "cosine":
            arr = _normalize_l2(arr)

        n_clusters = int(config.get("n_clusters", max(1, int(np.sqrt(arr.shape[0])))))
        if n_clusters <= 0:
            raise ValueError("index_error: n_clusters must be > 0")
        if n_clusters > arr.shape[0]:
            raise ValueError("index_error: n_clusters cannot exceed number of vectors")
        max_iter = int(config.get("max_iter", 25))
        seed = int(config.get("random_state", 0))
        nprobe = int(config.get("nprobe", max(1, n_clusters // 8)))
        if nprobe <= 0:
            raise ValueError("index_error: nprobe must be > 0")

        centroids = _kmeans_lloyd(arr, n_clusters, max_iter=max_iter, seed=seed)
        centroid_scores = _pairwise_scores(arr, centroids, metric)
        labels = np.argmax(centroid_scores, axis=1) if metric.higher_is_better else np.argmin(centroid_scores, axis=1)

        self.xb = arr
        self.metric = metric
        self.centroids = centroids
        self.labels = labels.astype(np.int64)
        self.n_clusters = n_clusters
        self.nprobe = min(nprobe, n_clusters)

    def add(self, xb: np.ndarray) -> np.ndarray:
        if self.xb is None or self.metric is None or self.centroids is None:
            raise RuntimeError("backend is not built")
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        if self.metric.name == "cosine":
            arr = _normalize_l2(arr)
        start = self.xb.shape[0]
        centroid_scores = _pairwise_scores(arr, self.centroids, self.metric)
        new_labels = (
            np.argmax(centroid_scores, axis=1) if self.metric.higher_is_better else np.argmin(centroid_scores, axis=1)
        )
        self.xb = np.vstack([self.xb, arr])
        self.labels = np.concatenate([self.labels, new_labels.astype(np.int64)])
        return np.arange(start, start + arr.shape[0], dtype=np.int64)

    def search(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.xb is None or self.metric is None or self.centroids is None or self.labels is None:
            raise RuntimeError("backend is not built")
        queries = np.ascontiguousarray(xq, dtype=np.float32)
        if self.metric.name == "cosine":
            queries = _normalize_l2(queries)

        centroid_scores = _pairwise_scores(queries, self.centroids, self.metric)
        nprobe = min(self.nprobe, self.n_clusters)
        if self.metric.higher_is_better:
            probe_clusters = np.argsort(-centroid_scores, axis=1)[:, :nprobe]
        else:
            probe_clusters = np.argsort(centroid_scores, axis=1)[:, :nprobe]

        n_queries = queries.shape[0]
        k_eff = min(k, self.xb.shape[0])
        worst_score = -np.inf if self.metric.higher_is_better else np.inf
        out_scores = np.full((n_queries, k_eff), worst_score, dtype=np.float32)
        out_ids = np.full((n_queries, k_eff), -1, dtype=np.int64)
        candidate_counts = np.zeros(n_queries, dtype=np.int64)

        for cluster in range(self.n_clusters):
            query_idx = np.flatnonzero(np.any(probe_clusters == cluster, axis=1))
            candidate_idx = np.flatnonzero(self.labels == cluster)
            if query_idx.size == 0 or candidate_idx.size == 0:
                continue

            scores = _pairwise_scores(queries[query_idx], self.xb[candidate_idx], self.metric)
            this_k = min(k_eff, candidate_idx.size)
            top = _top_k_indices(scores, this_k, self.metric.higher_is_better)
            cluster_scores = np.take_along_axis(scores, top, axis=1)
            cluster_ids = candidate_idx[top]
            candidate_counts[query_idx] += candidate_idx.size
            merged_scores = np.concatenate((out_scores[query_idx], cluster_scores), axis=1)
            merged_ids = np.concatenate((out_ids[query_idx], cluster_ids), axis=1)
            merged_top = _top_k_indices(merged_scores, k_eff, self.metric.higher_is_better)
            out_scores[query_idx] = np.take_along_axis(merged_scores, merged_top, axis=1)
            out_ids[query_idx] = np.take_along_axis(merged_ids, merged_top, axis=1)

        fallback_idx = np.flatnonzero(candidate_counts == 0)
        if fallback_idx.size:
            scores = _pairwise_scores(queries[fallback_idx], self.xb, self.metric)
            top = _top_k_indices(scores, k_eff, self.metric.higher_is_better)
            out_scores[fallback_idx] = np.take_along_axis(scores, top, axis=1)
            out_ids[fallback_idx] = top
            candidate_counts[fallback_idx] = self.xb.shape[0]

        invalid = np.arange(k_eff)[None, :] >= np.minimum(candidate_counts, k_eff)[:, None]
        out_scores[invalid] = np.nan
        out_ids[invalid] = -1

        return out_scores, out_ids

    def save(self, path: str) -> None:
        if self.xb is None or self.metric is None or self.centroids is None or self.labels is None:
            raise RuntimeError("backend is not built")
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self.xb)
        np.save(os.path.join(path, "centroids.npy"), self.centroids)
        np.save(os.path.join(path, "labels.npy"), self.labels)
        with open(os.path.join(path, "backend_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "metric_name": self.metric.name,
                    "higher_is_better": self.metric.higher_is_better,
                    "has_custom_metric": self.metric.fn is not None,
                    "n_clusters": self.n_clusters,
                    "nprobe": self.nprobe,
                },
                f,
            )

    def get_runtime_stats(self) -> dict[str, float | int | str]:
        count = int(self.xb.shape[0]) if self.xb is not None else 0
        dim = int(self.xb.shape[1]) if self.xb is not None else 0
        metric_name = self.metric.name if self.metric is not None else "unknown"
        return {
            "backend": self.name,
            "count": count,
            "dim": dim,
            "metric": metric_name,
            "n_clusters": int(self.n_clusters),
            "nprobe": int(self.nprobe),
            "vector_bytes": int(self.xb.nbytes) if self.xb is not None else 0,
        }

    @classmethod
    def load(cls, path: str) -> "IVFBackend":
        with open(os.path.join(path, "backend_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["has_custom_metric"]:
            raise ValueError("cannot load custom ivf metric without explicit code hook")
        backend = cls()
        backend.metric = Metric(name=meta["metric_name"], higher_is_better=meta["higher_is_better"])
        backend.xb = np.load(os.path.join(path, "vectors.npy"))
        backend.centroids = np.load(os.path.join(path, "centroids.npy"))
        backend.labels = np.load(os.path.join(path, "labels.npy"))
        backend.n_clusters = int(meta["n_clusters"])
        backend.nprobe = int(meta["nprobe"])
        return backend

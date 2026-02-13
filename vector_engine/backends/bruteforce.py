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


@dataclass
class BruteForceBackend:
    name: str = "bruteforce"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "supports_delete": False,
            "supports_custom_metric": True,
            "supports_persistence": True,
        }
    )
    xb: np.ndarray | None = None
    metric: Metric | None = None

    def build(self, xb: np.ndarray, metric: Metric, config: dict) -> None:
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        self.metric = metric
        if metric.name == "cosine":
            arr = _normalize_l2(arr)
        self.xb = arr

    def add(self, xb: np.ndarray) -> np.ndarray:
        if self.xb is None or self.metric is None:
            raise RuntimeError("backend is not built")
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        if self.metric.name == "cosine":
            arr = _normalize_l2(arr)
        start = self.xb.shape[0]
        self.xb = np.vstack([self.xb, arr])
        return np.arange(start, start + arr.shape[0], dtype=np.int64)

    def search(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.xb is None or self.metric is None:
            raise RuntimeError("backend is not built")
        queries = np.ascontiguousarray(xq, dtype=np.float32)
        if self.metric.name == "cosine":
            queries = _normalize_l2(queries)

        if self.metric.fn is not None:
            # Metric fn returns pairwise matrix, shape (n_queries, n_db).
            scores = self.metric.fn(queries, self.xb)
        elif self.metric.name == "ip" or self.metric.name == "cosine":
            scores = queries @ self.xb.T
        elif self.metric.name == "l2":
            diff = queries[:, None, :] - self.xb[None, :, :]
            scores = np.sum(diff * diff, axis=2)
        else:
            raise ValueError(f"unsupported metric for brute force: {self.metric.name}")

        k = min(k, self.xb.shape[0])
        if self.metric.higher_is_better:
            idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(scores.shape[0])[:, None]
            val = scores[row, idx]
            order = np.argsort(-val, axis=1)
        else:
            idx = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]
            row = np.arange(scores.shape[0])[:, None]
            val = scores[row, idx]
            order = np.argsort(val, axis=1)
        sorted_idx = np.take_along_axis(idx, order, axis=1)
        sorted_scores = np.take_along_axis(scores, sorted_idx, axis=1)
        return sorted_scores, sorted_idx.astype(np.int64)

    def save(self, path: str) -> None:
        if self.xb is None or self.metric is None:
            raise RuntimeError("backend is not built")
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self.xb)
        with open(os.path.join(path, "backend_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "metric_name": self.metric.name,
                    "higher_is_better": self.metric.higher_is_better,
                    "has_custom_metric": self.metric.fn is not None,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BruteForceBackend":
        with open(os.path.join(path, "backend_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["has_custom_metric"]:
            raise ValueError("cannot load custom brute force metric without explicit code hook")
        xb = np.load(os.path.join(path, "vectors.npy"))
        backend = cls()
        backend.metric = Metric(name=meta["metric_name"], higher_is_better=meta["higher_is_better"])
        backend.xb = xb
        return backend

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
class FaissBackend:
    name: str = "faiss"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "supports_delete": False,
            "supports_custom_metric": False,
            "supports_persistence": True,
        }
    )
    _index: object | None = None
    _metric: Metric | None = None
    _count: int = 0
    _config: dict | None = None

    def _faiss(self) -> object:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - import path depends on env
            raise ImportError(
                "Faiss backend requested but faiss is not installed. "
                "Install with `pip install faiss-cpu`."
            ) from exc
        return faiss

    def build(self, xb: np.ndarray, metric: Metric, config: dict) -> None:
        if metric.fn is not None:
            raise ValueError("faiss backend does not support custom metric functions")
        faiss = self._faiss()
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        if metric.name == "cosine":
            arr = _normalize_l2(arr)
            metric_for_faiss = "ip"
        else:
            metric_for_faiss = metric.name

        index_factory = config.get("index_factory", "Flat")
        if metric_for_faiss == "l2":
            index = faiss.index_factory(arr.shape[1], index_factory, faiss.METRIC_L2)
        elif metric_for_faiss == "ip":
            index = faiss.index_factory(arr.shape[1], index_factory, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"unsupported metric for faiss: {metric.name}")

        if hasattr(index, "train") and not index.is_trained:
            index.train(arr)
        index.add(arr)

        nprobe = config.get("nprobe")
        if nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)

        self._index = index
        self._metric = metric
        self._count = arr.shape[0]
        self._config = dict(config)

    def add(self, xb: np.ndarray) -> np.ndarray:
        if self._index is None or self._metric is None:
            raise RuntimeError("backend is not built")
        arr = np.ascontiguousarray(xb, dtype=np.float32)
        if self._metric.name == "cosine":
            arr = _normalize_l2(arr)
        start = self._count
        self._index.add(arr)
        self._count += arr.shape[0]
        return np.arange(start, self._count, dtype=np.int64)

    def search(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._metric is None:
            raise RuntimeError("backend is not built")
        queries = np.ascontiguousarray(xq, dtype=np.float32)
        if self._metric.name == "cosine":
            queries = _normalize_l2(queries)
        scores, internal_ids = self._index.search(queries, k)
        return scores, internal_ids.astype(np.int64)

    def save(self, path: str) -> None:
        if self._index is None or self._metric is None:
            raise RuntimeError("backend is not built")
        faiss = self._faiss()
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "backend_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "metric_name": self._metric.name,
                    "higher_is_better": self._metric.higher_is_better,
                    "config": self._config or {},
                    "count": self._count,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FaissBackend":
        backend = cls()
        faiss = backend._faiss()
        with open(os.path.join(path, "backend_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        backend._index = faiss.read_index(os.path.join(path, "faiss.index"))
        backend._metric = Metric(name=meta["metric_name"], higher_is_better=meta["higher_is_better"])
        backend._config = meta.get("config", {})
        backend._count = int(meta.get("count", 0))
        return backend

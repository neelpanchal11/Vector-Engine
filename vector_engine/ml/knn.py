from __future__ import annotations

import numpy as np

from vector_engine.array import VectorArray
from vector_engine.index import VectorIndex


def _safe_weights(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    eps = 1e-8
    if higher_is_better:
        shifted = scores - np.min(scores, axis=1, keepdims=True) + eps
        return shifted
    return 1.0 / (scores + eps)


def knn_classify(
    index: VectorIndex,
    queries: VectorArray,
    *,
    y_train: np.ndarray,
    k: int = 5,
    weights: str = "distance",
) -> np.ndarray:
    res = index.search(queries, k=k, return_metadata=False)
    neighbor_ids = res.ids
    y = np.asarray(y_train)
    out = []
    w = _safe_weights(res.scores, index.metric.higher_is_better) if weights == "distance" else None

    for i in range(neighbor_ids.shape[0]):
        ids = [int(x) for x in neighbor_ids[i]]
        labels = y[ids]
        if weights is None or weights == "uniform":
            values, counts = np.unique(labels, return_counts=True)
            out.append(values[np.argmax(counts)])
        elif weights == "distance":
            scores = {}
            for lbl, weight in zip(labels, w[i]):
                scores[lbl] = scores.get(lbl, 0.0) + float(weight)
            out.append(max(scores.items(), key=lambda item: item[1])[0])
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
    return np.asarray(out)


def knn_regress(
    index: VectorIndex,
    queries: VectorArray,
    *,
    y_train: np.ndarray,
    k: int = 5,
    weights: str = "distance",
) -> np.ndarray:
    res = index.search(queries, k=k, return_metadata=False)
    neighbor_ids = res.ids
    y = np.asarray(y_train, dtype=np.float32)
    w = _safe_weights(res.scores, index.metric.higher_is_better) if weights == "distance" else None

    out = np.zeros(neighbor_ids.shape[0], dtype=np.float32)
    for i in range(neighbor_ids.shape[0]):
        ids = [int(x) for x in neighbor_ids[i]]
        vals = y[ids]
        if weights is None or weights == "uniform":
            out[i] = float(np.mean(vals))
        elif weights == "distance":
            ww = w[i]
            out[i] = float(np.sum(vals * ww) / np.sum(ww))
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
    return out

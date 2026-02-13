from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _to_set_iter(gt: np.ndarray | Iterable[Iterable[object]]) -> list[set[object]]:
    if isinstance(gt, np.ndarray):
        return [set(row.tolist()) for row in gt]
    return [set(row) for row in gt]


def precision_at_k(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray, k: int) -> float:
    retrieved = retrieved_ids[:, :k]
    gt = _to_set_iter(ground_truth_ids)
    vals = []
    for i in range(retrieved.shape[0]):
        hit = sum(1 for item in retrieved[i].tolist() if item in gt[i])
        vals.append(hit / float(k))
    return float(np.mean(vals))


def recall_at_k(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray, k: int) -> float:
    retrieved = retrieved_ids[:, :k]
    gt = _to_set_iter(ground_truth_ids)
    vals = []
    for i in range(retrieved.shape[0]):
        if len(gt[i]) == 0:
            vals.append(0.0)
            continue
        hit = sum(1 for item in retrieved[i].tolist() if item in gt[i])
        vals.append(hit / float(len(gt[i])))
    return float(np.mean(vals))


def ndcg_at_k(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray, k: int) -> float:
    retrieved = retrieved_ids[:, :k]
    gt = _to_set_iter(ground_truth_ids)
    vals = []
    for i in range(retrieved.shape[0]):
        dcg = 0.0
        for rank, item in enumerate(retrieved[i].tolist(), start=1):
            rel = 1.0 if item in gt[i] else 0.0
            dcg += rel / math.log2(rank + 1)
        ideal_hits = min(k, len(gt[i]))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        vals.append(0.0 if idcg == 0 else dcg / idcg)
    return float(np.mean(vals))


def retrieval_report(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics for multiple k values."""
    report: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("eval_error: each k must be > 0")
        report[f"precision@{k}"] = precision_at_k(retrieved_ids, ground_truth_ids, k)
        report[f"recall@{k}"] = recall_at_k(retrieved_ids, ground_truth_ids, k)
        report[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ground_truth_ids, k)
    return report

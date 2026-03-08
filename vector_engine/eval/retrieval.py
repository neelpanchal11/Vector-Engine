from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _normalize_ground_truth(
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    n_queries: int,
) -> list[set[object]]:
    if isinstance(ground_truth_ids, np.ndarray):
        if ground_truth_ids.ndim == 2:
            rows = [ground_truth_ids[i].tolist() for i in range(ground_truth_ids.shape[0])]
        elif ground_truth_ids.ndim == 1:
            rows = ground_truth_ids.tolist()
        else:
            raise ValueError("eval_error: ground_truth_ids must be 1D or 2D")
    else:
        rows = list(ground_truth_ids)

    if len(rows) != n_queries:
        raise ValueError("eval_error: retrieved_ids and ground_truth_ids must have same number of queries")

    normalized: list[set[object]] = []
    for i, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
            raise TypeError(f"eval_error: ground_truth_ids row {i} must be an iterable of IDs")
        normalized.append(set(row))
    return normalized


def _ensure_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"eval_error: {name} must be a 2D array")
    if arr.shape[0] == 0:
        raise ValueError(f"eval_error: {name} cannot be empty")
    return arr


def _validate_inputs(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    k: int,
) -> tuple[np.ndarray, list[set[object]]]:
    if not isinstance(k, int):
        raise TypeError("eval_error: k must be an int")
    if k <= 0:
        raise ValueError("eval_error: k must be > 0")
    retrieved = _ensure_2d("retrieved_ids", retrieved_ids)
    if k > retrieved.shape[1]:
        raise ValueError("eval_error: k cannot exceed retrieved_ids width")
    return retrieved, _normalize_ground_truth(ground_truth_ids, n_queries=retrieved.shape[0])


def _per_query_precision(retrieved_row: Sequence[object], gt_row: set[object], k: int) -> float:
    hit = sum(1 for item in list(retrieved_row)[:k] if item in gt_row)
    return hit / float(k)


def _per_query_recall(retrieved_row: Sequence[object], gt_row: set[object], k: int) -> float:
    if len(gt_row) == 0:
        return 0.0
    hit = sum(1 for item in list(retrieved_row)[:k] if item in gt_row)
    return hit / float(len(gt_row))


def _per_query_ndcg(retrieved_row: Sequence[object], gt_row: set[object], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(list(retrieved_row)[:k], start=1):
        rel = 1.0 if item in gt_row else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal_hits = min(k, len(gt_row))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def precision_at_k(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    k: int,
) -> float:
    retrieved, gt = _validate_inputs(retrieved_ids, ground_truth_ids, k)
    vals = [_per_query_precision(retrieved[i].tolist(), gt[i], k) for i in range(retrieved.shape[0])]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def recall_at_k(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    k: int,
) -> float:
    retrieved, gt = _validate_inputs(retrieved_ids, ground_truth_ids, k)
    vals = [_per_query_recall(retrieved[i].tolist(), gt[i], k) for i in range(retrieved.shape[0])]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def ndcg_at_k(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    k: int,
) -> float:
    retrieved, gt = _validate_inputs(retrieved_ids, ground_truth_ids, k)
    vals = [_per_query_ndcg(retrieved[i].tolist(), gt[i], k) for i in range(retrieved.shape[0])]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def retrieval_report(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics for multiple k values."""
    if len(ks) == 0:
        raise ValueError("eval_error: ks cannot be empty")
    report: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("eval_error: each k must be > 0")
        report[f"precision@{k}"] = precision_at_k(retrieved_ids, ground_truth_ids, k)
        report[f"recall@{k}"] = recall_at_k(retrieved_ids, ground_truth_ids, k)
        report[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ground_truth_ids, k)
    return report


def retrieval_report_detailed(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    ks: Sequence[int] = (1, 5, 10),
    *,
    include_per_query: bool = True,
    include_error_buckets: bool = True,
) -> dict[str, object]:
    """Compute aggregate and per-query retrieval metrics."""
    retrieved = _ensure_2d("retrieved_ids", retrieved_ids)
    summary = retrieval_report(retrieved_ids, ground_truth_ids, ks=ks)
    payload: dict[str, object] = {"summary": summary}
    if include_per_query:
        per_query: list[dict[str, float]] = []
        gt = _normalize_ground_truth(ground_truth_ids, n_queries=retrieved.shape[0])
        for i in range(retrieved.shape[0]):
            row: dict[str, float] = {}
            for k in ks:
                row[f"precision@{k}"] = _per_query_precision(retrieved[i].tolist(), gt[i], k)
                row[f"recall@{k}"] = _per_query_recall(retrieved[i].tolist(), gt[i], k)
                row[f"ndcg@{k}"] = _per_query_ndcg(retrieved[i].tolist(), gt[i], k)
            per_query.append(row)
        payload["per_query"] = per_query
    if include_error_buckets:
        gt = _normalize_ground_truth(ground_truth_ids, n_queries=retrieved.shape[0])
        buckets: dict[str, float] = {}
        for k in ks:
            zero_hit = 0
            perfect_recall = 0
            no_ground_truth = 0
            for i in range(retrieved.shape[0]):
                gt_row = gt[i]
                if len(gt_row) == 0:
                    no_ground_truth += 1
                    continue
                hit = sum(1 for item in retrieved[i].tolist()[:k] if item in gt_row)
                if hit == 0:
                    zero_hit += 1
                if hit == len(gt_row):
                    perfect_recall += 1
            n = float(retrieved.shape[0])
            buckets[f"zero_hit_rate@{k}"] = zero_hit / n
            buckets[f"perfect_recall_rate@{k}"] = perfect_recall / n
            buckets[f"no_ground_truth_rate@{k}"] = no_ground_truth / n
        payload["error_buckets"] = buckets
    return payload


def _bootstrap_ci(
    vals: np.ndarray,
    *,
    confidence: float,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    if vals.size == 0:
        raise ValueError("eval_error: bootstrap values cannot be empty")
    if n_bootstrap <= 0:
        raise ValueError("eval_error: n_bootstrap must be > 0")
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("eval_error: confidence must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    means = np.zeros(n_bootstrap, dtype=np.float64)
    n = vals.shape[0]
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(vals[sample_idx]))
    alpha = 1.0 - confidence
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def batch_metrics_summary(
    reports: Sequence[dict[str, float]],
    *,
    include_std: bool = False,
    include_ci: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 400,
    random_state: int = 0,
) -> dict[str, float]:
    """Aggregate multiple retrieval reports into macro means."""
    if len(reports) == 0:
        raise ValueError("eval_error: reports cannot be empty")
    keys = sorted(reports[0].keys())
    for r in reports:
        if sorted(r.keys()) != keys:
            raise ValueError("eval_error: all reports must contain identical metric keys")
    summary: dict[str, float] = {}
    for key in keys:
        vals = [float(r[key]) for r in reports]
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = float(np.mean(arr))
        if include_std:
            summary[f"{key}_std"] = float(np.std(arr))
        if include_ci:
            lo, hi = _bootstrap_ci(
                arr,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            summary[f"{key}_ci_lower"] = lo
            summary[f"{key}_ci_upper"] = hi
    summary["num_reports"] = float(len(reports))
    return summary


def retrieval_cohort_report(
    retrieved_ids: np.ndarray,
    ground_truth_ids: np.ndarray | Iterable[Iterable[object]],
    cohorts: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, object]:
    """Compute retrieval metrics grouped by query cohort labels."""
    retrieved = _ensure_2d("retrieved_ids", retrieved_ids)
    if len(cohorts) != retrieved.shape[0]:
        raise ValueError("eval_error: cohorts length must match number of queries")
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(cohorts):
        if not isinstance(label, str) or not label.strip():
            raise ValueError("eval_error: cohort labels must be non-empty strings")
        groups.setdefault(label, []).append(i)

    gt = _normalize_ground_truth(ground_truth_ids, n_queries=retrieved.shape[0])
    per_cohort: dict[str, dict[str, float]] = {}
    for label, idxs in groups.items():
        sub_retrieved = np.asarray([retrieved[i].tolist() for i in idxs], dtype=object)
        sub_gt = [gt[i] for i in idxs]
        per_cohort[label] = retrieval_report(sub_retrieved, sub_gt, ks=ks)
    overall = retrieval_report(retrieved, gt, ks=ks)
    return {
        "overall": overall,
        "per_cohort": per_cohort,
        "cohort_sizes": {k: float(len(v)) for k, v in groups.items()},
    }

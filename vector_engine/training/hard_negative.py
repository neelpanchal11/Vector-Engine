from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from vector_engine.array import VectorArray
from vector_engine.index import VectorIndex


@dataclass(frozen=True)
class TripletBatch:
    anchors: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray
    negative_scores: np.ndarray | None = None


def _prepare_exclusion_mask(
    retrieved_ids: np.ndarray,
    *,
    exclude_ids: set[object] | None,
    exclude_mask: np.ndarray | None,
) -> np.ndarray:
    n_queries, k = retrieved_ids.shape
    mask = np.zeros((n_queries, k), dtype=bool)
    if exclude_mask is not None:
        raw = np.asarray(exclude_mask, dtype=bool)
        if raw.shape != mask.shape:
            raise ValueError("training_error: exclude_mask shape must match retrieved IDs shape")
        mask |= raw
    if exclude_ids:
        for i in range(n_queries):
            for j, cid in enumerate(retrieved_ids[i].tolist()):
                if cid in exclude_ids:
                    mask[i, j] = True
    return mask


def _pick_from_candidates(
    ids_row: np.ndarray,
    scores_row: np.ndarray,
    *,
    strategy: str,
    mask_row: np.ndarray,
    rng: np.random.Generator,
    topk_sample_size: int,
    distance_band: tuple[int, int],
) -> tuple[object, float]:
    valid_indices = [idx for idx in range(len(ids_row)) if not mask_row[idx]]
    if not valid_indices:
        fallback_idx = int(len(ids_row) - 1)
        return ids_row[fallback_idx], float(scores_row[fallback_idx])

    if strategy in {"exclude_positive", "top1"}:
        idx = valid_indices[0]
        return ids_row[idx], float(scores_row[idx])

    if strategy == "topk_sample":
        sampled = valid_indices[: min(len(valid_indices), topk_sample_size)]
        idx = int(rng.choice(np.asarray(sampled, dtype=np.int64)))
        return ids_row[idx], float(scores_row[idx])

    if strategy == "distance_band":
        lo_rank, hi_rank = distance_band
        lo = min(max(0, lo_rank), len(valid_indices))
        hi = min(max(lo + 1, hi_rank), len(valid_indices))
        if hi <= lo:
            idx = valid_indices[0]
            return ids_row[idx], float(scores_row[idx])
        idx = valid_indices[int(rng.integers(lo, hi))]
        return ids_row[idx], float(scores_row[idx])

    raise ValueError(
        "training_error: unsupported strategy; expected one of "
        "{'top1','exclude_positive','topk_sample','distance_band'}"
    )


def mine_hard_negatives(
    index: VectorIndex,
    anchors: VectorArray,
    *,
    positives: np.ndarray,
    k: int = 20,
    strategy: str = "exclude_positive",
    topk_sample_size: int = 5,
    distance_band: tuple[int, int] = (1, 5),
    exclude_ids: Sequence[object] | None = None,
    exclude_mask: np.ndarray | None = None,
    random_state: int = 0,
) -> TripletBatch:
    """Return one hard negative ID per anchor query."""
    if k <= 0:
        raise ValueError("training_error: k must be > 0")
    res = index.search(anchors, k=k, return_metadata=False)
    pos = np.asarray(positives, dtype=object)
    if len(pos) != len(anchors.ids):
        raise ValueError("training_error: positives length must equal number of anchors")
    if not isinstance(topk_sample_size, int) or topk_sample_size <= 0:
        raise ValueError("training_error: topk_sample_size must be a positive int")
    if (
        not isinstance(distance_band, tuple)
        or len(distance_band) != 2
        or not all(isinstance(x, int) for x in distance_band)
    ):
        raise ValueError("training_error: distance_band must be a tuple[int, int]")
    if distance_band[0] < 0 or distance_band[1] <= distance_band[0]:
        raise ValueError("training_error: distance_band must satisfy 0 <= lo < hi")
    exclude_set = set(exclude_ids) if exclude_ids is not None else None
    mask = _prepare_exclusion_mask(
        res.ids,
        exclude_ids=exclude_set,
        exclude_mask=exclude_mask,
    )
    rng = np.random.default_rng(random_state)
    negatives = []
    negative_scores = []
    for i, row in enumerate(res.ids):
        row_mask = np.array(mask[i], dtype=bool)
        if strategy in {"exclude_positive", "top1", "topk_sample", "distance_band"}:
            row_mask = row_mask | (row == pos[i])
        neg_id, neg_score = _pick_from_candidates(
            row,
            res.scores[i],
            strategy=strategy,
            mask_row=row_mask,
            rng=rng,
            topk_sample_size=topk_sample_size,
            distance_band=distance_band,
        )
        negatives.append(neg_id)
        negative_scores.append(neg_score)
    return TripletBatch(
        anchors=np.asarray(anchors.ids, dtype=object),
        positives=pos,
        negatives=np.asarray(negatives, dtype=object),
        negative_scores=np.asarray(negative_scores, dtype=np.float32),
    )

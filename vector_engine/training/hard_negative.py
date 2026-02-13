from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_engine.array import VectorArray
from vector_engine.index import VectorIndex


@dataclass(frozen=True)
class TripletBatch:
    anchors: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray


def mine_hard_negatives(
    index: VectorIndex,
    anchors: VectorArray,
    *,
    positives: np.ndarray,
    k: int = 20,
    strategy: str = "exclude_positive",
) -> TripletBatch:
    """Return one hard negative ID per anchor query."""
    res = index.search(anchors, k=k, return_metadata=False)
    pos = np.asarray(positives, dtype=object)
    negatives = []
    for i, row in enumerate(res.ids):
        chosen = None
        for cand in row.tolist():
            if strategy == "exclude_positive" and cand == pos[i]:
                continue
            chosen = cand
            break
        if chosen is None:
            chosen = row[-1]
        negatives.append(chosen)
    return TripletBatch(
        anchors=np.asarray(anchors.ids, dtype=object),
        positives=pos,
        negatives=np.asarray(negatives, dtype=object),
    )

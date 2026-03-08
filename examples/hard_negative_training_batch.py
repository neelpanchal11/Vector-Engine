from __future__ import annotations

import numpy as np

from vector_engine import VectorArray, VectorIndex
from vector_engine.training import mine_hard_negatives


def main() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(100, 32)).astype(np.float32)
    ids = [f"item-{i}" for i in range(100)]
    corpus = VectorArray.from_numpy(embeddings, ids=ids, normalize=True)
    index = VectorIndex.create(corpus, metric="cosine", backend="bruteforce")

    anchor_ids = ids[:8]
    anchors = corpus.subset(anchor_ids)
    positives = np.asarray(anchor_ids, dtype=object)
    batch = mine_hard_negatives(
        index,
        anchors,
        positives=positives,
        k=20,
        strategy="topk_sample",
        topk_sample_size=5,
        random_state=7,
    )

    print("anchors", batch.anchors.tolist())
    print("positives", batch.positives.tolist())
    print("negatives", batch.negatives.tolist())
    print("negative_scores", batch.negative_scores.tolist())


if __name__ == "__main__":
    main()

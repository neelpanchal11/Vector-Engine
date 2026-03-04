from __future__ import annotations

import os
import sys

import numpy as np

try:
    from vector_engine import VectorArray, VectorIndex
except ModuleNotFoundError:
    # Fallback for monorepo usage before package installation.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vector_engine import VectorArray, VectorIndex


def main() -> None:
    item_ids = [f"item-{i}" for i in range(6)]
    rng = np.random.default_rng(123)
    xb = rng.normal(size=(6, 16)).astype(np.float32)
    # Make item-2 intentionally similar to item-0 for a meaningful demo hit.
    xb[2] = xb[0] + 0.01 * rng.normal(size=(16,)).astype(np.float32)

    base = VectorArray.from_numpy(
        xb,
        ids=item_ids,
        metadata=[{"category": "demo", "item_id": x} for x in item_ids],
        normalize=True,
    )
    index = VectorIndex.create(base, metric="cosine", backend="bruteforce")
    query = base.subset(["item-0"])
    res = index.search(query, k=4, return_metadata=True)

    # Exclude self to show nearest related items.
    neighbors = [x for x in res.ids[0].tolist() if x != "item-0"][:3]
    print("query item: item-0")
    print("neighbors:", neighbors)


if __name__ == "__main__":
    main()

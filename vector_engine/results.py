from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Top-k search results for batched queries."""

    ids: np.ndarray
    scores: np.ndarray
    metadata: list[list[dict[str, Any]]] | None = None

from __future__ import annotations

from typing import Protocol

import numpy as np

from vector_engine.metric import Metric


class BaseBackend(Protocol):
    name: str
    capabilities: dict[str, bool]

    def build(self, xb: np.ndarray, metric: Metric, config: dict) -> None:
        ...

    def add(self, xb: np.ndarray) -> np.ndarray:
        ...

    def search(self, xq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str) -> "BaseBackend":
        ...

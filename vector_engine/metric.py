from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Metric:
    """Distance or similarity metric definition."""

    name: str
    higher_is_better: bool
    fn: MetricFn | None = None

    @staticmethod
    def cosine() -> "Metric":
        return Metric(name="cosine", higher_is_better=True, fn=None)

    @staticmethod
    def l2() -> "Metric":
        return Metric(name="l2", higher_is_better=False, fn=None)

    @staticmethod
    def inner_product() -> "Metric":
        return Metric(name="ip", higher_is_better=True, fn=None)

    @staticmethod
    def custom(
        name: str,
        fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        higher_is_better: bool,
    ) -> "Metric":
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric_error: custom metric name must be a non-empty string")
        if not callable(fn):
            raise TypeError("metric_error: custom metric function must be callable")
        return Metric(name=name, higher_is_better=higher_is_better, fn=fn)

    @staticmethod
    def from_value(value: str | "Metric") -> "Metric":
        if isinstance(value, Metric):
            return value
        if not isinstance(value, str):
            raise TypeError("metric_error: metric must be a Metric instance or metric name string")
        mapping = {
            "cosine": Metric.cosine,
            "l2": Metric.l2,
            "ip": Metric.inner_product,
            "inner_product": Metric.inner_product,
        }
        key = value.lower()
        if key not in mapping:
            raise ValueError(f"metric_error: unsupported metric '{value}'")
        return mapping[key]()

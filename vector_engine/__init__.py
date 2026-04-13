"""Vector Engine public API."""

__version__ = "1.1.0"

from .array import VectorArray
from .index import VectorIndex
from .metric import Metric
from .results import SearchResult

__all__ = [
    "__version__",
    "Metric",
    "SearchResult",
    "VectorArray",
    "VectorIndex",
]

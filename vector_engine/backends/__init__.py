from .bruteforce import BruteForceBackend
from .faiss_backend import FaissBackend
from .registry import get_backend, register_backend

register_backend("bruteforce", BruteForceBackend)
register_backend("faiss", FaissBackend)

__all__ = [
    "BruteForceBackend",
    "FaissBackend",
    "get_backend",
    "register_backend",
]

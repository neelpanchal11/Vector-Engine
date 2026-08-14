from .bruteforce import BruteForceBackend
from .faiss_backend import FaissBackend
from .ivf import IVFBackend
from .registry import get_backend, register_backend

register_backend("bruteforce", BruteForceBackend)
register_backend("faiss", FaissBackend)
register_backend("ivf", IVFBackend)

__all__ = [
    "BruteForceBackend",
    "FaissBackend",
    "IVFBackend",
    "get_backend",
    "register_backend",
]

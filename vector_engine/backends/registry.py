from __future__ import annotations

from typing import Any

BACKENDS: dict[str, type[Any]] = {}


def register_backend(name: str, backend_cls: type[Any]) -> None:
    BACKENDS[name] = backend_cls


def get_backend(name: str) -> type[Any]:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend: {name}")
    return BACKENDS[name]

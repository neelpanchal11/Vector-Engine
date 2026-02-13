from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_MANIFEST_VERSIONS = {"0.1"}


@dataclass
class IndexManifest:
    version: str
    backend: str
    metric_name: str
    higher_is_better: bool
    dim: int
    count: int
    backend_config: dict[str, Any]
    ids_sha256: str | None = None
    metadata_sha256: str | None = None


def save_manifest(path: str, manifest: IndexManifest) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(path: str) -> IndexManifest:
    with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    required = {
        "version",
        "backend",
        "metric_name",
        "higher_is_better",
        "dim",
        "count",
        "backend_config",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ValueError(f"manifest_error: missing required fields: {missing}")
    if raw["version"] not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(
            "manifest_error: unsupported manifest version "
            f"'{raw['version']}', supported={sorted(SUPPORTED_MANIFEST_VERSIONS)}"
        )
    return IndexManifest(**raw)

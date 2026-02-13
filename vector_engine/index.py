from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vector_engine.array import VectorArray
from vector_engine.backends import get_backend
from vector_engine.io import IndexManifest, load_manifest, save_manifest
from vector_engine.metric import Metric
from vector_engine.results import SearchResult


def _to_external_ids(
    internal_ids: np.ndarray,
    row_to_external: list[int | str],
) -> np.ndarray:
    out = np.empty(internal_ids.shape, dtype=object)
    for i in range(internal_ids.shape[0]):
        for j in range(internal_ids.shape[1]):
            idx = int(internal_ids[i, j])
            if idx < 0 or idx >= len(row_to_external):
                out[i, j] = None
            else:
                out[i, j] = row_to_external[idx]
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _sha256_json(value: Any) -> str:
    normalized = _json_safe(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class VectorIndex:
    backend_name: str
    metric: Metric
    backend_config: dict[str, Any] = field(default_factory=dict)
    _backend: Any | None = None
    _row_to_external: list[int | str] = field(default_factory=list)
    _metadata: list[dict[str, Any]] | None = None
    _dim: int = 0

    @classmethod
    def create(
        cls,
        vectors: VectorArray,
        *,
        metric: Metric | str = "cosine",
        backend: str = "bruteforce",
        backend_config: dict[str, Any] | None = None,
        id_field: str = "id",
    ) -> "VectorIndex":
        if not isinstance(id_field, str) or not id_field.strip():
            raise ValueError("index_error: id_field must be a non-empty string")
        metric_obj = Metric.from_value(metric)
        backend_cls = get_backend(backend)
        backend_instance = backend_cls()
        if backend_config is not None and not isinstance(backend_config, dict):
            raise TypeError("index_error: backend_config must be a dict when provided")
        config = dict(backend_config or {})
        backend_instance.build(vectors.values, metric_obj, config)
        md = [dict(x) for x in vectors.metadata] if vectors.metadata is not None else None
        return cls(
            backend_name=backend,
            metric=metric_obj,
            backend_config=config,
            _backend=backend_instance,
            _row_to_external=vectors.ids.tolist(),
            _metadata=md,
            _dim=vectors.shape[1],
        )

    def add(self, vectors: VectorArray) -> None:
        if self._backend is None:
            raise RuntimeError("index_error: index is not initialized")
        if vectors.shape[1] != self._dim:
            raise ValueError(f"index_error: expected vectors with dim={self._dim}, got dim={vectors.shape[1]}")
        existing = set(self._row_to_external)
        incoming = vectors.ids.tolist()
        dup = [id_ for id_ in incoming if id_ in existing]
        if dup:
            raise ValueError(f"index_error: duplicate IDs in add(): {dup[:3]}")
        self._backend.add(vectors.values)
        self._row_to_external.extend(incoming)
        if vectors.metadata is not None:
            if self._metadata is None:
                self._metadata = [{} for _ in range(len(self._row_to_external) - len(incoming))]
            self._metadata.extend(dict(x) for x in vectors.metadata)
        elif self._metadata is not None:
            self._metadata.extend({} for _ in incoming)

    def search(
        self,
        queries: VectorArray,
        *,
        k: int = 10,
        return_metadata: bool = True,
    ) -> SearchResult:
        if self._backend is None:
            raise RuntimeError("index_error: index is not initialized")
        if queries.shape[1] != self._dim:
            raise ValueError(f"index_error: expected queries with dim={self._dim}, got dim={queries.shape[1]}")
        if not isinstance(k, int):
            raise TypeError("index_error: k must be an int")
        if k <= 0:
            raise ValueError("index_error: k must be > 0")
        scores, internal_ids = self._backend.search(queries.values, k)
        external_ids = _to_external_ids(internal_ids, self._row_to_external)
        md_out = None
        if return_metadata and self._metadata is not None:
            md_out = []
            for row in internal_ids:
                items = []
                for idx in row:
                    rid = int(idx)
                    if rid < 0 or rid >= len(self._metadata):
                        items.append({})
                    else:
                        items.append(self._metadata[rid])
                md_out.append(items)
        return SearchResult(ids=external_ids, scores=scores, metadata=md_out)

    def save(self, path: str) -> None:
        if self._backend is None:
            raise RuntimeError("index_error: index is not initialized")
        os.makedirs(path, exist_ok=True)
        backend_path = os.path.join(path, "backend")
        self._backend.save(backend_path)
        save_manifest(
            path,
            IndexManifest(
                version="0.1",
                backend=self.backend_name,
                metric_name=self.metric.name,
                higher_is_better=self.metric.higher_is_better,
                dim=self._dim,
                count=len(self._row_to_external),
                backend_config=self.backend_config,
                ids_sha256=_sha256_json(self._row_to_external),
                metadata_sha256=_sha256_json(self._metadata),
            ),
        )
        with open(os.path.join(path, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(self._row_to_external), f)
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(self._metadata), f)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        manifest = load_manifest(path)
        backend_cls = get_backend(manifest.backend)
        backend = backend_cls.load(os.path.join(path, "backend"))
        with open(os.path.join(path, "ids.json"), "r", encoding="utf-8") as f:
            row_to_external = json.load(f)
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if len(row_to_external) != int(manifest.count):
            raise ValueError(
                "index_error: ids artifact count does not match manifest count "
                f"({len(row_to_external)} != {manifest.count})"
            )
        if manifest.ids_sha256 is not None and _sha256_json(row_to_external) != manifest.ids_sha256:
            raise ValueError("index_error: ids artifact checksum does not match manifest")
        if manifest.metadata_sha256 is not None and _sha256_json(metadata) != manifest.metadata_sha256:
            raise ValueError("index_error: metadata artifact checksum does not match manifest")
        return cls(
            backend_name=manifest.backend,
            metric=Metric(name=manifest.metric_name, higher_is_better=manifest.higher_is_better),
            backend_config=manifest.backend_config,
            _backend=backend,
            _row_to_external=row_to_external,
            _metadata=metadata,
            _dim=manifest.dim,
        )

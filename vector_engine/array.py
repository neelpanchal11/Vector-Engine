from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


@dataclass(frozen=True)
class VectorArray:
    """Canonical vector container with IDs and metadata."""

    values: np.ndarray
    ids: np.ndarray
    metadata: list[dict[str, Any]] | None = None
    source_framework: str = "numpy"
    source_device: str = "cpu"
    _id_to_row: dict[int | str, int] | None = None

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("vector_array_error: values must be a 2D array with shape (n, d)")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("vector_array_error: values must have at least one row and one column")
        if len(self.ids) != values.shape[0]:
            raise ValueError("vector_array_error: ids length must match number of rows")
        if self.metadata is not None and len(self.metadata) != values.shape[0]:
            raise ValueError("vector_array_error: metadata length must match number of rows")
        id_to_row: dict[int | str, int] = {}
        for i, raw in enumerate(self.ids.tolist()):
            if not isinstance(raw, (int, str, np.integer)):
                raise TypeError("vector_array_error: ids must contain only int or str values")
            if isinstance(raw, np.integer):
                raw = int(raw)
            if raw in id_to_row:
                raise ValueError(f"vector_array_error: duplicate id found: {raw}")
            id_to_row[raw] = i

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_id_to_row", id_to_row)

    @classmethod
    def from_numpy(
        cls,
        x: np.ndarray,
        *,
        ids: Sequence[int | str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        normalize: bool = False,
    ) -> "VectorArray":
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vector_array_error: input numpy array must be shape (n, d)")
        arr = np.ascontiguousarray(arr)
        if normalize:
            arr = _normalize_l2(arr)
        if ids is None:
            ids_array = np.arange(arr.shape[0], dtype=np.int64)
        else:
            ids_list = list(ids)
            if len(ids_list) != arr.shape[0]:
                raise ValueError("vector_array_error: ids length must match number of rows")
            ids_array = np.asarray(ids_list, dtype=object)
        md = [dict(item) for item in metadata] if metadata is not None else None
        return cls(values=arr, ids=ids_array, metadata=md, source_framework="numpy")

    @classmethod
    def from_torch(
        cls,
        x: "torch.Tensor",  # type: ignore[name-defined]
        *,
        ids: Sequence[int | str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        to_numpy: bool = True,
        normalize: bool = False,
    ) -> "VectorArray":
        if not to_numpy:
            raise ValueError("vector_array_error: v0.1 requires to_numpy=True for torch inputs")
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError("vector_array_error: x must be a torch.Tensor")
        device = str(x.device)
        arr = x.detach().to("cpu").numpy()
        out = cls.from_numpy(arr, ids=ids, metadata=metadata, normalize=normalize)
        return cls(
            values=out.values,
            ids=out.ids,
            metadata=out.metadata,
            source_framework="torch",
            source_device=device,
        )

    @classmethod
    def from_jax(
        cls,
        x: "jax.Array",  # type: ignore[name-defined]
        *,
        ids: Sequence[int | str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        to_numpy: bool = True,
        normalize: bool = False,
    ) -> "VectorArray":
        if not to_numpy:
            raise ValueError("vector_array_error: v0.1 requires to_numpy=True for jax inputs")
        import jax.numpy as jnp

        arr = np.asarray(jnp.asarray(x), dtype=np.float32)
        out = cls.from_numpy(arr, ids=ids, metadata=metadata, normalize=normalize)
        return cls(
            values=out.values,
            ids=out.ids,
            metadata=out.metadata,
            source_framework="jax",
            source_device="device",
        )

    def to_numpy(self, *, dtype: np.dtype = np.float32, copy: bool = False) -> np.ndarray:
        arr = self.values.astype(dtype, copy=False)
        if copy:
            return np.array(arr, copy=True)
        return arr

    def subset(self, ids: Sequence[int | str]) -> "VectorArray":
        assert self._id_to_row is not None
        rows = []
        for id_ in ids:
            if id_ not in self._id_to_row:
                raise KeyError(f"vector_array_error: unknown id in subset(): {id_}")
            rows.append(self._id_to_row[id_])
        values = self.values[rows]
        out_ids = np.asarray(list(ids), dtype=object)
        out_meta = None
        if self.metadata is not None:
            out_meta = [self.metadata[i] for i in rows]
        return VectorArray(
            values=values,
            ids=out_ids,
            metadata=out_meta,
            source_framework=self.source_framework,
            source_device=self.source_device,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

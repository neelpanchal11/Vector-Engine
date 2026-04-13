from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    embeddings: np.ndarray
    ids: list[object]
    metadata: list[dict[str, Any]] | None = None
    labels: list[object] | None = None
    splits: list[str] | None = None
    ground_truth: list[list[object]] | None = None
    query_groups: list[str] | None = None


def _ensure_2d_float32(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"dataset_error: {name} must be a 2D matrix")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"dataset_error: {name} must have non-zero rows and columns")
    if not np.isfinite(arr).all():
        raise ValueError(f"dataset_error: {name} must contain only finite values")
    return arr


def _ensure_unique_ids(ids: list[object]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError("dataset_error: ids must be unique")


def _load_optional_list(path: str | None, *, field: str, expected_len: int) -> list[Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"dataset_error: {field} file must be a JSON list")
    if len(raw) != expected_len:
        raise ValueError(f"dataset_error: {field} length must match embedding rows")
    return list(raw)


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset_error: row {i} must be an object")
            yield row


def estimate_numpy_bundle_memory_mb(n: int, d: int) -> float:
    if n <= 0 or d <= 0:
        raise ValueError("dataset_error: n and d must be > 0 for memory estimate")
    return float((n * d * 4) / (1024**2))


def load_numpy_bundle(
    embeddings_path: str,
    ids_path: str,
    metadata_path: str | None = None,
    *,
    labels_path: str | None = None,
    splits_path: str | None = None,
    ground_truth_path: str | None = None,
    query_groups_path: str | None = None,
    mmap_mode: str | None = None,
) -> DatasetBundle:
    xb = _ensure_2d_float32("embeddings", np.load(embeddings_path, mmap_mode=mmap_mode))
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    if not isinstance(ids, list):
        raise ValueError("dataset_error: ids file must be a JSON list")
    if len(ids) != xb.shape[0]:
        raise ValueError("dataset_error: ids length must match embedding rows")
    ids_list = list(ids)
    _ensure_unique_ids(ids_list)
    metadata = None
    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("dataset_error: metadata file must be a JSON list")
        if len(raw) != xb.shape[0]:
            raise ValueError("dataset_error: metadata length must match embedding rows")
        metadata = [dict(x) for x in raw]
    labels = _load_optional_list(labels_path, field="labels", expected_len=xb.shape[0])
    splits = _load_optional_list(splits_path, field="splits", expected_len=xb.shape[0])
    if splits is not None:
        splits = [str(x) for x in splits]
    ground_truth_raw = _load_optional_list(ground_truth_path, field="ground_truth", expected_len=xb.shape[0])
    ground_truth = None
    if ground_truth_raw is not None:
        ground_truth = []
        for i, item in enumerate(ground_truth_raw):
            if not isinstance(item, list):
                raise ValueError(f"dataset_error: ground_truth[{i}] must be a list")
            ground_truth.append(list(item))
    query_groups_raw = _load_optional_list(query_groups_path, field="query_groups", expected_len=xb.shape[0])
    query_groups = [str(x) for x in query_groups_raw] if query_groups_raw is not None else None
    return DatasetBundle(
        embeddings=xb,
        ids=ids_list,
        metadata=metadata,
        labels=labels,
        splits=splits,
        ground_truth=ground_truth,
        query_groups=query_groups,
    )


def load_jsonl_bundle(
    path: str,
    *,
    vector_field: str = "embedding",
    id_field: str = "id",
    label_field: str | None = None,
    split_field: str | None = None,
    ground_truth_field: str | None = None,
    query_group_field: str | None = None,
) -> DatasetBundle:
    ids: list[object] = []
    vectors: list[list[float]] = []
    metadata: list[dict[str, Any]] = []
    labels: list[object] | None = [] if label_field is not None else None
    splits: list[str] | None = [] if split_field is not None else None
    ground_truth: list[list[object]] | None = [] if ground_truth_field is not None else None
    query_groups: list[str] | None = [] if query_group_field is not None else None
    row_count = 0
    for i, row in enumerate(_iter_jsonl(path)):
        row_count += 1
        if id_field not in row:
            raise ValueError(f"dataset_error: row {i} missing id field '{id_field}'")
        if vector_field not in row:
            raise ValueError(f"dataset_error: row {i} missing vector field '{vector_field}'")
        ids.append(row[id_field])
        vec = row[vector_field]
        if not isinstance(vec, list):
            raise ValueError(f"dataset_error: row {i} vector field '{vector_field}' must be a list")
        vectors.append(vec)
        exclude = {id_field, vector_field}
        if label_field is not None:
            if label_field not in row:
                raise ValueError(f"dataset_error: row {i} missing label field '{label_field}'")
            labels.append(row[label_field])  # type: ignore[union-attr]
            exclude.add(label_field)
        if split_field is not None:
            if split_field not in row:
                raise ValueError(f"dataset_error: row {i} missing split field '{split_field}'")
            splits.append(str(row[split_field]))  # type: ignore[union-attr]
            exclude.add(split_field)
        if ground_truth_field is not None:
            if ground_truth_field not in row:
                raise ValueError(f"dataset_error: row {i} missing ground_truth field '{ground_truth_field}'")
            raw_gt = row[ground_truth_field]
            if not isinstance(raw_gt, list):
                raise ValueError(f"dataset_error: row {i} ground_truth field '{ground_truth_field}' must be a list")
            ground_truth.append(list(raw_gt))  # type: ignore[union-attr]
            exclude.add(ground_truth_field)
        if query_group_field is not None:
            if query_group_field not in row:
                raise ValueError(f"dataset_error: row {i} missing query_group field '{query_group_field}'")
            query_groups.append(str(row[query_group_field]))  # type: ignore[union-attr]
            exclude.add(query_group_field)
        metadata.append({k: v for k, v in row.items() if k not in exclude})
    if row_count == 0:
        raise ValueError("dataset_error: JSONL file has no records")
    xb = _ensure_2d_float32("embeddings", np.asarray(vectors, dtype=np.float32))
    _ensure_unique_ids(ids)
    return DatasetBundle(
        embeddings=xb,
        ids=ids,
        metadata=metadata,
        labels=labels,
        splits=splits,
        ground_truth=ground_truth,
        query_groups=query_groups,
    )


def load_parquet_bundle(
    path: str,
    *,
    vector_field: str = "embedding",
    id_field: str = "id",
    label_field: str | None = None,
    split_field: str | None = None,
    ground_truth_field: str | None = None,
    query_group_field: str | None = None,
) -> DatasetBundle:
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("dataset_error: pandas is required to load parquet bundles") from exc
    frame = pd.read_parquet(Path(path))
    if id_field not in frame.columns or vector_field not in frame.columns:
        raise ValueError(
            f"dataset_error: parquet must include columns '{id_field}' and '{vector_field}'"
        )
    ids = frame[id_field].tolist()
    _ensure_unique_ids(ids)
    vectors = frame[vector_field].tolist()
    xb = _ensure_2d_float32("embeddings", np.asarray(vectors, dtype=np.float32))
    labels = frame[label_field].tolist() if label_field and label_field in frame.columns else None
    splits = [str(x) for x in frame[split_field].tolist()] if split_field and split_field in frame.columns else None
    if split_field and split_field not in frame.columns:
        raise ValueError(f"dataset_error: parquet missing split field '{split_field}'")
    if label_field and label_field not in frame.columns:
        raise ValueError(f"dataset_error: parquet missing label field '{label_field}'")
    ground_truth = None
    if ground_truth_field is not None:
        if ground_truth_field not in frame.columns:
            raise ValueError(f"dataset_error: parquet missing ground_truth field '{ground_truth_field}'")
        raw_gt = frame[ground_truth_field].tolist()
        ground_truth = []
        for i, item in enumerate(raw_gt):
            if not isinstance(item, list):
                raise ValueError(f"dataset_error: parquet ground_truth row {i} must be a list")
            ground_truth.append(list(item))
    query_groups = None
    if query_group_field is not None:
        if query_group_field not in frame.columns:
            raise ValueError(f"dataset_error: parquet missing query_group field '{query_group_field}'")
        query_groups = [str(x) for x in frame[query_group_field].tolist()]
    meta_exclude = {id_field, vector_field, label_field, split_field, ground_truth_field, query_group_field}
    metadata_cols = [c for c in frame.columns if c not in meta_exclude]
    metadata = frame[metadata_cols].to_dict(orient="records") if metadata_cols else None
    return DatasetBundle(
        embeddings=xb,
        ids=ids,
        metadata=metadata,
        labels=labels,
        splits=splits,
        ground_truth=ground_truth,
        query_groups=query_groups,
    )


def with_deterministic_splits(
    bundle: DatasetBundle,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 7,
) -> DatasetBundle:
    if train_ratio <= 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("dataset_error: split ratios must satisfy train>0, val>=0, train+val<1")
    n = len(bundle.ids)
    if n == 0:
        raise ValueError("dataset_error: cannot split empty dataset")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train_end = int(round(n * train_ratio))
    val_end = int(round(n * (train_ratio + val_ratio)))
    splits = ["test"] * n
    for idx in order[:train_end]:
        splits[int(idx)] = "train"
    for idx in order[train_end:val_end]:
        splits[int(idx)] = "val"
    return DatasetBundle(
        embeddings=bundle.embeddings,
        ids=bundle.ids,
        metadata=bundle.metadata,
        labels=bundle.labels,
        splits=splits,
        ground_truth=bundle.ground_truth,
        query_groups=bundle.query_groups,
    )

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.artifact_contracts import validate_ingest_manifest

ARTIFACT_CONTRACT_VERSION = "1.0"


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], *, dim: int) -> np.ndarray: ...


@dataclass(frozen=True)
class HashEmbeddingProvider:
    seed: int = 7

    def embed(self, texts: list[str], *, dim: int) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            values = []
            counter = 0
            while len(values) < dim:
                payload = f"{self.seed}:{text}:{counter}".encode("utf-8")
                digest = hashlib.sha256(payload).digest()
                for b in digest:
                    values.append((b / 255.0) * 2.0 - 1.0)
                    if len(values) == dim:
                        break
                counter += 1
            out[i] = np.asarray(values, dtype=np.float32)
        return out


def _iter_jsonl(path: str) -> tuple[int, dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"ingest_error: row {i} must be an object")
            yield i, raw


def _estimate_ingest_memory_mb(*, record_count: int, embedding_dim: int) -> float:
    # Embeddings dominate memory footprint; add small overhead budget for ids/metadata lists.
    embedding_mb = (record_count * embedding_dim * 4) / (1024**2)
    overhead_mb = (record_count * 96) / (1024**2)
    return float(embedding_mb + overhead_mb)


def _count_jsonl_records(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_ingest(
    *,
    input_jsonl: str,
    output_dir: str,
    id_field: str,
    text_field: str,
    embedding_dim: int,
    seed: int,
    label_field: str | None = None,
    split_field: str | None = None,
    query_group_field: str | None = None,
    ground_truth_field: str | None = None,
    batch_size: int = 1024,
    max_memory_mb: float | None = None,
) -> dict[str, object]:
    if embedding_dim <= 0:
        raise ValueError("ingest_error: embedding_dim must be > 0")
    if batch_size <= 0:
        raise ValueError("ingest_error: batch_size must be > 0")
    provider: EmbeddingProvider = HashEmbeddingProvider(seed=seed)
    record_count = _count_jsonl_records(input_jsonl)
    if record_count == 0:
        raise ValueError("ingest_error: input JSONL has no records")
    estimated_memory_mb = _estimate_ingest_memory_mb(record_count=record_count, embedding_dim=embedding_dim)
    if max_memory_mb is not None and estimated_memory_mb > max_memory_mb:
        raise ValueError(
            "ingest_error: estimated ingest memory exceeds max_memory_mb "
            f"({estimated_memory_mb:.2f} > {max_memory_mb:.2f})"
        )

    ids: list[object] = []
    metadata: list[dict[str, object]] = []
    labels: list[object] | None = [] if label_field else None
    splits: list[str] | None = [] if split_field else None
    query_groups: list[str] | None = [] if query_group_field else None
    ground_truth: list[list[object]] | None = [] if ground_truth_field else None
    embeddings_chunks: list[np.ndarray] = []
    batch_texts: list[str] = []

    for i, row in _iter_jsonl(input_jsonl):
        if id_field not in row:
            raise ValueError(f"ingest_error: row {i} missing id field '{id_field}'")
        if text_field not in row:
            raise ValueError(f"ingest_error: row {i} missing text field '{text_field}'")
        ids.append(row[id_field])
        text_value = row[text_field]
        if not isinstance(text_value, str):
            raise ValueError(f"ingest_error: row {i} text field '{text_field}' must be a string")
        batch_texts.append(text_value)
        exclude = {id_field, text_field}
        if label_field:
            if label_field not in row:
                raise ValueError(f"ingest_error: row {i} missing label field '{label_field}'")
            labels.append(row[label_field])  # type: ignore[union-attr]
            exclude.add(label_field)
        if split_field:
            if split_field not in row:
                raise ValueError(f"ingest_error: row {i} missing split field '{split_field}'")
            splits.append(str(row[split_field]))  # type: ignore[union-attr]
            exclude.add(split_field)
        if query_group_field:
            if query_group_field not in row:
                raise ValueError(f"ingest_error: row {i} missing query_group field '{query_group_field}'")
            query_groups.append(str(row[query_group_field]))  # type: ignore[union-attr]
            exclude.add(query_group_field)
        if ground_truth_field:
            if ground_truth_field not in row:
                raise ValueError(f"ingest_error: row {i} missing ground_truth field '{ground_truth_field}'")
            gt = row[ground_truth_field]
            if not isinstance(gt, list):
                raise ValueError(f"ingest_error: row {i} ground_truth must be a list")
            ground_truth.append(list(gt))  # type: ignore[union-attr]
            exclude.add(ground_truth_field)
        metadata.append({k: v for k, v in row.items() if k not in exclude})
        if len(batch_texts) >= batch_size:
            embeddings_chunks.append(provider.embed(batch_texts, dim=embedding_dim))
            batch_texts = []

    if len(set(ids)) != len(ids):
        raise ValueError("ingest_error: ids must be unique")
    if batch_texts:
        embeddings_chunks.append(provider.embed(batch_texts, dim=embedding_dim))
    embeddings = np.vstack(embeddings_chunks)
    if not np.isfinite(embeddings).all():
        raise ValueError("ingest_error: generated embeddings contain non-finite values")

    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    ids_path = os.path.join(output_dir, "ids.json")
    metadata_path = os.path.join(output_dir, "metadata.json")
    labels_path = os.path.join(output_dir, "labels.json")
    splits_path = os.path.join(output_dir, "splits.json")
    query_groups_path = os.path.join(output_dir, "query_groups.json")
    ground_truth_path = os.path.join(output_dir, "ground_truth.json")
    manifest_path = os.path.join(output_dir, "ingest_manifest.v1.json")

    np.save(embeddings_path, embeddings.astype(np.float32))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    if labels is not None:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    if splits is not None:
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
    if query_groups is not None:
        with open(query_groups_path, "w", encoding="utf-8") as f:
            json.dump(query_groups, f, indent=2)
    if ground_truth is not None:
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

    manifest: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": input_jsonl,
        "output_dir": output_dir,
        "record_count": len(ids),
        "embedding_dim": embedding_dim,
        "provider": "hash",
        "seed": seed,
        "batch_size": batch_size,
        "max_memory_mb": max_memory_mb,
        "estimated_memory_mb": estimated_memory_mb,
        "fields": {
            "id_field": id_field,
            "text_field": text_field,
            "label_field": label_field,
            "split_field": split_field,
            "query_group_field": query_group_field,
            "ground_truth_field": ground_truth_field,
        },
        "artifacts": {
            "embeddings_path": embeddings_path,
            "ids_path": ids_path,
            "metadata_path": metadata_path,
            "labels_path": labels_path if labels is not None else None,
            "splits_path": splits_path if splits is not None else None,
            "query_groups_path": query_groups_path if query_groups is not None else None,
            "ground_truth_path": ground_truth_path if ground_truth is not None else None,
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
    }
    validate_ingest_manifest(manifest)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL text rows into a reproducible embedding bundle.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--split-field", default=None)
    parser.add_argument("--query-group-field", default=None)
    parser.add_argument("--ground-truth-field", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=None,
        help="Optional preflight memory cap; aborts ingest when estimated in-memory usage exceeds this.",
    )
    args = parser.parse_args()
    manifest = run_ingest(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        id_field=args.id_field,
        text_field=args.text_field,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        label_field=args.label_field,
        split_field=args.split_field,
        query_group_field=args.query_group_field,
        ground_truth_field=args.ground_truth_field,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory_mb,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

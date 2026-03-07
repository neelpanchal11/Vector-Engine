from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from vector_engine import VectorArray, VectorIndex
from vector_engine.eval import retrieval_report
from scripts.artifact_contracts import validate_real_corpus_payload

ARTIFACT_CONTRACT_VERSION = "1.0"


def _parse_ks(raw: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("input_error: ks cannot be empty")
    return values


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_latency(index: VectorIndex, queries: VectorArray, k: int, loops: int) -> dict[str, float]:
    latencies_ms: list[float] = []
    t0 = time.perf_counter()
    for _ in range(loops):
        t1 = time.perf_counter()
        index.search(queries, k=k, return_metadata=False)
        latencies_ms.append((time.perf_counter() - t1) * 1000.0)
    elapsed = time.perf_counter() - t0
    qps = (queries.shape[0] * loops) / elapsed
    return {
        "latency_p50_ms": float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 50)),
        "latency_p95_ms": float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 95)),
        "qps": float(qps),
    }


def evaluate_real_corpus(
    *,
    embeddings_path: str,
    query_embeddings_path: str,
    ids_path: str,
    ground_truth_path: str,
    metadata_path: str | None,
    output_path: str,
    backend: str,
    k: int,
    ks: tuple[int, ...],
    loops: int,
    threshold_recall: float | None,
    threshold_ndcg: float | None,
    threshold_p95_ms: float | None,
) -> dict[str, object]:
    t_start = time.perf_counter()
    xb = np.load(embeddings_path).astype(np.float32)
    xq = np.load(query_embeddings_path).astype(np.float32)
    if xb.ndim != 2 or xq.ndim != 2:
        raise ValueError("input_error: embeddings and query embeddings must be 2D arrays")
    if xb.shape[1] != xq.shape[1]:
        raise ValueError("input_error: embeddings and query embeddings must have matching dimensions")
    ids = _load_json(ids_path)
    if not isinstance(ids, list):
        raise ValueError("input_error: ids file must be a JSON list")
    if len(ids) != xb.shape[0]:
        raise ValueError("input_error: ids length must match corpus embeddings rows")
    metadata = None
    if metadata_path:
        md_raw = _load_json(metadata_path)
        if not isinstance(md_raw, list):
            raise ValueError("input_error: metadata file must be a JSON list")
        if len(md_raw) != xb.shape[0]:
            raise ValueError("input_error: metadata length must match corpus embeddings rows")
        metadata = md_raw
    gt_raw = _load_json(ground_truth_path)
    if not isinstance(gt_raw, list):
        raise ValueError("input_error: ground truth file must be a JSON list")
    if len(gt_raw) != xq.shape[0]:
        raise ValueError("input_error: ground truth length must match number of queries")
    gt = np.array(gt_raw, dtype=object)

    base = VectorArray.from_numpy(xb, ids=ids, metadata=metadata, normalize=True)
    queries = VectorArray.from_numpy(xq, ids=[f"q{i}" for i in range(xq.shape[0])], normalize=True)
    index = VectorIndex.create(base, metric="cosine", backend=backend)
    res = index.search(queries, k=k, return_metadata=False)
    metrics = retrieval_report(res.ids, gt, ks=ks)
    perf = _run_latency(index, queries, k=k, loops=loops)

    checks: dict[str, bool] = {}
    if threshold_recall is not None:
        checks["recall_gate"] = metrics.get(f"recall@{max(ks)}", 0.0) >= threshold_recall
    if threshold_ndcg is not None:
        checks["ndcg_gate"] = metrics.get(f"ndcg@{max(ks)}", 0.0) >= threshold_ndcg
    if threshold_p95_ms is not None:
        checks["latency_gate"] = perf["latency_p95_ms"] <= threshold_p95_ms

    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "k": k,
        "ks": list(ks),
        "metrics": metrics,
        "performance": perf,
        "topk_ids": res.ids.tolist(),
        "runtime_seconds": float(time.perf_counter() - t_start),
        "checks": checks,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "inputs": {
            "embeddings_path": embeddings_path,
            "query_embeddings_path": query_embeddings_path,
            "ids_path": ids_path,
            "ground_truth_path": ground_truth_path,
            "metadata_path": metadata_path,
        },
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
    }
    validate_real_corpus_payload(payload)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Vector Engine on real corpus artifacts.")
    parser.add_argument("--embeddings", required=True, help="Path to corpus embeddings .npy (shape: n,d).")
    parser.add_argument("--query-embeddings", required=True, help="Path to query embeddings .npy (shape: nq,d).")
    parser.add_argument("--ids", required=True, help="Path to JSON list of corpus IDs.")
    parser.add_argument("--ground-truth", required=True, help="Path to JSON list of relevant IDs per query.")
    parser.add_argument("--metadata", default=None, help="Optional JSON list of metadata aligned with corpus IDs.")
    parser.add_argument("--output", default="artifacts/real_corpus_eval.json", help="Output report JSON path.")
    parser.add_argument("--backend", default="bruteforce", choices=("bruteforce", "faiss"), help="Backend.")
    parser.add_argument("--k", type=int, default=10, help="Search top-k.")
    parser.add_argument("--ks", default="1,5,10", help="Metric ks, comma-separated.")
    parser.add_argument("--loops", type=int, default=5, help="Number of timed loops for latency/QPS.")
    parser.add_argument("--threshold-recall", type=float, default=None, help="Optional minimum recall gate.")
    parser.add_argument("--threshold-ndcg", type=float, default=None, help="Optional minimum NDCG gate.")
    parser.add_argument("--threshold-p95-ms", type=float, default=None, help="Optional max p95 latency gate.")
    args = parser.parse_args()

    payload = evaluate_real_corpus(
        embeddings_path=args.embeddings,
        query_embeddings_path=args.query_embeddings,
        ids_path=args.ids,
        ground_truth_path=args.ground_truth,
        metadata_path=args.metadata,
        output_path=args.output,
        backend=args.backend,
        k=args.k,
        ks=_parse_ks(args.ks),
        loops=args.loops,
        threshold_recall=args.threshold_recall,
        threshold_ndcg=args.threshold_ndcg,
        threshold_p95_ms=args.threshold_p95_ms,
    )
    print(json.dumps(payload["metrics"], indent=2))
    print(json.dumps(payload["performance"], indent=2))
    print(f"wrote report: {args.output}")


if __name__ == "__main__":
    main()

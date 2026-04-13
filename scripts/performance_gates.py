from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"input_error: expected JSON object at {path}")
    return payload


def _metric_mean(metric_summary: dict[str, Any], key: str) -> float | None:
    row = metric_summary.get(key)
    if not isinstance(row, dict):
        return None
    value = row.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _perf_mean(perf_summary: dict[str, Any], key: str) -> float | None:
    row = perf_summary.get(key)
    if not isinstance(row, dict):
        return None
    value = row.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _check(name: str, passed: bool, detail: str, *, severity: str = "error") -> dict[str, str]:
    return {"check": name, "status": "pass" if passed else "fail", "severity": severity, "detail": detail}


def run_performance_gates(
    *,
    matrix_summary_path: str,
    stability_summary_path: str,
    output_path: str,
    min_recall: float = 0.70,
    min_ndcg: float = 0.70,
    max_latency_p95_ms: float = 120.0,
    min_qps: float = 0.0,
    min_flat_overlap: float = 0.99,
) -> dict[str, Any]:
    matrix = _read_json(matrix_summary_path)
    stability = _read_json(stability_summary_path)
    checks: list[dict[str, str]] = []

    metric_summary = stability.get("metric_summary", {})
    perf_summary = stability.get("performance_summary", {})
    matrix_backends = matrix.get("backend_summary", {})

    recall_candidates = [k for k in metric_summary.keys() if str(k).startswith("recall@")]
    ndcg_candidates = [k for k in metric_summary.keys() if str(k).startswith("ndcg@")]
    recall_key = sorted(recall_candidates)[-1] if recall_candidates else None
    ndcg_key = sorted(ndcg_candidates)[-1] if ndcg_candidates else None

    recall_mean = _metric_mean(metric_summary, recall_key) if recall_key else None
    ndcg_mean = _metric_mean(metric_summary, ndcg_key) if ndcg_key else None
    p95_mean = _perf_mean(perf_summary, "latency_p95_ms")
    qps_mean = _perf_mean(perf_summary, "qps")

    checks.append(
        _check(
            "stability_recall_floor",
            recall_mean is not None and recall_mean >= min_recall,
            f"{recall_key}={recall_mean}, threshold={min_recall}",
        )
    )
    checks.append(
        _check(
            "stability_ndcg_floor",
            ndcg_mean is not None and ndcg_mean >= min_ndcg,
            f"{ndcg_key}={ndcg_mean}, threshold={min_ndcg}",
        )
    )
    checks.append(
        _check(
            "stability_latency_p95_ceiling",
            p95_mean is not None and p95_mean <= max_latency_p95_ms,
            f"latency_p95_ms.mean={p95_mean}, threshold={max_latency_p95_ms}",
        )
    )
    checks.append(
        _check(
            "stability_qps_floor",
            qps_mean is not None and qps_mean >= min_qps,
            f"qps.mean={qps_mean}, threshold={min_qps}",
        )
    )

    flat_overlap = None
    if isinstance(matrix_backends, dict):
        flat = matrix_backends.get("faiss_flat", {})
        if isinstance(flat, dict):
            overlap = flat.get("overlap_vs_bruteforce", {})
            if isinstance(overlap, dict) and isinstance(overlap.get("mean"), (int, float)):
                flat_overlap = float(overlap["mean"])
    checks.append(
        _check(
            "matrix_flat_overlap_floor",
            flat_overlap is None or flat_overlap >= min_flat_overlap,
            f"faiss_flat overlap mean={flat_overlap}, threshold={min_flat_overlap}",
            severity="warn",
        )
    )

    failed_errors = [c for c in checks if c["status"] == "fail" and c["severity"] == "error"]
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failed_errors else "fail",
        "inputs": {
            "matrix_summary_path": matrix_summary_path,
            "stability_summary_path": stability_summary_path,
        },
        "thresholds": {
            "min_recall": min_recall,
            "min_ndcg": min_ndcg,
            "max_latency_p95_ms": max_latency_p95_ms,
            "min_qps": min_qps,
            "min_flat_overlap": min_flat_overlap,
        },
        "checks": checks,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate release performance gates from matrix and stability artifacts.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--stability-summary", required=True)
    parser.add_argument("--output", default="artifacts/release_gates/performance_gate_report.v1.json")
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    parser.add_argument("--max-latency-p95-ms", type=float, default=120.0)
    parser.add_argument("--min-qps", type=float, default=0.0)
    parser.add_argument("--min-flat-overlap", type=float, default=0.99)
    args = parser.parse_args()
    report = run_performance_gates(
        matrix_summary_path=args.matrix_summary,
        stability_summary_path=args.stability_summary,
        output_path=args.output,
        min_recall=args.min_recall,
        min_ndcg=args.min_ndcg,
        max_latency_p95_ms=args.max_latency_p95_ms,
        min_qps=args.min_qps,
        min_flat_overlap=args.min_flat_overlap,
    )
    print(json.dumps({"status": report["status"], "check_count": len(report["checks"])}, indent=2))
    print(f"wrote gate report: {args.output}")


if __name__ == "__main__":
    main()

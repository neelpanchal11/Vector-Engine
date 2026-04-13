from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.artifact_contracts import (
    validate_matrix_summary,
    validate_publishable_summary,
    validate_stability_summary,
)

ARTIFACT_CONTRACT_VERSION = "1.0"


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_deltas(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current_backends = current.get("matrix_backend_summary", {})
    previous_backends = previous.get("matrix_backend_summary", {})
    if isinstance(current_backends, dict) and isinstance(previous_backends, dict):
        backend_deltas: dict[str, Any] = {}
        for backend, current_metrics in current_backends.items():
            if not isinstance(current_metrics, dict):
                continue
            prev_metrics = previous_backends.get(backend, {})
            if not isinstance(prev_metrics, dict):
                continue
            metric_delta: dict[str, float] = {}
            for metric_name in ("latency_p95_ms", "qps", "overlap_vs_bruteforce"):
                current_mean = current_metrics.get(metric_name, {}).get("mean") if isinstance(current_metrics.get(metric_name), dict) else None
                prev_mean = prev_metrics.get(metric_name, {}).get("mean") if isinstance(prev_metrics.get(metric_name), dict) else None
                if isinstance(current_mean, (int, float)) and isinstance(prev_mean, (int, float)):
                    metric_delta[f"{metric_name}_mean_delta"] = float(current_mean) - float(prev_mean)
            if metric_delta:
                backend_deltas[backend] = metric_delta
        out["matrix_backend_delta"] = backend_deltas

    current_stability = current.get("stability_performance_summary", {})
    previous_stability = previous.get("stability_performance_summary", {})
    if isinstance(current_stability, dict) and isinstance(previous_stability, dict):
        stability_delta: dict[str, float] = {}
        for metric_name in ("latency_p95_ms", "qps"):
            current_mean = current_stability.get(metric_name, {}).get("mean") if isinstance(current_stability.get(metric_name), dict) else None
            prev_mean = previous_stability.get(metric_name, {}).get("mean") if isinstance(previous_stability.get(metric_name), dict) else None
            if isinstance(current_mean, (int, float)) and isinstance(prev_mean, (int, float)):
                stability_delta[f"{metric_name}_mean_delta"] = float(current_mean) - float(prev_mean)
        if stability_delta:
            out["stability_performance_delta"] = stability_delta
    return out


def build_publishable_summary(
    *,
    matrix_summary_path: str,
    stability_summary_path: str,
    out_path: str,
    previous_summary_path: str | None = None,
) -> dict[str, Any]:
    matrix = _read_json(matrix_summary_path)
    stability = _read_json(stability_summary_path)
    validate_matrix_summary(matrix)
    validate_stability_summary(stability)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "matrix_summary_path": matrix_summary_path,
            "stability_summary_path": stability_summary_path,
            "previous_summary_path": previous_summary_path,
        },
        "matrix_backend_summary": matrix.get("backend_summary", {}),
        "stability_performance_summary": stability.get("performance_summary", {}),
        "stability_metric_summary": stability.get("metric_summary", {}),
        "protocol": {
            "matrix_protocol": matrix.get("protocol", {}),
            "stability_config": stability.get("config", {}),
        },
        "environment": {
            "matrix": matrix.get("environment", {}),
            "stability": stability.get("environment", {}),
        },
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
    }
    if previous_summary_path is not None:
        previous = _read_json(previous_summary_path)
        payload["release_over_release_delta"] = _compute_deltas(payload, previous)
    validate_publishable_summary(payload)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose publishable v1 performance summary from artifact inputs.")
    parser.add_argument("--matrix-summary", default="artifacts/benchmark_matrix/matrix_summary.json")
    parser.add_argument("--stability-summary", default="artifacts/testing_runs/stability_summary_bruteforce_200.json")
    parser.add_argument("--previous-summary", default=None)
    parser.add_argument("--output", default="artifacts/benchmark_matrix/publishable_results.v1.json")
    args = parser.parse_args()
    payload = build_publishable_summary(
        matrix_summary_path=args.matrix_summary,
        stability_summary_path=args.stability_summary,
        out_path=args.output,
        previous_summary_path=args.previous_summary,
    )
    print(json.dumps(payload["sources"], indent=2))
    print(f"wrote publishable summary: {args.output}")


if __name__ == "__main__":
    main()

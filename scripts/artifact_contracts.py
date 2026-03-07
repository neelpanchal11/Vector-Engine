from __future__ import annotations

from typing import Any


def _require_keys(payload: dict[str, Any], keys: list[str], *, name: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"contract_error: {name} missing required keys: {missing}")


def _require_numeric(value: Any, *, field: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"contract_error: field '{field}' must be numeric")


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"contract_error: field '{field}' must be an object")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"contract_error: field '{field}' must be a list")
    return value


def validate_real_corpus_payload(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        [
            "timestamp_utc",
            "backend",
            "k",
            "ks",
            "metrics",
            "performance",
            "topk_ids",
            "runtime_seconds",
            "checks",
            "environment",
            "inputs",
            "artifact_contract_version",
        ],
        name="real_corpus_payload",
    )
    metrics = _require_mapping(payload["metrics"], field="metrics")
    perf = _require_mapping(payload["performance"], field="performance")
    _require_mapping(payload["environment"], field="environment")
    _require_mapping(payload["inputs"], field="inputs")
    _require_mapping(payload["checks"], field="checks")
    _require_list(payload["ks"], field="ks")
    _require_list(payload["topk_ids"], field="topk_ids")
    _require_numeric(payload["k"], field="k")
    _require_numeric(payload["runtime_seconds"], field="runtime_seconds")
    for key, value in perf.items():
        _require_numeric(value, field=f"performance.{key}")
    for key, value in metrics.items():
        _require_numeric(value, field=f"metrics.{key}")


def validate_benchmark_report(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        ["timestamp_utc", "config", "environment", "results", "artifact_contract_version"],
        name="benchmark_report",
    )
    _require_mapping(payload["config"], field="config")
    _require_mapping(payload["environment"], field="environment")
    rows = _require_list(payload["results"], field="results")
    if not rows:
        raise ValueError("contract_error: benchmark report results cannot be empty")
    for i, row in enumerate(rows):
        row_map = _require_mapping(row, field=f"results[{i}]")
        _require_keys(
            row_map,
            ["backend", "qps", "latency_p50_ms", "latency_p95_ms", "overlap_vs_bruteforce", "memory_mb_estimate"],
            name=f"benchmark_result_row[{i}]",
        )
        _require_numeric(row_map["qps"], field=f"results[{i}].qps")
        _require_numeric(row_map["latency_p50_ms"], field=f"results[{i}].latency_p50_ms")
        _require_numeric(row_map["latency_p95_ms"], field=f"results[{i}].latency_p95_ms")
        _require_numeric(row_map["overlap_vs_bruteforce"], field=f"results[{i}].overlap_vs_bruteforce")
        _require_numeric(row_map["memory_mb_estimate"], field=f"results[{i}].memory_mb_estimate")


def validate_matrix_summary(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        ["timestamp_utc", "protocol", "environment", "matrix", "backend_summary", "runs_dir", "artifact_contract_version"],
        name="matrix_summary",
    )
    _require_mapping(payload["protocol"], field="protocol")
    _require_mapping(payload["environment"], field="environment")
    _require_list(payload["matrix"], field="matrix")
    backend_summary = _require_mapping(payload["backend_summary"], field="backend_summary")
    if not backend_summary:
        raise ValueError("contract_error: backend_summary cannot be empty")


def validate_stability_summary(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        [
            "timestamp_utc",
            "run_count",
            "backend",
            "config",
            "environment",
            "performance_summary",
            "metric_summary",
            "check_pass_rate",
            "input_files",
            "runs_path",
            "artifact_contract_version",
        ],
        name="stability_summary",
    )
    _require_numeric(payload["run_count"], field="run_count")
    _require_mapping(payload["config"], field="config")
    _require_mapping(payload["environment"], field="environment")
    perf = _require_mapping(payload["performance_summary"], field="performance_summary")
    _require_mapping(payload["metric_summary"], field="metric_summary")
    _require_mapping(payload["check_pass_rate"], field="check_pass_rate")
    _require_mapping(payload["input_files"], field="input_files")
    for metric_name in ("latency_p50_ms", "latency_p95_ms", "qps"):
        metric = _require_mapping(perf.get(metric_name), field=f"performance_summary.{metric_name}")
        for stat_key in ("mean", "median", "std", "cv", "p02_5", "p97_5", "min", "max"):
            _require_numeric(metric.get(stat_key), field=f"performance_summary.{metric_name}.{stat_key}")


def validate_publishable_summary(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        [
            "generated_at_utc",
            "sources",
            "matrix_backend_summary",
            "stability_performance_summary",
            "stability_metric_summary",
            "protocol",
            "environment",
            "artifact_contract_version",
        ],
        name="publishable_summary",
    )
    _require_mapping(payload["sources"], field="sources")
    _require_mapping(payload["matrix_backend_summary"], field="matrix_backend_summary")
    _require_mapping(payload["stability_performance_summary"], field="stability_performance_summary")
    _require_mapping(payload["stability_metric_summary"], field="stability_metric_summary")
    _require_mapping(payload["protocol"], field="protocol")
    _require_mapping(payload["environment"], field="environment")

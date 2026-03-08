from __future__ import annotations

from typing import Any


def _require_keys(payload: dict[str, Any], keys: list[str], *, name: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"contract_error: {name} missing required keys: {missing}")


def _require_numeric(value: Any, *, field: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"contract_error: field '{field}' must be numeric")


def _require_between(value: Any, *, field: str, lo: float, hi: float) -> None:
    _require_numeric(value, field=field)
    value_f = float(value)
    if value_f < lo or value_f > hi:
        raise ValueError(f"contract_error: field '{field}' must be in range [{lo}, {hi}]")


def _require_non_negative(value: Any, *, field: str) -> None:
    _require_numeric(value, field=field)
    if float(value) < 0.0:
        raise ValueError(f"contract_error: field '{field}' must be >= 0")


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
    ks = _require_list(payload["ks"], field="ks")
    _require_list(payload["topk_ids"], field="topk_ids")
    _require_non_negative(payload["runtime_seconds"], field="runtime_seconds")
    k_value = payload["k"]
    if not isinstance(k_value, int) or k_value <= 0:
        raise ValueError("contract_error: field 'k' must be a positive int")
    if not ks:
        raise ValueError("contract_error: field 'ks' cannot be empty")
    for i, kval in enumerate(ks):
        if not isinstance(kval, int) or kval <= 0:
            raise ValueError(f"contract_error: field 'ks[{i}]' must be a positive int")
    if k_value < max(ks):
        raise ValueError("contract_error: field 'k' must be >= max(ks)")
    backend = payload.get("backend")
    if backend not in {"bruteforce", "faiss"}:
        raise ValueError("contract_error: field 'backend' must be one of {'bruteforce','faiss'}")
    for key, value in perf.items():
        _require_non_negative(value, field=f"performance.{key}")
    for key, value in metrics.items():
        _require_numeric(value, field=f"metrics.{key}")
        if str(key).startswith(("recall@", "ndcg@", "precision@")):
            _require_between(value, field=f"metrics.{key}", lo=0.0, hi=1.0)
    if "detailed_metrics" in payload:
        detailed = _require_mapping(payload["detailed_metrics"], field="detailed_metrics")
        _require_mapping(detailed.get("summary"), field="detailed_metrics.summary")
        if "per_query" in detailed:
            _require_list(detailed["per_query"], field="detailed_metrics.per_query")
        if "error_buckets" in detailed:
            buckets = _require_mapping(detailed["error_buckets"], field="detailed_metrics.error_buckets")
            for key, value in buckets.items():
                _require_numeric(value, field=f"detailed_metrics.error_buckets.{key}")
    if "cohort_metrics" in payload:
        cohort = _require_mapping(payload["cohort_metrics"], field="cohort_metrics")
        _require_mapping(cohort.get("overall"), field="cohort_metrics.overall")
        _require_mapping(cohort.get("per_cohort"), field="cohort_metrics.per_cohort")


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
    config = payload["config"]
    mode = config.get("mode")
    if mode not in {"exact", "ann", "all"}:
        raise ValueError("contract_error: config.mode must be one of {'exact','ann','all'}")
    k = config.get("k")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("contract_error: config.k must be a positive int")
    overlap_gate = config.get("min_flat_overlap")
    if overlap_gate is not None:
        _require_between(overlap_gate, field="config.min_flat_overlap", lo=0.0, hi=1.0)
    has_bruteforce = False
    for i, row in enumerate(rows):
        row_map = _require_mapping(row, field=f"results[{i}]")
        _require_keys(
            row_map,
            ["backend", "qps", "latency_p50_ms", "latency_p95_ms", "overlap_vs_bruteforce", "memory_mb_estimate"],
            name=f"benchmark_result_row[{i}]",
        )
        backend = row_map["backend"]
        if backend not in {"bruteforce", "faiss_flat", "faiss_ivf"}:
            raise ValueError("contract_error: benchmark result backend not in allowlist")
        has_bruteforce = has_bruteforce or backend == "bruteforce"
        _require_non_negative(row_map["qps"], field=f"results[{i}].qps")
        _require_non_negative(row_map["latency_p50_ms"], field=f"results[{i}].latency_p50_ms")
        _require_non_negative(row_map["latency_p95_ms"], field=f"results[{i}].latency_p95_ms")
        _require_between(row_map["overlap_vs_bruteforce"], field=f"results[{i}].overlap_vs_bruteforce", lo=0.0, hi=1.0)
        _require_non_negative(row_map["memory_mb_estimate"], field=f"results[{i}].memory_mb_estimate")
    if not has_bruteforce:
        raise ValueError("contract_error: benchmark report must include a bruteforce row")


def validate_matrix_summary(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        ["timestamp_utc", "protocol", "environment", "matrix", "backend_summary", "runs_dir", "artifact_contract_version"],
        name="matrix_summary",
    )
    _require_mapping(payload["protocol"], field="protocol")
    _require_mapping(payload["environment"], field="environment")
    matrix = _require_list(payload["matrix"], field="matrix")
    if not matrix:
        raise ValueError("contract_error: matrix cannot be empty")
    protocol = payload["protocol"]
    if protocol.get("mode") not in {"exact", "ann", "all"}:
        raise ValueError("contract_error: protocol.mode must be one of {'exact','ann','all'}")
    for key in ("warmup", "loops", "seed", "matrix_size"):
        value = protocol.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"contract_error: protocol.{key} must be a non-negative int")
    if int(protocol.get("matrix_size", -1)) != len(matrix):
        raise ValueError("contract_error: protocol.matrix_size must match matrix length")
    overlap_gate = protocol.get("min_flat_overlap")
    if overlap_gate is not None:
        _require_between(overlap_gate, field="protocol.min_flat_overlap", lo=0.0, hi=1.0)
    if "max_memory_mb" in protocol and protocol.get("max_memory_mb") is not None:
        _require_non_negative(protocol.get("max_memory_mb"), field="protocol.max_memory_mb")
    backend_summary = _require_mapping(payload["backend_summary"], field="backend_summary")
    if not backend_summary:
        raise ValueError("contract_error: backend_summary cannot be empty")
    for backend, metrics in backend_summary.items():
        if backend not in {"bruteforce", "faiss_flat", "faiss_ivf"}:
            raise ValueError("contract_error: backend_summary backend not in allowlist")
        metric_map = _require_mapping(metrics, field=f"backend_summary.{backend}")
        for metric_name in ("latency_p50_ms", "latency_p95_ms", "qps", "overlap_vs_bruteforce"):
            if metric_name not in metric_map:
                raise ValueError(f"contract_error: backend_summary.{backend} missing metric '{metric_name}'")
            stat_map = _require_mapping(metric_map[metric_name], field=f"backend_summary.{backend}.{metric_name}")
            for stat_name in ("mean", "median", "min", "max"):
                _require_numeric(stat_map.get(stat_name), field=f"backend_summary.{backend}.{metric_name}.{stat_name}")


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
    if int(payload["run_count"]) <= 0:
        raise ValueError("contract_error: run_count must be > 0")
    if payload.get("backend") not in {"bruteforce", "faiss"}:
        raise ValueError("contract_error: backend must be one of {'bruteforce','faiss'}")
    _require_mapping(payload["config"], field="config")
    _require_mapping(payload["environment"], field="environment")
    perf = _require_mapping(payload["performance_summary"], field="performance_summary")
    metric_summary = _require_mapping(payload["metric_summary"], field="metric_summary")
    if not metric_summary:
        raise ValueError("contract_error: metric_summary cannot be empty")
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
    sources = _require_mapping(payload["sources"], field="sources")
    if not sources:
        raise ValueError("contract_error: sources cannot be empty")
    for key, value in sources.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"contract_error: sources.{key} must be a non-empty string path")
    matrix_backend = _require_mapping(payload["matrix_backend_summary"], field="matrix_backend_summary")
    if not matrix_backend:
        raise ValueError("contract_error: matrix_backend_summary cannot be empty")
    stability_perf = _require_mapping(payload["stability_performance_summary"], field="stability_performance_summary")
    if not stability_perf:
        raise ValueError("contract_error: stability_performance_summary cannot be empty")
    stability_metric = _require_mapping(payload["stability_metric_summary"], field="stability_metric_summary")
    if not stability_metric:
        raise ValueError("contract_error: stability_metric_summary cannot be empty")
    protocol = _require_mapping(payload["protocol"], field="protocol")
    _require_mapping(protocol.get("matrix_protocol"), field="protocol.matrix_protocol")
    _require_mapping(protocol.get("stability_config"), field="protocol.stability_config")
    environment = _require_mapping(payload["environment"], field="environment")
    _require_mapping(environment.get("matrix"), field="environment.matrix")
    _require_mapping(environment.get("stability"), field="environment.stability")


def validate_ingest_manifest(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        [
            "timestamp_utc",
            "input_jsonl",
            "output_dir",
            "record_count",
            "embedding_dim",
            "provider",
            "seed",
            "fields",
            "artifacts",
            "environment",
            "artifact_contract_version",
        ],
        name="ingest_manifest",
    )
    if not isinstance(payload["input_jsonl"], str) or not payload["input_jsonl"]:
        raise ValueError("contract_error: input_jsonl must be a non-empty string")
    if not isinstance(payload["output_dir"], str) or not payload["output_dir"]:
        raise ValueError("contract_error: output_dir must be a non-empty string")
    if not isinstance(payload["provider"], str) or not payload["provider"]:
        raise ValueError("contract_error: provider must be a non-empty string")
    if not isinstance(payload["seed"], int):
        raise ValueError("contract_error: seed must be an int")
    if not isinstance(payload["record_count"], int) or payload["record_count"] <= 0:
        raise ValueError("contract_error: record_count must be a positive int")
    if not isinstance(payload["embedding_dim"], int) or payload["embedding_dim"] <= 0:
        raise ValueError("contract_error: embedding_dim must be a positive int")
    fields = _require_mapping(payload["fields"], field="fields")
    for key in (
        "id_field",
        "text_field",
        "label_field",
        "split_field",
        "query_group_field",
        "ground_truth_field",
    ):
        if key not in fields:
            raise ValueError(f"contract_error: fields missing key '{key}'")
        value = fields[key]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"contract_error: fields.{key} must be string or null")
    artifacts = _require_mapping(payload["artifacts"], field="artifacts")
    for key in ("embeddings_path", "ids_path", "metadata_path"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"contract_error: artifacts.{key} must be a non-empty string path")
    for key in ("labels_path", "splits_path", "query_groups_path", "ground_truth_path"):
        value = artifacts.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"contract_error: artifacts.{key} must be string path or null")
    _require_mapping(payload["environment"], field="environment")

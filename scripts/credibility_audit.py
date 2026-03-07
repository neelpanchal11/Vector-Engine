from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.artifact_contracts import (
    validate_matrix_summary,
    validate_publishable_summary,
    validate_real_corpus_payload,
    validate_stability_summary,
)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"input_error: expected JSON object at {path}")
    return raw


def _status(name: str, ok: bool, detail: str, severity: str = "error") -> dict[str, str]:
    return {
        "check": name,
        "status": "pass" if ok else "fail",
        "severity": severity,
        "detail": detail,
    }


def _check_no_sensitive_paths(payload: dict[str, Any], *, field: str) -> dict[str, str]:
    value = payload.get(field, {})
    text = json.dumps(value)
    bad_markers = ["/Users/", "C:\\", "/home/"]
    hit = [m for m in bad_markers if m in text]
    ok = len(hit) == 0
    detail = "no absolute local paths detected" if ok else f"detected local path markers: {hit}"
    return _status("sensitive_path_scan", ok, detail, severity="warn")


def run_audit(
    *,
    matrix_summary_path: str,
    stability_summary_path: str,
    publishable_summary_path: str,
    real_corpus_report_path: str | None,
    output_path: str,
) -> dict[str, Any]:
    matrix = _read_json(matrix_summary_path)
    stability = _read_json(stability_summary_path)
    publishable = _read_json(publishable_summary_path)

    checks: list[dict[str, str]] = []

    try:
        validate_matrix_summary(matrix)
        checks.append(_status("matrix_contract", True, "matrix summary contract valid"))
    except Exception as exc:
        checks.append(_status("matrix_contract", False, str(exc)))

    try:
        validate_stability_summary(stability)
        checks.append(_status("stability_contract", True, "stability summary contract valid"))
    except Exception as exc:
        checks.append(_status("stability_contract", False, str(exc)))

    try:
        validate_publishable_summary(publishable)
        checks.append(_status("publishable_contract", True, "publishable summary contract valid"))
    except Exception as exc:
        checks.append(_status("publishable_contract", False, str(exc)))

    matrix_mode = matrix.get("protocol", {}).get("mode")
    overlap_gate = matrix.get("protocol", {}).get("min_flat_overlap")
    checks.append(
        _status(
            "benchmark_mode_declared",
            matrix_mode in {"exact", "ann", "all"},
            f"mode={matrix_mode}",
        )
    )
    checks.append(
        _status(
            "exact_overlap_gate_declared",
            (matrix_mode != "exact") or (overlap_gate is not None),
            f"mode={matrix_mode}, min_flat_overlap={overlap_gate}",
            severity="warn",
        )
    )

    metric_summary = stability.get("metric_summary", {})
    checks.append(
        _status(
            "quality_metrics_present",
            any(str(k).startswith("recall@") for k in metric_summary.keys())
            and any(str(k).startswith("ndcg@") for k in metric_summary.keys()),
            "metric_summary contains recall@k and ndcg@k keys",
        )
    )

    checks.append(_check_no_sensitive_paths(matrix, field="runs_dir"))
    checks.append(_check_no_sensitive_paths(stability, field="input_files"))
    checks.append(_check_no_sensitive_paths(publishable, field="sources"))

    if real_corpus_report_path is not None:
        real_report = _read_json(real_corpus_report_path)
        try:
            validate_real_corpus_payload(real_report)
            checks.append(_status("real_corpus_contract", True, "real-corpus report contract valid"))
        except Exception as exc:
            checks.append(_status("real_corpus_contract", False, str(exc)))
        checks.append(_check_no_sensitive_paths(real_report, field="inputs"))

    failed_error = [c for c in checks if c["status"] == "fail" and c["severity"] == "error"]
    failed_warn = [c for c in checks if c["status"] == "fail" and c["severity"] == "warn"]
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failed_error else "fail",
        "failed_error_count": len(failed_error),
        "failed_warn_count": len(failed_warn),
        "checks": checks,
        "inputs": {
            "matrix_summary_path": matrix_summary_path,
            "stability_summary_path": stability_summary_path,
            "publishable_summary_path": publishable_summary_path,
            "real_corpus_report_path": real_corpus_report_path,
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run credibility red-team audit for publication evidence bundle.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--stability-summary", required=True)
    parser.add_argument("--publishable-summary", required=True)
    parser.add_argument("--real-corpus-report", default=None)
    parser.add_argument("--output", default="artifacts/audit/credibility_audit.v1.json")
    args = parser.parse_args()
    report = run_audit(
        matrix_summary_path=args.matrix_summary,
        stability_summary_path=args.stability_summary,
        publishable_summary_path=args.publishable_summary,
        real_corpus_report_path=args.real_corpus_report,
        output_path=args.output,
    )
    print(json.dumps({"status": report["status"], "failed_error_count": report["failed_error_count"]}, indent=2))
    print(f"wrote audit report: {args.output}")


if __name__ == "__main__":
    main()

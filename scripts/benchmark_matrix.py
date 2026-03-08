from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.artifact_contracts import validate_benchmark_report, validate_matrix_summary

ARTIFACT_CONTRACT_VERSION = "1.0"


PROFILE_MATRICES: dict[str, list[dict[str, Any]]] = {
    "dev": [
        {"name": "dev_tiny", "n": 2000, "d": 64, "nq": 64, "k": 10},
        {"name": "dev_small", "n": 4000, "d": 64, "nq": 128, "k": 10},
    ],
    "medium": [
        {"name": "s_small", "n": 5000, "d": 64, "nq": 100, "k": 10},
        {"name": "m_balanced", "n": 10000, "d": 128, "nq": 200, "k": 10},
        {"name": "l_wide", "n": 25000, "d": 256, "nq": 300, "k": 20},
    ],
    "paper": [
        {"name": "paper_a", "n": 20000, "d": 128, "nq": 256, "k": 10},
        {"name": "paper_b", "n": 50000, "d": 256, "nq": 512, "k": 20},
        {"name": "paper_c", "n": 100000, "d": 256, "nq": 512, "k": 20},
    ],
}
DEFAULT_MATRIX = PROFILE_MATRICES["medium"]


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _estimate_profile_memory_mb(cfg: dict[str, Any]) -> float:
    n = int(cfg["n"])
    d = int(cfg["d"])
    nq = int(cfg["nq"])
    # Input/query vectors as float32, matching benchmark script generation.
    return float(((n * d * 4) + (nq * d * 4)) / (1024**2))


def _apply_memory_limit(matrix: list[dict[str, Any]], *, max_memory_mb: float) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for cfg in matrix:
        if _estimate_profile_memory_mb(cfg) <= max_memory_mb:
            filtered.append(cfg)
    if not filtered:
        raise ValueError(
            f"input_error: no benchmark configs fit max_memory_mb={max_memory_mb}; "
            "increase memory cap or use a smaller profile"
        )
    return filtered


def _parse_matrix(path: str | None, *, profile: str) -> list[dict[str, Any]]:
    if path is None:
        if profile not in PROFILE_MATRICES:
            raise ValueError(f"input_error: unknown benchmark profile '{profile}'")
        return list(PROFILE_MATRICES[profile])
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("input_error: matrix config must be a non-empty JSON list")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"input_error: matrix row {i} must be an object")
        for key in ("name", "n", "d", "nq", "k"):
            if key not in row:
                raise ValueError(f"input_error: matrix row {i} missing '{key}'")
        out.append(row)
    return out


def run_matrix(
    *,
    matrix: list[dict[str, Any]],
    mode: str,
    warmup: int,
    loops: int,
    seed: int,
    min_flat_overlap: float | None,
    out_dir: str,
    profile: str,
    max_memory_mb: float,
) -> dict[str, Any]:
    matrix = _apply_memory_limit(matrix, max_memory_mb=max_memory_mb)
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cfg in matrix:
        out_path = os.path.join(out_dir, f"{cfg['name']}.json")
        cmd = [
            sys.executable,
            os.path.join("benchmarks", "compare_bruteforce_vs_faiss.py"),
            "--mode",
            mode,
            "--n",
            str(cfg["n"]),
            "--d",
            str(cfg["d"]),
            "--nq",
            str(cfg["nq"]),
            "--k",
            str(cfg["k"]),
            "--warmup",
            str(warmup),
            "--loops",
            str(loops),
            "--seed",
            str(seed),
            "--output",
            out_path,
        ]
        if min_flat_overlap is not None:
            cmd.extend(["--min-flat-overlap", str(min_flat_overlap)])
        subprocess.check_call(cmd)
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        validate_benchmark_report(payload)
        rows.append(payload)

    by_backend: dict[str, dict[str, list[float]]] = {}
    for report in rows:
        for row in report["results"]:
            backend = row["backend"]
            by_backend.setdefault(
                backend,
                {
                    "latency_p50_ms": [],
                    "latency_p95_ms": [],
                    "qps": [],
                    "overlap_vs_bruteforce": [],
                },
            )
            by_backend[backend]["latency_p50_ms"].append(float(row["latency_p50_ms"]))
            by_backend[backend]["latency_p95_ms"].append(float(row["latency_p95_ms"]))
            by_backend[backend]["qps"].append(float(row["qps"]))
            by_backend[backend]["overlap_vs_bruteforce"].append(float(row["overlap_vs_bruteforce"]))

    summary_backends: dict[str, dict[str, dict[str, float]]] = {}
    for backend, vals in by_backend.items():
        summary_backends[backend] = {metric: _stats(metric_vals) for metric, metric_vals in vals.items()}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "profile": profile,
            "mode": mode,
            "warmup": warmup,
            "loops": loops,
            "seed": seed,
            "min_flat_overlap": min_flat_overlap,
            "max_memory_mb": max_memory_mb,
            "matrix_size": len(matrix),
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "matrix": matrix,
        "backend_summary": summary_backends,
        "runs_dir": out_dir,
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
    }
    validate_matrix_summary(summary)
    summary_path = os.path.join(out_dir, "matrix_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark matrix and aggregate backend stats.")
    parser.add_argument("--matrix-config", default=None, help="Optional path to JSON list of matrix configs.")
    parser.add_argument(
        "--profile",
        default="medium",
        choices=("dev", "medium", "paper"),
        help="Profile preset used when --matrix-config is not provided.",
    )
    parser.add_argument("--mode", default="exact", choices=("exact", "ann", "all"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--min-flat-overlap",
        type=float,
        default=None,
        help=(
            "Optional overlap gate for faiss_flat in exact mode. "
            "Leave unset for no-FAISS-safe default behavior."
        ),
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=4096.0,
        help="Skip matrix configs that exceed this estimated in-memory vector/query buffer size.",
    )
    parser.add_argument("--output-dir", default="artifacts/benchmark_matrix")
    args = parser.parse_args()

    matrix = _parse_matrix(args.matrix_config, profile=args.profile)
    summary = run_matrix(
        matrix=matrix,
        mode=args.mode,
        warmup=args.warmup,
        loops=args.loops,
        seed=args.seed,
        min_flat_overlap=args.min_flat_overlap,
        out_dir=args.output_dir,
        profile=args.profile,
        max_memory_mb=args.max_memory_mb,
    )
    print(json.dumps(summary["backend_summary"], indent=2))
    print(f"wrote matrix summary: {os.path.join(args.output_dir, 'matrix_summary.json')}")


if __name__ == "__main__":
    main()

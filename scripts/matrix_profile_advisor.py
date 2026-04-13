from __future__ import annotations

import argparse
import json
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"input_error: expected JSON object at {path}")
    return payload


def recommend_configs(matrix_summary_path: str, *, top_n: int = 3) -> dict[str, Any]:
    summary = _read_json(matrix_summary_path)
    by_run: dict[str, dict[str, Any]] = {}
    for run_name in summary.get("expanded_runs", []):
        by_run[str(run_name)] = {}
    run_dir = summary.get("runs_dir")
    if not isinstance(run_dir, str):
        raise ValueError("input_error: matrix summary missing runs_dir")

    # Pull per-run backend performance from run artifacts.
    import os

    for run_name in by_run.keys():
        run_path = os.path.join(run_dir, f"{run_name}.json")
        try:
            run = _read_json(run_path)
        except Exception:
            continue
        for row in run.get("results", []):
            backend = str(row.get("backend"))
            by_run[run_name][backend] = {
                "qps": float(row.get("qps", 0.0)),
                "latency_p95_ms": float(row.get("latency_p95_ms", 0.0)),
                "overlap_vs_bruteforce": float(row.get("overlap_vs_bruteforce", 0.0)),
            }

    scored: list[dict[str, Any]] = []
    for run_name, backend_rows in by_run.items():
        if "faiss_ivf" not in backend_rows:
            continue
        ivf = backend_rows["faiss_ivf"]
        score = (
            (ivf["qps"] / max(ivf["latency_p95_ms"], 1e-6))
            * max(ivf["overlap_vs_bruteforce"], 0.0)
        )
        scored.append({"run": run_name, "score": float(score), **ivf})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {
        "matrix_summary_path": matrix_summary_path,
        "recommendations": scored[: max(int(top_n), 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend ANN matrix runs using a latency/quality-throughput heuristic.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = recommend_configs(args.matrix_summary, top_n=args.top_n)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

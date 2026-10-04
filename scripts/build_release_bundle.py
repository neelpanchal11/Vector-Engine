from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_version() -> str:
    project_text = Path("pyproject.toml").read_text(encoding="utf-8")
    project_section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", project_text)
    if project_section is None:
        raise ValueError("release_bundle_error: [project] section missing from pyproject.toml")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_section.group(1))
    if match is None:
        raise ValueError("release_bundle_error: project version missing from pyproject.toml")
    return match.group(1)


def _file_status(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": path,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
    }


def build_bundle(output_dir: str) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = _project_version()

    docs = [
        "README.md",
        "docs/kpi_charter.md",
        "docs/research_claims.md",
        "docs/reproducibility.md",
        "docs/credibility_audit.md",
        "docs/limitations.md",
        "docs/api_stability.md",
        f"docs/releases/v{version}.md",
        "docs/paper/v1_manuscript_outline.md",
        "docs/paper/reproducibility_appendix.md",
    ]
    governance = [
        "LICENSE",
        "CITATION.cff",
    ]
    artifacts = [
        "artifacts/benchmark_matrix/matrix_summary.json",
        "artifacts/benchmark_matrix/publishable_results.v1.json",
        "artifacts/testing_runs/stability_summary_bruteforce_200.json",
        "artifacts/audit/credibility_audit.v1.json",
        "artifacts/release_gates/performance_gate_report.v1.json",
        "artifacts/ivf_benchmark/ivf_batching_comparison.json",
        "artifacts/ivf_benchmark/ivf_batched_recall_sweep.json",
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_version": f"v{version}",
        "documents": [_file_status(path) for path in docs],
        "governance": [_file_status(path) for path in governance],
        "artifacts": [_file_status(path) for path in artifacts],
    }
    missing = [entry["path"] for section in ("documents", "governance", "artifacts") for entry in payload[section] if not entry["exists"]]
    payload["missing_paths"] = missing
    payload["ready_for_submission"] = len(missing) == 0

    out_path = out_dir / "release_bundle_manifest.v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication and OSS release bundle manifest.")
    parser.add_argument("--output-dir", default="artifacts/release_bundle")
    args = parser.parse_args()
    payload = build_bundle(args.output_dir)
    print(json.dumps({"ready_for_submission": payload["ready_for_submission"], "missing_count": len(payload["missing_paths"])}, indent=2))
    print(f"wrote bundle manifest: {os.path.join(args.output_dir, 'release_bundle_manifest.v1.json')}")


if __name__ == "__main__":
    main()

# Research Claim Specification (v1.0)

This document defines publishable claims, evidence requirements, and acceptance criteria.

## Claim 1: Reproducible retrieval quality evaluation

Vector Engine provides a reproducible pipeline for retrieval quality evaluation across repeated runs with explicit input validation and machine-readable outputs.

### Evidence mapping (Claim 1)

| Component | Source |
| --- | --- |
| Evaluation API | `vector_engine/eval/retrieval.py` |
| Real-corpus evaluator | `scripts/rag_real_corpus_eval.py` |
| Stability harness | `scripts/stability_runs.py` |
| Stability artifact | `artifacts/testing_runs/stability_summary_*.json` |

### Metrics and criteria (Claim 1)

- Quality metrics: `precision@k`, `recall@k`, `ndcg@k`.
- Repeated-run reliability: coefficient of variation (`cv`) on quality and performance metrics.
- Input robustness: malformed input paths must produce stable `input_error` or `eval_error` prefix behavior.

Acceptance criteria:

- All configured quality thresholds pass for canonical workload or are disclosed.
- Repeated-run summaries include `mean`, `std`, `cv`, and interval bands.
- Script outputs validate against artifact contract checks.

## Claim 2: Transparent exact-vs-ANN quality/performance tradeoff analysis (optional Faiss path)

Vector Engine enables reproducible, protocol-controlled backend comparison with quality overlap gates and tail-latency reporting when Faiss is available.

### Evidence mapping (Claim 2)

| Component | Source |
| --- | --- |
| Benchmark runner | `benchmarks/compare_bruteforce_vs_faiss.py` |
| Matrix driver | `scripts/benchmark_matrix.py` |
| Publishable summary | `scripts/publishable_results.py` |
| Benchmark artifacts | `artifacts/benchmark_matrix/*.json` |

### Metrics and criteria (Claim 2)

- Performance: `qps`, `latency_p50_ms`, `latency_p95_ms`.
- Quality retention: `overlap_vs_bruteforce`.
- Protocol traceability: seed, warmup, loops, backend mode, environment metadata.

Acceptance criteria:

- When Faiss is available and gate is enabled, exact mode enforces `overlap_vs_bruteforce >= 0.99` for `faiss_flat`.
- When Faiss is unavailable, bruteforce-only artifacts remain the required reproducibility path and optional Faiss claims are omitted.
- Matrix reports include protocol and environment metadata.
- Publishable summary links to source artifacts with contract validation.

## Claim 3: Stable developer-facing retrieval API for local ML iteration

Vector Engine exposes a stable API for core retrieval workflows, while preserving consistent error taxonomy and persistence compatibility expectations.

### Evidence mapping (Claim 3)

| Component | Source |
| --- | --- |
| Public API docs | `docs/api.md` |
| API stability tests | `tests/test_api_stability.py` |
| Core compatibility tests | `tests/test_core.py`, `tests/test_persistence_compat.py` |
| Error taxonomy docs | `docs/concepts.md` |

### Metrics and criteria (Claim 3)

- API stability: required signatures remain unchanged within v1 minor releases.
- Compatibility: persistence load checks and checksum validation pass.
- Error contract: stable error-prefix semantics (`vector_array_error`, `index_error`, `eval_error`, etc.).

Acceptance criteria:

- API stability tests pass on release branch.
- No unannounced breaking change in public API contract.
- Compatibility tests pass in CI for release candidates.

## Statistical and reporting policy

- Use repeated runs for benchmark claims, not single-run snapshots.
- Report both center and spread (`median/mean`, `std`, `cv`, intervals).
- Include hardware/runtime metadata with all externally shared results.
- Distinguish synthetic/mock workloads from private real-corpus workloads.

## Threats to validity checklist

- Dataset representativeness limitations disclosed.
- Hardware and runtime variability disclosed.
- ANN configuration sensitivity disclosed.
- Private-data constraints and publication-safe artifact boundaries disclosed.

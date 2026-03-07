# Credibility Audit Playbook (v1.0)

This playbook defines red-team checks to prevent weak claims and fragile publication evidence.

## Audit categories

## 1) Claim integrity

- Verify every public claim maps to:
  - code path,
  - artifact path,
  - metric or statistical criterion.
- Reject claims without reproducible command paths.

## 2) Benchmark fairness

- Keep exact and ANN claims separate.
- Enforce explicit overlap gate when reporting exact-equivalence (`>= 0.99` for `faiss_flat` in exact mode).
- Report p50 and p95 latency, not only mean latency.

## 3) Reproducibility robustness

- Validate artifact contracts (`artifact_contract_version`) before using results in docs/paper.
- Require stable protocol fields (seed, warmup, loops, mode).
- Repeat runs and report spread (`std`, `cv`, interval bands).

## 4) Sensitive-data hygiene

- Do not publish private raw embeddings, IDs, or metadata.
- Keep publication bundle public-safe and synthetic where needed.
- Review output artifacts for absolute local paths and sensitive tokens.

## 5) API and compatibility trust

- Run API stability tests and persistence compatibility tests for release candidates.
- Disclose any deprecations and migration notes before release.

## Limitation disclosure requirements

Release notes and manuscript must explicitly include:

- non-goals (not a new ANN algorithm),
- hardware/runtime dependency of benchmark outcomes,
- private data constraints and publication-safe artifact boundaries,
- configuration sensitivity for ANN backends.

## Audit cadence

- Pre-release candidate: run full audit and publish report.
- Release candidate: zero critical audit failures allowed.
- Post-release (first 90 days): run biweekly quick audit and track regressions.

## Automation

Use:

```bash
python scripts/credibility_audit.py \
  --matrix-summary artifacts/benchmark_matrix/matrix_summary.json \
  --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json \
  --publishable-summary artifacts/benchmark_matrix/publishable_results.v1.json \
  --real-corpus-report artifacts/real_corpus_runs/run_1.json \
  --output artifacts/audit/credibility_audit.v1.json
```

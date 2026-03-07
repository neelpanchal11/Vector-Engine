# Reproducibility Workflow (v0.3.0)

This guide defines standard commands and artifact locations for reproducible reports.

## Command workflow

1. Baseline exact-first report:

```bash
python scripts/rag_baseline.py --output-dir artifacts --k 3
```

1. Real-corpus evaluation report:

```bash
python scripts/rag_real_corpus_eval.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --output artifacts/real_corpus_runs/run_1.json --backend bruteforce --k 10 --ks 1,5,10 --loops 5
```

1. Stability run report:

```bash
python scripts/stability_runs.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --backend bruteforce --run-count 200 --output-dir artifacts/testing_runs
```

1. Exact-equivalence benchmark report:

```bash
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact --min-flat-overlap 0.99 --output artifacts/faiss_equivalence/run_1.json
```

1. Benchmark matrix report (publishable aggregate):

```bash
python scripts/benchmark_matrix.py --mode exact --warmup 2 --loops 8 --seed 7 --min-flat-overlap 0.99 --output-dir artifacts/benchmark_matrix
```

1. Publishable summary bundle:

```bash
python scripts/publishable_results.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --output artifacts/benchmark_matrix/publishable_results.v1.json
```

## Canonical protocol

- fixed random seed for all benchmark families (`seed=7` by default)
- warmup loops before timing (`warmup=2` baseline)
- timed loops (`loops=8` baseline, increase for stricter studies)
- fixed matrix definitions for cross-release comparability
- always include hardware/runtime metadata in benchmark artifacts
- enforce artifact contract validation via `artifact_contract_version`

## Artifact contract checks

All reproducibility scripts now emit `artifact_contract_version` and validate required fields/types before writing publishable artifacts.

Contract validators live in:

- `scripts/artifact_contracts.py`

Validated outputs:

- real-corpus evaluation report
- benchmark per-run reports
- benchmark matrix summary
- stability summary
- publishable summary bundle

## One-command clean-environment smoke run

Use this command to verify end-to-end pipeline behavior on synthetic public-safe inputs:

```bash
python scripts/repro_smoke.py --output-dir artifacts/repro_smoke
```

This command generates synthetic embeddings/query sets, runs baseline/eval/stability/benchmark/matrix/publishable composition, and validates all artifact contracts.

## Standard artifact locations

- Baseline: `artifacts/rag_baseline_metrics.v1.json`
- Real-corpus runs: `artifacts/real_corpus_runs/run_*.json`
- Stability traces: `artifacts/testing_runs/stability_runs_*.jsonl`
- Stability summaries: `artifacts/testing_runs/stability_summary_*.json`
- Exact-equivalence: `artifacts/faiss_equivalence/run_*.json`
- Matrix per-config reports: `artifacts/benchmark_matrix/*.json`
- Matrix aggregate summary: `artifacts/benchmark_matrix/matrix_summary.json`
- Publishable summary bundle: `artifacts/benchmark_matrix/publishable_results.v1.json`

## Publish/private policy

Publish:

- benchmark summaries
- stability aggregate summaries and charts
- non-sensitive mock inputs and examples

Keep private:

- raw embeddings from private corpora
- private query sets and relevance labels tied to sensitive data
- metadata/ID mappings containing confidential content

# Reproducibility Workflow (v1.0.0)

This guide defines standard commands and artifact locations for reproducible reports.

## Canonical local bootstrap (macOS arm64 + Python 3.12)

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c requirements/constraints-macos-arm64-py312.txt -e ".[dev,ml]"
python -m pytest -q
```

## Command workflow

1. Ingest raw JSONL into reproducible bundle:

```bash
python scripts/ingest_dataset.py --input-jsonl artifacts/raw/source.jsonl --output-dir artifacts/ingest_bundle --id-field id --text-field text --embedding-dim 256 --seed 7 --label-field label --split-field split --query-group-field query_group --ground-truth-field ground_truth
```

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

1. Optional exact-equivalence benchmark report (Faiss path):

```bash
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact --min-flat-overlap 0.99 --output artifacts/faiss_equivalence/run_1.json
```

1. Benchmark matrix report (required, bruteforce-first):

```bash
python scripts/benchmark_matrix.py --profile medium --mode exact --warmup 2 --loops 8 --seed 7 --max-memory-mb 4096 --output-dir artifacts/benchmark_matrix
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

## Benchmark profile presets (single-machine)

- `dev`: fast local smoke for iteration
- `medium`: default M3 Pro friendly profile
- `paper`: larger profile for publishable comparisons

Example:

```bash
python scripts/benchmark_matrix.py --profile dev --mode exact --output-dir artifacts/benchmark_matrix_dev
python scripts/benchmark_matrix.py --profile medium --mode exact --max-memory-mb 6144 --output-dir artifacts/benchmark_matrix_medium
```

Use `--max-memory-mb` to skip configurations that exceed your local memory budget.

## Dataset adapters

Use `scripts/datasets.py` helpers for local datasets:

- NumPy embeddings + JSON IDs/metadata (`load_numpy_bundle`)
- JSONL records with `id` + `embedding` fields (`load_jsonl_bundle`)
- Parquet records with `id` + `embedding` columns (`load_parquet_bundle`, requires pandas)
- Deterministic train/val/test assignment (`with_deterministic_splits`)

## Artifact contract checks

All reproducibility scripts now emit `artifact_contract_version` and validate required fields/types before writing publishable artifacts.

Contract validators live in:

- `scripts/artifact_contracts.py`

Validated outputs:

- ingest manifest
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

## Practical recipe blocks

### Retrieval quality study (single backend)

```bash
python scripts/rag_real_corpus_eval.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --output artifacts/real_corpus_runs/run_1.json --backend bruteforce --k 10 --ks 1,5,10 --loops 5 --threshold-recall 0.75 --threshold-ndcg 0.70
```

### Backend comparison study (optional exact overlap-gated)

```bash
python scripts/benchmark_matrix.py --profile medium --mode exact --min-flat-overlap 0.99 --max-memory-mb 4096 --output-dir artifacts/benchmark_matrix
python scripts/publishable_results.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --output artifacts/benchmark_matrix/publishable_results.v1.json
```

### Regression gate before release

```bash
python -m pytest -q tests/test_eval_surface_v1.py tests/test_profile_local.py
python -m pytest -q tests/test_artifact_contracts.py tests/test_credibility_audit.py tests/test_release_bundle.py tests/test_api_stability.py
python scripts/credibility_audit.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --publishable-summary artifacts/benchmark_matrix/publishable_results.v1.json --output artifacts/audit/credibility_audit.v1.json
```

## Standard artifact locations

- Baseline: `artifacts/rag_baseline_metrics.v1.json`
- Real-corpus runs: `artifacts/real_corpus_runs/run_*.json`
- Stability traces: `artifacts/testing_runs/stability_runs_*.jsonl`
- Stability summaries: `artifacts/testing_runs/stability_summary_*.json`
- Optional exact-equivalence (Faiss): `artifacts/faiss_equivalence/run_*.json`
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

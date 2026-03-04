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

## Standard artifact locations

- Baseline: `artifacts/rag_baseline_metrics.v1.json`
- Real-corpus runs: `artifacts/real_corpus_runs/run_*.json`
- Stability traces: `artifacts/testing_runs/stability_runs_*.jsonl`
- Stability summaries: `artifacts/testing_runs/stability_summary_*.json`
- Exact-equivalence: `artifacts/faiss_equivalence/run_*.json`

## Publish/private policy

Publish:

- benchmark summaries
- stability aggregate summaries and charts
- non-sensitive mock inputs and examples

Keep private:

- raw embeddings from private corpora
- private query sets and relevance labels tied to sensitive data
- metadata/ID mappings containing confidential content

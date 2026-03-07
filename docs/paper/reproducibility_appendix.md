# Reproducibility Appendix (v1)

## Environment requirements

- Python >= 3.10
- `numpy>=1.24`
- optional: `faiss-cpu` for Faiss backend paths

## Install

```bash
pip install -e ".[dev,ml]"
```

Optional:

```bash
pip install -e ".[faiss]"
```

## Canonical command sequence

```bash
python scripts/rag_baseline.py --output-dir artifacts --k 3
python scripts/rag_real_corpus_eval.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --output artifacts/real_corpus_runs/run_1.json --backend bruteforce --k 10 --ks 1,5,10 --loops 5
python scripts/stability_runs.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --backend bruteforce --run-count 200 --output-dir artifacts/testing_runs
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact --min-flat-overlap 0.99 --output artifacts/faiss_equivalence/run_1.json
python scripts/benchmark_matrix.py --mode exact --warmup 2 --loops 8 --seed 7 --min-flat-overlap 0.99 --output-dir artifacts/benchmark_matrix
python scripts/publishable_results.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --output artifacts/benchmark_matrix/publishable_results.v1.json
python scripts/credibility_audit.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --publishable-summary artifacts/benchmark_matrix/publishable_results.v1.json --output artifacts/audit/credibility_audit.v1.json
```

## Clean-environment smoke path

```bash
python scripts/repro_smoke.py --output-dir artifacts/repro_smoke
```

## Artifact contract policy

All publication-facing artifacts must include:

- `artifact_contract_version`
- protocol metadata
- environment metadata
- metric/performance payload sections relevant to the artifact type

Contracts are enforced by `scripts/artifact_contracts.py`.

## Privacy and publication policy

Publish:

- aggregate benchmark and stability summaries
- synthetic/mock example inputs

Keep private:

- raw private embeddings
- sensitive metadata and ID mappings
- private relevance labels tied to confidential corpora

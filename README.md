# Vector Engine v1.0.0

Reproducibility-first vector retrieval toolkit for local ML and IR workflows.

Vector Engine provides a clean Python API for vector indexing/search, evaluation, training utilities, and evidence-oriented benchmarking on a single machine.

## Why Vector Engine

- ANN libraries are fast but often backend-specific and low-level.
- Vector databases focus on serving and infra, not local experimentation loops.
- ML teams still need one local toolkit for ingest, retrieval, evaluation, and reproducibility.

Vector Engine focuses on that local workflow and keeps evidence outputs machine-checkable.

## Start Here

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip && python -m pip install vector-engine
```

## Install Options

PyPI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vector-engine
```

Local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,ml]"
python -m pytest -q
```

macOS arm64 + Python 3.12 constrained setup:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c requirements/constraints-macos-arm64-py312.txt -e ".[dev,ml]"
python -m pytest -q
```

Optional FAISS extra:

```bash
python -m pip install -e ".[faiss]"
```

If you hit `externally-managed-environment`, use a virtual environment as shown above.

## 60-Second Quickstart

```python
import numpy as np
from vector_engine import VectorArray, VectorIndex

xb = VectorArray.from_numpy(
    np.random.randn(1000, 384).astype("float32"),
    ids=[f"doc-{i}" for i in range(1000)],
    normalize=True,
)
xq = VectorArray.from_numpy(np.random.randn(2, 384).astype("float32"), normalize=True)

index = VectorIndex.create(xb, metric="cosine", backend="bruteforce")
results = index.search(xq, k=5)
print(results.ids[0], results.scores[0])
```

## v1.0.0 Surface

- Core: `VectorArray`, `VectorIndex`, `Metric`, `SearchResult`
- ML: `knn_classify`, `knn_regress`, `kmeans`, `KMeansResult`
- Training: `mine_hard_negatives`, `TripletBatch`
- Eval: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `retrieval_report`, `retrieval_report_detailed`, `batch_metrics_summary`, `retrieval_cohort_report`
- Ingest/connectors: `load_numpy_bundle`, `load_jsonl_bundle`, `load_parquet_bundle`, `with_deterministic_splits`, `scripts/ingest_dataset.py`

## API Contract Highlights

- `VectorArray` requires non-empty 2D tensors `(n, d)` and unique `int`/`str` IDs.
- `VectorIndex.search(..., k=...)` requires positive integer `k`.
- Metadata lengths align with vector row counts.
- `kmeans(..., random_state=...)` validates finite vectors and deterministic seeds.
- Retrieval evaluation validates malformed ground truth with stable `eval_error` prefixes.

## Data Ingest to Eval Recipe

1) Build a reproducible ingest bundle from JSONL:

```bash
python scripts/ingest_dataset.py \
  --input-jsonl artifacts/raw/source.jsonl \
  --output-dir artifacts/ingest_bundle \
  --id-field id \
  --text-field text \
  --embedding-dim 256 \
  --seed 7 \
  --label-field label \
  --split-field split \
  --query-group-field query_group \
  --ground-truth-field ground_truth
```

1) Run retrieval evaluation:

```bash
python scripts/rag_real_corpus_eval.py \
  --embeddings artifacts/ingest_bundle/embeddings.npy \
  --query-embeddings artifacts/repro_smoke/real_corpus_inputs/query_embeddings.npy \
  --ids artifacts/ingest_bundle/ids.json \
  --ground-truth artifacts/ingest_bundle/ground_truth.json \
  --metadata artifacts/ingest_bundle/metadata.json \
  --output artifacts/real_corpus_runs/run_1.json \
  --backend bruteforce \
  --k 6 \
  --ks 1,3,6 \
  --loops 5 \
  --threshold-recall 0.75 \
  --threshold-ndcg 0.70 \
  --threshold-p95-ms 120
```

Bundle outputs include:

- `embeddings.npy`, `ids.json`, `metadata.json`
- optional `labels.json`, `splits.json`, `query_groups.json`, `ground_truth.json`
- `ingest_manifest.v1.json` (contract-validated)

## Backends

| Backend | Search | Add | Save/Load | Custom Metric |
| --- | ---: | ---: | ---: | ---: |
| `bruteforce` | yes | yes | yes | yes |
| `faiss` | yes | yes | yes | no |

FAISS is optional. The required reproducibility path is bruteforce-safe.

## Reproducibility and Evidence

Recommended release evidence flow:

```bash
python scripts/repro_smoke.py --output-dir artifacts/repro_smoke
python scripts/benchmark_matrix.py --mode exact --warmup 2 --loops 8 --seed 7 --output-dir artifacts/benchmark_matrix
python scripts/publishable_results.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --output artifacts/benchmark_matrix/publishable_results.v1.json
python scripts/credibility_audit.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --publishable-summary artifacts/benchmark_matrix/publishable_results.v1.json --output artifacts/audit/credibility_audit.v1.json
```

## Examples

- `examples/minimal_rag_integration.py`
- `examples/hard_negative_training_batch.py`
- `examples/cohort_eval_workflow.py`
- `notebooks/01_semantic_search.ipynb`
- `notebooks/02_knn_baseline.ipynb`
- `notebooks/03_recommender_similarity.ipynb`

## Troubleshooting

- **`externally-managed-environment`**: install inside a venv.
- **No FAISS available**: run bruteforce path and skip overlap-gated FAISS checks.
- **Dimension mismatch**: ensure query and index embeddings share the same dimension.
- **NumPy segfault on macOS/Python 3.12**: reinstall with `requirements/constraints-macos-arm64-py312.txt` and run `python scripts/env_diagnostics.py`.

## Project Links

- `docs/releases/v1.0.0.md`
- `docs/releases/v1.0.0-checklist.md`
- `docs/reproducibility.md`
- `docs/use_cases.md`
- `docs/api_stability.md`
- `docs/research_claims.md`
- `LICENSE`
- `CITATION.cff`

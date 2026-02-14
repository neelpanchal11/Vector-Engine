# Vector Engine

ML-first vector computation and retrieval for Python.

Vector Engine provides one clean API for exact search, ANN backends, metadata-aware retrieval, and ML utilities such as kNN and retrieval metrics.

## Why this exists

- ANN libraries are powerful but low-level and backend-specific.
- Vector DBs solve infra and ops, but many ML workflows need fast local experimentation.
- Existing ML APIs do not offer a unified, backend-pluggable vector layer for embedding-heavy work.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev,ml]"
pip install -e ".[faiss]"
```

## API contracts (v0.2-alpha)

- `VectorArray` accepts only 2D arrays with shape `(n, d)` where `n > 0` and `d > 0`.
- `VectorArray` IDs must be unique and must be `int` or `str`.
- Metadata length must always align with number of vectors.
- `VectorIndex.search(..., k=...)` requires `k` to be an integer `> 0`.
- Score direction is explicit in `Metric.higher_is_better`:
  - cosine/ip: higher is better
  - l2: lower is better
- `kmeans(..., random_state=...)` requires an integer seed and finite vectors.
- `mine_hard_negatives` supports `top1`, `topk_sample`, and `distance_band` strategies with explicit exclusion controls.
- Retrieval eval validates malformed ground-truth shapes/types with stable `eval_error` messages.

## 60-second quickstart

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
res = index.search(xq, k=5)
print(res.ids[0], res.scores[0])
```

## Core API

- `VectorArray`: canonical vector storage with IDs and metadata.
- `Metric`: built-in and custom metric definitions.
- `VectorIndex`: backend-agnostic build/add/search/save/load.
- `vector_engine.ml`: `knn_classify`, `knn_regress`, `kmeans`.
- `vector_engine.training`: `mine_hard_negatives` with configurable sampling strategies.
- `vector_engine.eval`: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `retrieval_report`, `retrieval_report_detailed`, `batch_metrics_summary`.

## Backend support matrix

| Backend | Search | Add | Save/Load | Custom Metric |
| --- | ---: | ---: | ---: | ---: |
| `bruteforce` | yes | yes | yes | yes |
| `faiss` | yes | yes | yes | no |

## Examples and notebooks

- `notebooks/01_semantic_search.ipynb`
- `notebooks/02_knn_baseline.ipynb`
- `notebooks/03_recommender_similarity.ipynb`

## Benchmarks

Run:

```bash
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact
```

With exact-overlap gate and artifact output:

```bash
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact --min-flat-overlap 0.99 --output artifacts/benchmark_exact.json
```

ANN mode (optional):

```bash
python benchmarks/compare_bruteforce_vs_faiss.py --mode ann
```

The benchmark reports:

- `qps`: queries per second (higher is better)
- `latency_p50_ms` and `latency_p95_ms`: median and tail latency (lower is better)
- `overlap_vs_bruteforce`: top-k neighbor overlap against exact brute-force (closer to `1.0` is better)
- `memory_mb_estimate`: coarse in-process memory estimate for vector/query buffers

Recommended protocol for publishable results:

- Use fixed seed and fixed hardware notes.
- Warm up before timed runs.
- Run at least 3 repeated trials and report median numbers.
- Keep dataset size (`n`, `d`, `nq`, `k`) fixed across backend comparisons.

## Track B validation snapshot

Artifacts produced in this repo:

- Real-corpus style 3-run reports:
  - `artifacts/real_corpus_runs/run_1.json`
  - `artifacts/real_corpus_runs/run_2.json`
  - `artifacts/real_corpus_runs/run_3.json`
- Faiss Flat exact-equivalence checks (3 runs):
  - `artifacts/faiss_equivalence/run_1.json`
  - `artifacts/faiss_equivalence/run_2.json`
  - `artifacts/faiss_equivalence/run_3.json`
- 200-run stability study:
  - `artifacts/testing_runs/stability_runs_bruteforce_200.jsonl`
  - `artifacts/testing_runs/stability_summary_bruteforce_200.json`
  - `artifacts/testing_runs/stability_plot_p95_qps.png`

Observed outcomes for current mock/public-safe corpus:

- 3-run quality is identical across runs (`recall@1/3/6 = 1.0`, `ndcg@1/3/6 = 1.0`).
- 3-run latency envelope: p95 ranges from `0.0376 ms` to `0.0717 ms`.
- Faiss Flat exact mode achieves `overlap_vs_bruteforce = 1.0` for all 3 runs with `--min-flat-overlap 0.99`.
- In exact benchmark runs (`n=10000`, `d=128`, `nq=200`, `k=10`), Faiss Flat p95 latency is `4.17-15.03 ms` vs bruteforce `29.99-37.63 ms`.
- 200-run bruteforce study: p95 mean `0.0255 ms` (95% interval `0.0203-0.0547 ms`), QPS mean `188,097` (95% interval `117,499-214,111`).

Run the 200-run study:

```bash
python3 scripts/stability_runs.py \
  --embeddings artifacts/real_corpus_inputs/embeddings.npy \
  --query-embeddings artifacts/real_corpus_inputs/query_embeddings.npy \
  --ids artifacts/real_corpus_inputs/ids.json \
  --ground-truth artifacts/real_corpus_inputs/ground_truth.json \
  --metadata artifacts/real_corpus_inputs/metadata.json \
  --backend bruteforce \
  --k 6 \
  --ks 1,3,6 \
  --loops 5 \
  --run-count 200 \
  --output-dir artifacts/testing_runs \
  --threshold-recall 0.75 \
  --threshold-ndcg 0.70 \
  --threshold-p95-ms 120
```

Example result table format:

| Backend | QPS | p50 ms | p95 ms | overlap@k vs brute-force |
| --- | ---: | ---: | ---: | ---: |
| bruteforce | ... | ... | ... | 1.000 |
| faiss_flat | ... | ... | ... | ... |
| faiss_ivf (optional) | ... | ... | ... | ... |

## Project adoption checklist

- Install: `pip install -e ".[dev,ml]"` and optional `.[faiss]`.
- Validation: run `pytest -q`.
- Quality baseline: run `python scripts/rag_baseline.py`.
- Real corpus eval: run `python scripts/rag_real_corpus_eval.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --threshold-recall 0.75 --threshold-ndcg 0.70 --threshold-p95-ms 120`.
- Persistence: verify `VectorIndex.save/load` on your own embeddings snapshot.
- Performance: run benchmark script with your target `n`, `d`, `nq`, `k`.
- Integration: run `python examples/minimal_rag_integration.py`.

## v0.2-alpha feature preview

- `kmeans` now returns richer outputs (`labels`, `centers`, `inertia`, `n_iter`).
- Hard-negative mining supports `top1`, `topk_sample`, and `distance_band`, plus `exclude_ids` / `exclude_mask`.
- Retrieval evaluation includes `retrieval_report_detailed(include_per_query=...)` and `batch_metrics_summary(include_std=True)`.
- Public demo bootstrap is available under `demo_repo_template/`.

## Error cases

Stable error prefixes are used for fast debugging:

- `vector_array_error`: malformed array, IDs, metadata, subset lookup
- `metric_error`: unsupported or invalid metric definitions
- `index_error`: index lifecycle/search/add/persistence consistency issues
- `manifest_error`: missing/unsupported manifest fields or version

## Troubleshooting

- **Faiss not available**
  - Install with `pip install -e ".[faiss]"`.
- **Dimension mismatch at search/add**
  - Ensure both base vectors and query vectors use the same embedding dimension.
- **Metric confusion**
  - For cosine similarity, pass normalized vectors or set `normalize=True`.
- **Persistence load failure**
  - Check manifest version compatibility and whether artifacts were modified after save.

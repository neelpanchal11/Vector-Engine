# Integration Guides (v0.3.0)

## 1) Local RAG integration

Goal: index local embeddings and retrieve context for generation.

```bash
pip install -e ".[dev,ml]"
python examples/minimal_rag_integration.py
```

Expected result:

- returns top-k document IDs and scores
- prints a small retrieval metric summary

## 2) Batch retrieval evaluation

Goal: evaluate quality and latency over a query set with thresholds.

```bash
python scripts/rag_real_corpus_eval.py \
  --embeddings artifacts/real_corpus_inputs/embeddings.npy \
  --query-embeddings artifacts/real_corpus_inputs/query_embeddings.npy \
  --ids artifacts/real_corpus_inputs/ids.json \
  --ground-truth artifacts/real_corpus_inputs/ground_truth.json \
  --metadata artifacts/real_corpus_inputs/metadata.json \
  --output artifacts/real_corpus_runs/run_1.json \
  --backend bruteforce \
  --k 6 \
  --ks 1,3,6 \
  --loops 5 \
  --threshold-recall 0.75 \
  --threshold-ndcg 0.70 \
  --threshold-p95-ms 120
```

Expected result:

- writes JSON report with `metrics`, `performance`, `checks`, and `topk_ids`

## 3) Benchmark interpretation

Goal: compare exact bruteforce against Faiss Flat while enforcing overlap quality.

```bash
python benchmarks/compare_bruteforce_vs_faiss.py \
  --mode exact \
  --min-flat-overlap 0.99 \
  --output artifacts/faiss_equivalence/run_1.json
```

Interpretation:

- `overlap_vs_bruteforce` close to `1.0` indicates near-exact neighbor agreement
- `latency_p95_ms` indicates tail latency behavior
- `qps` gives throughput capacity under current benchmark configuration

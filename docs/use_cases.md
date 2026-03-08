# Use Cases

This document captures practical, runnable workflows that map directly to common local ML/retrieval tasks.

## 1) RAG retrieval baseline with quality checks

Goal: build a baseline retriever and track stable quality/performance metrics.

```bash
python scripts/rag_baseline.py --output-dir artifacts --k 3
python scripts/rag_real_corpus_eval.py --embeddings artifacts/repro_smoke/real_corpus_inputs/embeddings.npy --query-embeddings artifacts/repro_smoke/real_corpus_inputs/query_embeddings.npy --ids artifacts/repro_smoke/real_corpus_inputs/ids.json --ground-truth artifacts/repro_smoke/real_corpus_inputs/ground_truth.json --metadata artifacts/repro_smoke/real_corpus_inputs/metadata.json --output artifacts/real_corpus_runs/run_1.json --backend bruteforce --k 10 --ks 1,5,10 --loops 5
```

Primary outputs:

- `artifacts/rag_baseline_metrics.v1.json`
- `artifacts/real_corpus_runs/run_1.json`

## 2) Hard-negative mining for contrastive training

Goal: generate difficult negatives for triplet or contrastive objectives.

```bash
python examples/hard_negative_training_batch.py
```

What to verify:

- negatives differ from positives
- strategy and `random_state` produce deterministic batches
- `negative_scores` are captured for downstream curriculum logic

## 3) Cohort-based retrieval diagnostics

Goal: inspect retrieval quality by query segment (for example FAQ vs long-tail).

```bash
python examples/cohort_eval_workflow.py
```

What to verify:

- per-cohort metric differences (`recall@k`, `ndcg@k`)
- cohort size coverage
- worst-performing cohorts to prioritize data/model improvements

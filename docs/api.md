# API Reference (v0.3.0)

## Core

- `vector_engine.VectorArray`
  - `from_numpy`, `from_torch`, `from_jax`
  - `to_numpy`, `subset`, `shape`
- `vector_engine.Metric`
  - `cosine`, `l2`, `inner_product`, `custom`
- `vector_engine.VectorIndex`
  - `create`, `add`, `search`, `save`, `load`
- `vector_engine.SearchResult`

## ML Utilities

- `vector_engine.ml.knn_classify`
- `vector_engine.ml.knn_regress`
- `vector_engine.ml.kmeans`
- `vector_engine.ml.KMeansResult`

## Retrieval Evaluation

- `vector_engine.eval.precision_at_k`
- `vector_engine.eval.recall_at_k`
- `vector_engine.eval.ndcg_at_k`
- `vector_engine.eval.retrieval_report`
- `vector_engine.eval.retrieval_report_detailed`
- `vector_engine.eval.batch_metrics_summary`

## Training Helpers

- `vector_engine.training.mine_hard_negatives`

## Scripts and Integration

- `scripts/rag_baseline.py` (reproducible exact-first RAG baseline + artifact output)
- `scripts/rag_real_corpus_eval.py` (real-corpus evaluation and quality/latency gates)
- `scripts/stability_runs.py` (multi-run stability harness with JSONL run traces + aggregate summary)
- `examples/minimal_rag_integration.py` (drop-in retrieval example for application code)

## Integration Docs

- `docs/integration_guides.md` (local RAG, batch eval, benchmark interpretation)
- `docs/reproducibility.md` (standard command workflow + artifact policy)

## v1.0 Stability Contract

The following public surfaces are treated as stability-critical for v1.0:

- `vector_engine.VectorArray`, `vector_engine.Metric`, `vector_engine.VectorIndex`
- `vector_engine.ml.kmeans`, `vector_engine.ml.knn_classify`, `vector_engine.ml.knn_regress`
- `vector_engine.training.mine_hard_negatives`
- `vector_engine.eval.retrieval_report`, `retrieval_report_detailed`, `batch_metrics_summary`

Compatibility checks are enforced with regression tests in `tests/test_api_stability.py`.

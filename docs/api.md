# API Reference (v0.1)

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

## Retrieval Evaluation

- `vector_engine.eval.precision_at_k`
- `vector_engine.eval.recall_at_k`
- `vector_engine.eval.ndcg_at_k`
- `vector_engine.eval.retrieval_report`

## Training Helpers

- `vector_engine.training.mine_hard_negatives`

## Scripts and Integration

- `scripts/rag_baseline.py` (reproducible exact-first RAG baseline + artifact output)
- `examples/minimal_rag_integration.py` (drop-in retrieval example for application code)

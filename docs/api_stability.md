# API Stability Policy (v1.0)

This policy defines the compatibility guarantees for Vector Engine v1.x.

## Public API surface

The following modules and symbols are considered public and stability-scoped:

- `vector_engine.VectorArray`
- `vector_engine.VectorIndex`
- `vector_engine.Metric`
- `vector_engine.SearchResult`
- `vector_engine.ml.knn_classify`
- `vector_engine.ml.knn_regress`
- `vector_engine.ml.kmeans`
- `vector_engine.ml.KMeansResult`
- `vector_engine.training.mine_hard_negatives`
- `vector_engine.training.TripletBatch`
- `vector_engine.eval.precision_at_k`
- `vector_engine.eval.recall_at_k`
- `vector_engine.eval.ndcg_at_k`
- `vector_engine.eval.retrieval_report`
- `vector_engine.eval.retrieval_report_detailed`
- `vector_engine.eval.batch_metrics_summary`

## Compatibility guarantees

- No breaking signature changes in v1 minor/patch releases.
- Error prefix taxonomy remains stable for key categories.
- Persisted index manifest compatibility is maintained for supported versions.

## Deprecation policy

- Deprecations must be documented in release notes.
- Deprecated APIs remain available for at least one minor release window.
- Removal requires migration guidance and compatibility notes.

## Enforcement

- `tests/test_api_stability.py` validates signature-level stability.
- Core compatibility tests guard persistence and retrieval contracts.

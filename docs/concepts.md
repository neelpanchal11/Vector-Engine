# Concepts

## VectorArray

`VectorArray` is the canonical container for `float32` vectors in shape `(n, d)` plus aligned IDs and optional metadata.

Key properties:

- canonical contiguous NumPy buffer for backend interop
- support for NumPy, PyTorch, and JAX ingestion
- external IDs (`int`/`str`) with deterministic row alignment
- validation guarantees:
  - shape must be `(n, d)` with `n > 0`, `d > 0`
  - IDs must be unique and typed as `int` or `str`
  - metadata length must match number of rows

## Metric

`Metric` defines score semantics.

- `cosine`: similarity, higher is better
- `ip`: inner product similarity, higher is better
- `l2`: squared Euclidean distance, lower is better
- custom callable supported in brute-force backend

## VectorIndex

`VectorIndex` wraps the backend implementation and keeps:

- metric definition
- external IDs and metadata
- persistence manifest for compatibility

Main lifecycle:

1. `create(vectors, metric, backend)`
2. `search(queries, k)`
3. `add(new_vectors)`
4. `save(path)` and `load(path)`

## Persistence and compatibility

`VectorIndex.save()` writes:

- backend artifact directory
- `manifest.json` with version + schema fields
- `ids.json` and `metadata.json`

`VectorIndex.load()` validates manifest version and verifies artifact checksums when available.

## Error taxonomy

Common user-facing error prefixes:

- `vector_array_error`
- `metric_error`
- `index_error`
- `manifest_error`
- `ml_error` and `training_error` for ML helper validation failures
- `eval_error` for malformed retrieval-evaluation inputs

# Backends

Vector Engine routes index operations through a backend protocol:

- `build(xb, metric, config)`
- `add(xb)`
- `search(xq, k)`
- `save(path)`
- `load(path)`

## Bruteforce

- reference behavior for correctness
- supports built-in and custom metrics
- useful for testing and small/medium datasets

## Faiss

- optional dependency
- high-performance ANN/exact retrieval
- index configuration via `backend_config`, e.g.:

```python
VectorIndex.create(
    vectors,
    metric="ip",
    backend="faiss",
    backend_config={"index_factory": "IVF100,Flat", "nprobe": 8},
)
```

## Extending with new backends

Use backend registry:

```python
from vector_engine.backends import register_backend
register_backend("my_backend", MyBackendClass)
```

Your backend should expose capability flags such as:

- `supports_delete`
- `supports_custom_metric`
- `supports_persistence`

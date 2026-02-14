# Vector Engine Demo Template

This folder is a ready-to-copy template for a public demo repository that showcases Vector Engine in less than 15 minutes.

## What this demonstrates

- semantic retrieval end-to-end
- item similarity workflow
- exact-first quality checks
- benchmark interpretation

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/semantic_demo.py
python examples/item_similarity_demo.py
python examples/benchmark_demo.py
```

Expected setup time: ~10-15 minutes on a clean machine.

## Why not Faiss directly?

Faiss is excellent for search performance, but this demo highlights where Vector Engine adds value:

- one API for exact and ANN backends
- ID + metadata handling
- retrieval evaluation helpers
- reusable scripts for ML workflow iteration

## Suggested public repo structure

- `README.md`
- `examples/semantic_demo.py`
- `examples/item_similarity_demo.py`
- `examples/benchmark_demo.py`
- `artifacts/sample_report.json`
- `requirements.txt`

## Sample benchmark interpretation

Use `overlap_vs_bruteforce` as quality anchor and `latency_p95_ms` for user-facing tail performance.

## Example outputs

Semantic demo (abbreviated):

```text
metrics {'precision@1': 1.0, 'recall@1': 1.0, 'ndcg@1': 1.0}
top hits [['doc-0', 'doc-2']]
```

Item similarity demo (abbreviated):

```text
query item: item-0
neighbors: ['item-2', 'item-1', 'item-4']
```

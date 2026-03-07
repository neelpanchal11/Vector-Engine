# Limitations and Non-goals

This project prioritizes reproducible retrieval workflows and evaluation discipline for local-first ML iteration.

## Non-goals

- This is not a new ANN algorithm.
- This is not a managed vector database with distributed serving/ops features.
- This is not a replacement for production data governance tooling.

## Technical limitations

- Benchmark results are hardware/runtime dependent.
- Faiss paths require local Faiss availability and backend-specific tuning.
- Public artifact bundles cannot include private embeddings or sensitive metadata.
- Exact-vs-ANN overlap behavior is configuration sensitive (index factory, probes, vector distribution).

## Methodological limitations

- Synthetic or small benchmark datasets may not represent production workloads.
- Point estimates without spread are insufficient for performance claims.
- Metric quality depends on relevance-label quality and query set representativeness.

## Publication disclosure requirements

Any paper or public report should disclose:

- data source constraints and representativeness risks,
- hardware/runtime context,
- evaluation protocol and rerun variability,
- private-data exclusions and sanitization policy.

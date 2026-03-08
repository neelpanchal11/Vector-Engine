# Vector Engine KPI Charter (v1.0)

This charter defines release-blocking and directional KPIs for the v1.0 program.

## KPI tiers

- Tier 1: release blockers for v1.0.
- Tier 2: directional growth goals tracked per quarter.

## Tier 1 KPIs (release blockers)

| Area | KPI | Baseline (current repo) | v1.0 Gate |
| --- | --- | ---: | ---: |
| Research rigor | Claim-evidence coverage (claims mapped to code + artifact + method) | not formalized | 100% |
| Reproducibility | Clean-environment rerun success rate (3 environments) | not measured | >= 95% |
| Stability | Stability CV for p95 latency in canonical study | tracked in `stability_summary` | <= 0.35 |
| Stability | Stability CV for QPS in canonical study | tracked in `stability_summary` | <= 0.35 |
| Retrieval quality | Minimum recall gate in real-corpus eval | thresholded via script | >= configured threshold |
| Retrieval quality | Minimum NDCG gate in real-corpus eval | thresholded via script | >= configured threshold |
| Performance validity | Faiss Flat overlap vs brute-force in exact mode | overlap gate supported | >= 0.99 |
| API reliability | API freeze compatibility tests | partial | 100% passing |
| Package trust | Governance document completeness | incomplete | 100% present |

## Tier 2 KPIs (directional)

| Area | KPI | Quarter-1 target | Quarter-2 target |
| --- | --- | ---: | ---: |
| Community | Median issue first response time | <= 72h | <= 48h |
| Community | New contributor successful onboarding (clean setup) | >= 70% | >= 80% |
| Docs | External reviewer docs accuracy score | >= 8/10 | >= 8.5/10 |
| Adoption | Repro quickstart completion in <= 60 min | >= 80% | >= 90% |
| Quality | Critical defects open at release | 0 | 0 |

## Baseline capture procedure

Run the canonical commands and archive generated artifacts:

```bash
python scripts/rag_baseline.py --output-dir artifacts --k 3
python scripts/rag_real_corpus_eval.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --output artifacts/real_corpus_runs/run_1.json --backend bruteforce --k 10 --ks 1,5,10 --loops 5
python scripts/stability_runs.py --embeddings ... --query-embeddings ... --ids ... --ground-truth ... --backend bruteforce --run-count 200 --output-dir artifacts/testing_runs
python benchmarks/compare_bruteforce_vs_faiss.py --mode exact --min-flat-overlap 0.99 --output artifacts/faiss_equivalence/run_1.json
python scripts/benchmark_matrix.py --mode exact --warmup 2 --loops 8 --seed 7 --min-flat-overlap 0.99 --output-dir artifacts/benchmark_matrix
python scripts/publishable_results.py --matrix-summary artifacts/benchmark_matrix/matrix_summary.json --stability-summary artifacts/testing_runs/stability_summary_bruteforce_200.json --output artifacts/benchmark_matrix/publishable_results.v1.json
```

## Review cadence

- Week 0: baseline capture and KPI target lock.
- Every 2 weeks: KPI review with pass/fail states and corrective actions.
- Release gate: all Tier 1 KPIs must pass or be explicitly disclosed as exceptions in release notes.

## Milestone release cadence (post-v1)

- `v1.x.y` patch cadence: as-needed bugfix releases after contract-safe fixes.
- `v1.(x+1).0` minor cadence: every 4-6 weeks, only after evidence gates pass.

Required evidence package before each minor release:

1. targeted tests and API stability checks
2. benchmark matrix summary (`profile=medium` at minimum)
3. stability summary with CV fields
4. credibility audit report status `pass`
5. release bundle manifest with `ready_for_submission=true`

Do not cut a minor release when:

- any Tier 1 KPI fails without explicit waiver in release notes,
- artifact contracts fail validation,
- public API compatibility tests regress.

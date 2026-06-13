# Reproducibility

## Environment

- Python 3.12; `pip install -r requirements.txt` → numpy ≥1.24, scipy ≥1.10, duckdb ≥1.5, matplotlib ≥3.8.
- No GPU, no server. All analyses run single-machine; the heaviest step (computing STATS-CEB true
  cardinalities in DuckDB) is ~10–20 min and cached to `data/truth_cache.json` (computed once).

## Data provenance

- Benchmarks fetched by `scripts/fetch_data.sh` from
  `github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark` (STATS-CEB + JOB-light): true/sub-plan SQL and
  four released learned-estimator outputs (BayesCard, DeepDB, FLAT, NeuroCard). STATS-CEB true cardinalities
  are computed locally (DuckDB) from the shipped simplified STATS CSVs; JOB-light true cardinalities ship
  inline in the benchmark.
- A 5th, independent estimator is DuckDB's own native optimizer estimate (via `EXPLAIN (FORMAT json)`),
  cached to `data/duckdb_est.json`.

## Determinism

- All Monte-Carlo uses fixed seeds (`acs.py` 0xAC50, `step2.py`/`acs.py` 0xAC5E, bootstrap seed 7, splits
  seed 11). Results reproduce exactly given the same iteration order. The optimizer and κ are deterministic.

## Known limits (stated, not hidden)

- **Coverage:** 123/146 STATS-CEB queries have a complete plan lattice; 23 are dropped because at least one
  sub-plan join COUNT(\*) exceeds a 4 s timeout (intermediate results up to ~10¹⁰ rows). The dropped queries
  skew large/ill-conditioned. JOB-light is fully covered (67/70 usable; 3 dropped as degenerate — an exact
  plan tie, where the condition number is undefined). Full STATS-CEB coverage would need factorized COUNT
  for the monster joins.
- **Cost model:** C_out (sum of intermediate-result sizes). Real DBMS cost models differ; the *existence*
  of the regimes is the robust claim, not the exact crossover location.
- **ACS∞ scope:** a per-query measure (constant across estimators); it predicts *which queries* are
  regret-prone, not estimator-to-estimator variation within a query.

## Verifying correctness

`tests/test_core.py` contains 7 hand-computed checks (a 3-table chain with closed-form optimal plan,
regret, flip margin, and cost margin; DP optimum == brute-force enumeration; regret invariants). Run
`python tests/test_core.py` → `7/7 passed`.

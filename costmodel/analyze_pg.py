"""Analyze HYPOTHESIS-5: do C_out-geometry predictors (kappa, ACS_inf) predict REAL PostgreSQL
runtime regret, beating q-error? Reads costmodel-scratch/pg_results.csv + recomputes predictors.

NOTE deviations from the frozen H5 (reported honestly): runtime is k=1 (not k=3) and PPC-via-pinning
is not yet done -> this is the *runtime, preliminary* arm of H5a only.
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ce_metric_eval.experiment import prep_from
from ce_metric_eval.workload import load
from ce_metric_eval.acs_limit import acs_inf_limit
from ce_metric_eval.metrics import qerror

BASE = Path(str(Path(__file__).resolve().parent.parent / "data"))
SCRATCH = BASE / "costmodel-scratch"
BENCH = BASE / "End-to-End-CardEst-Benchmark"
ESTDIR = BENCH / "workloads/stats_CEB/sub_plan_queries/estimates"
SUBQ = BENCH / "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
METHODS = ["bayescard", "deepdb", "flat", "neurocard"]

# --- runtime regret from PG results ---
rows = list(csv.DictReader(open(SCRATCH / "pg_results.csv")))
def f(x):
    try: return float(x)
    except: return None
rt_regret = {}   # (qid, method) -> runtime ratio
for r in rows:
    qid = int(r["qid"]); tt = f(r["true_ms"])
    if tt is None or tt <= 0: continue
    for m in METHODS:
        mm = f(r[f"{m}_ms"])
        st = str(r[f"{m}_status"])
        if mm is not None and (st.startswith("ok") or st == "timeout"):  # timeout = >=120s lower bound
            rt_regret[(qid, m)] = max(mm / tt, 1e-9)

# --- per-method per-query q-error (from est vs truth on the join sub-plans) ---
lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
qid_lines = defaultdict(list); key = []
for i, l in enumerate(lines):
    body, qid = l.rsplit("||", 1); qid_lines[int(qid)].append(i); key.append(body.strip().rstrip(";").strip())
est = {m: [float(x) for x in (ESTDIR / f"stats_CEB_sub_queries_{m}.txt").read_text().split()] for m in METHODS}
truth = json.loads((BASE / "truth_cache.json").read_text())
tv = [truth.get(k, -1.0) for k in key]
def qerr_qm(qid, m):
    vals = [qerror(est[m][i], tv[i]) for i in qid_lines[qid] if tv[i] > 0]
    return max(vals) if vals else None

# --- predictors from C_out geometry (kappa, ACS_inf) per qid ---
prepared = {P["qid"]: P for P in prep_from(load())}
rng = np.random.default_rng(5)
kappa = {q: P["kf"] for q, P in prepared.items()}
acs = {q: acs_inf_limit(P, draws=2000, rng=rng) for q, P in prepared.items()}

# --- assemble per-(qid,method) and per-qid ---
pooled = []   # (regret, acs, qerr, kappa)
for (qid, m), reg in rt_regret.items():
    if qid in acs:
        pooled.append((reg, acs[qid], qerr_qm(qid, m), kappa[qid]))
per_q = {}
for qid in set(q for q, _ in rt_regret):
    if qid not in acs: continue
    regs = [rt_regret[(qid, m)] for m in METHODS if (qid, m) in rt_regret]
    qs = [qerr_qm(qid, m) for m in METHODS if (qid, m) in rt_regret]
    if regs:
        per_q[qid] = (np.mean(regs), acs[qid], np.mean([x for x in qs if x]), kappa[qid])

def sp(a, b): return spearmanr(a, b).statistic if len(a) >= 5 else float("nan")

print(f"pooled (q,method) points: {len(pooled)}   per-query: {len(per_q)}")
if not pooled or not per_q:
    print("NO USABLE POINTS YET (run may be incomplete)."); sys.exit(0)
pr = [p[0] for p in pooled]
print(f"runtime-regret: median={np.median(pr):.3f}  frac>1.1={np.mean([x>1.1 for x in pr]):.2f}  max={max(pr):.2f}")

# PER-QUERY (the H5a level): mean runtime regret vs predictors
pq = list(per_q.values())
reg = np.array([x[0] for x in pq]); acsv = np.array([x[1] for x in pq])
qv = np.array([x[2] for x in pq]); kv = np.array([x[3] for x in pq])
r_acs = sp(reg, acsv); r_q = sp(reg, qv); r_k = sp(reg, kv)
print("\n=== PER-QUERY (runtime regret) ===")
print(f"  rho(regret, ACS_inf) = {r_acs:+.3f}")
print(f"  rho(regret, q-error) = {r_q:+.3f}")
print(f"  rho(regret, kappa)   = {r_k:+.3f}")
# bootstrap margin ACS vs q-error (H5a)
rngb = np.random.default_rng(7); n = len(reg); ms = []
for _ in range(1000):
    idx = rngb.integers(0, n, n)
    ms.append(sp(reg[idx], acsv[idx]) - sp(reg[idx], qv[idx]))
ms = np.array(ms); ci = (np.nanpercentile(ms, 2.5), np.nanpercentile(ms, 97.5))
print(f"  margin(ACS - q) 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]   H5a(>0.10)? {ci[0] > 0.10}")

# POOLED per-(q,method)
preg = np.array([p[0] for p in pooled]); pacs = np.array([p[1] for p in pooled]); pq2 = np.array([p[2] for p in pooled])
print("\n=== POOLED (q,method) ===")
print(f"  rho(regret, ACS_inf)={sp(preg,pacs):+.3f}   rho(regret, q-error)={sp(preg,pq2):+.3f}")
print("\n(runtime arm; set REPS via run_pg.py arg. PPC arm infeasible — see RESULTS-5.)")

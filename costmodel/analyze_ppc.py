"""Analyze PPC-regret arm of HYPOTHESIS-5: does ACS_inf (C_out geometry) predict real PostgreSQL
plan-cost regret (noise-free, full coverage), beating q-error?  Reads ppc_results.csv.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ce_metric_eval.experiment import prep_from
from ce_metric_eval.workload import load
from ce_metric_eval.acs_limit import acs_inf_limit
from ce_metric_eval.metrics import qerror

BASE = Path(str(Path(__file__).resolve().parent.parent / "data"))
SCRATCH = BASE / "costmodel-scratch"; BENCH = BASE / "End-to-End-CardEst-Benchmark"
ESTDIR = BENCH / "workloads/stats_CEB/sub_plan_queries/estimates"
SUBQ = BENCH / "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
METHODS = ["bayescard", "deepdb", "flat", "neurocard"]
PINNED_ONLY = "--pinned" in sys.argv

rows = list(csv.DictReader(open(SCRATCH / "ppc_results.csv")))
def f(x):
    try: return float(x)
    except: return None
ppc = {}
for r in rows:
    qid = int(r["qid"])
    for m in METHODS:
        v = f(r.get(f"{m}_ppc"))
        if v is not None and v > 0:
            if PINNED_ONLY and r.get(f"{m}_pinned") not in ("True", "true", "1"):
                continue
            ppc[(qid, m)] = max(v, 1.0)

# per-method per-query q-error
lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
qid_lines = defaultdict(list); key = []
for i, l in enumerate(lines):
    body, qid = l.rsplit("||", 1); qid_lines[int(qid)].append(i); key.append(body.strip().rstrip(";").strip())
est = {m: [float(x) for x in (ESTDIR / f"stats_CEB_sub_queries_{m}.txt").read_text().split()] for m in METHODS}
truth = json.loads((BASE / "truth_cache.json").read_text()); tv = [truth.get(k, -1.0) for k in key]
def qerr_qm(qid, m):
    vals = [qerror(est[m][i], tv[i]) for i in qid_lines[qid] if tv[i] > 0]
    return max(vals) if vals else None

prepared = {P["qid"]: P for P in prep_from(load())}
rng = np.random.default_rng(5)
kappa = {q: P["kf"] for q, P in prepared.items()}
acs = {q: acs_inf_limit(P, draws=2000, rng=rng) for q, P in prepared.items()}

def sp(a, b): return spearmanr(a, b).statistic if len(a) >= 5 else float("nan")

pooled = [(reg, acs[q], qerr_qm(q, m), kappa[q]) for (q, m), reg in ppc.items() if q in acs]
per_q = {}
for q in set(qq for qq, _ in ppc):
    if q not in acs: continue
    regs = [ppc[(q, m)] for m in METHODS if (q, m) in ppc]
    qs = [qerr_qm(q, m) for m in METHODS if (q, m) in ppc]
    if regs:
        per_q[q] = (np.mean(regs), acs[q], np.mean([x for x in qs if x]), kappa[q])

print(f"PPC arm{' (pinned only)' if PINNED_ONLY else ''}: pooled={len(pooled)}  per-query={len(per_q)}")
allr = [p[0] for p in pooled]
print(f"PPC-regret: median={np.median(allr):.3f}  frac>1.05={np.mean([x>1.05 for x in allr]):.2f}  max={max(allr):.2f}")
pq = list(per_q.values())
reg = np.array([x[0] for x in pq]); acsv = np.array([x[1] for x in pq]); qv = np.array([x[2] for x in pq]); kv = np.array([x[3] for x in pq])
print("\n=== PER-QUERY (PPC regret) ===")
print(f"  rho(regret, ACS_inf) = {sp(reg, acsv):+.3f}")
print(f"  rho(regret, q-error) = {sp(reg, qv):+.3f}")
print(f"  rho(regret, kappa)   = {sp(reg, kv):+.3f}")
rngb = np.random.default_rng(7); n = len(reg); ms = []
for _ in range(1000):
    idx = rngb.integers(0, n, n)
    ms.append(sp(reg[idx], acsv[idx]) - sp(reg[idx], qv[idx]))
ms = np.array(ms); ci = (np.nanpercentile(ms, 2.5), np.nanpercentile(ms, 97.5))
print(f"  margin(ACS - q) 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]   H5a(>0.10)? {ci[0] > 0.10}")
preg = np.array([p[0] for p in pooled]); pacs = np.array([p[1] for p in pooled]); pq2 = np.array([p[2] for p in pooled])
print(f"\n=== POOLED (q,method) ===  rho(ACS)={sp(preg,pacs):+.3f}  rho(q-error)={sp(preg,pq2):+.3f}")

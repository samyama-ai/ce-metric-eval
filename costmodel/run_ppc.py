"""PPC-regret (noise-free, full coverage): cost_true(P_method) / cost_true(P*), where plans are
pinned with pg_hint_plan and costed under injected TRUE cardinalities.

Per query q, per method E:
  P_E   = plan PG chooses under E's join cards (EXPLAIN, no execute)
  hints = Leading + join/scan method hints extracted from P_E
  cost_true(P_E) = cost of  /*+ hints */ <q>  under injected TRUE cards (plan pinned)
  cost_true(P*)  = cost of  <q>  under injected TRUE cards (free optimizer)
  PPC-regret(q,E) = cost_true(P_E) / cost_true(P*)

Usage: python run_ppc.py [N_test]   (N_test>0 = test/validate on first N complete queries)
"""
import json, subprocess, csv, sys, re
from collections import defaultdict
from pathlib import Path

BASE = Path(str(Path(__file__).resolve().parent.parent / "data"))
BENCH = BASE / "End-to-End-CardEst-Benchmark"; SCRATCH = BASE / "costmodel-scratch"
SUBQ = BENCH / "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
FULLQ = BENCH / "workloads/stats_CEB/stats_CEB.sql"; ESTDIR = BENCH / "workloads/stats_CEB/sub_plan_queries/estimates"
METHODS = ["bayescard", "deepdb", "flat", "neurocard"]

# ---- plan -> pg_hint_plan hints ----
WRAP = {"Hash", "Materialize", "Gather", "Gather Merge", "Result", "Sort", "Aggregate", "Limit"}
JOIN = {"Hash Join": "HashJoin", "Merge Join": "MergeJoin", "Nested Loop": "NestLoop"}
SCAN = {"Seq Scan": "SeqScan", "Index Scan": "IndexScan", "Index Only Scan": "IndexOnlyScan",
        "Bitmap Heap Scan": "BitmapScan", "Tid Scan": "TidScan"}

def unwrap(n):
    while n["Node Type"] in WRAP and "Plans" in n:
        n = n["Plans"][0]
    return n

def walk(n):
    """-> (leading_expr, set(aliases), [hints])"""
    n = unwrap(n)
    nt = n["Node Type"]
    if nt in JOIN:
        parts = [walk(c) for c in n["Plans"]]
        al = set().union(*[p[1] for p in parts])
        lead = tuple(p[0] for p in parts)
        hints = sum([p[2] for p in parts], []) + [f"{JOIN[nt]}({' '.join(sorted(al))})"]
        return lead, al, hints
    if nt in SCAN or "Scan" in nt:
        a = n.get("Alias") or n.get("Relation Name")
        return a, {a}, [f"{SCAN.get(nt, 'SeqScan')}({a})"]
    if "Plans" in n:
        return walk(n["Plans"][0])
    a = n.get("Alias") or n.get("Relation Name") or "?"
    return a, {a}, []

def render(e):
    return e if isinstance(e, str) else "(" + " ".join(render(x) for x in e) + ")"

def plan_to_hints(plan_json):
    lead, al, hints = walk(plan_json["Plan"])
    if len(al) >= 2:
        hints = [f"Leading({render(lead)})"] + hints
    return "/*+ " + " ".join(hints) + " */"

def plan_sig(plan_json):
    lead, _, hints = walk(plan_json["Plan"])
    return (render(lead), tuple(sorted(h for h in hints)))

# ---- workload ----
lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
qid_lines = defaultdict(list); key = []
for i, l in enumerate(lines):
    body, qid = l.rsplit("||", 1); qid_lines[int(qid)].append(i); key.append(body.strip().rstrip(";").strip())
est = {m: [float(x) for x in (ESTDIR / f"stats_CEB_sub_queries_{m}.txt").read_text().split()] for m in METHODS}
import json as _j
truth = _j.loads((BASE / "truth_cache.json").read_text())
tv = [truth.get(k, -1.0) for k in key]
full = {}
for i, l in enumerate(l for l in FULLQ.read_text().splitlines() if l.strip()):
    full[i] = l.split("||", 1)[1].strip()
complete = [q for q in sorted(qid_lines) if all(tv[i] > 0 for i in qid_lines[q]) and q in full]

def psql(sqls):
    cmd = ["docker", "exec", "ce-pg", "psql", "-U", "postgres", "-d", "stats", "-t", "-A", "-c", sqls]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def explain(sql, slice_vals, hint=""):
    (SCRATCH / "slice.txt").write_text("\n".join(repr(float(v)) for v in slice_vals) + "\n")
    stmt = (f"SET ml_joinest_enabled=true; SET ml_joinest_fname='/scratch/slice.txt'; "
            f"{hint} EXPLAIN (FORMAT JSON) {sql}")
    r = psql(stmt)
    try:
        p = json.loads(r.stdout.strip())[0]
        return p, float(p["Plan"]["Total Cost"])
    except Exception:
        return None, None

def slice_of(q, m):
    return [(tv[i] if m == "true" else est[m][i]) for i in qid_lines[q]]

TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 0
qs = complete[:TEST] if TEST else complete
print(f"complete queries: {len(complete)}  running: {len(qs)}", flush=True)
rows = []
forced_ok = forced_bad = 0
for n, q in enumerate(qs):
    sql = full[q]
    p_star, cost_star = explain(sql, slice_of(q, "true"))
    if cost_star is None or cost_star <= 0:
        continue
    rec = {"qid": q, "n_tables": len(set().union(*[walk(p_star["Plan"])[1]]))}
    for m in METHODS:
        p_e, _ = explain(sql, slice_of(q, m))
        if p_e is None:
            rec[f"{m}_ppc"] = None; continue
        hints = plan_to_hints(p_e)
        p_forced, cost_forced = explain(sql, slice_of(q, "true"), hint=hints)
        if p_forced is None:
            rec[f"{m}_ppc"] = None; continue
        ok = plan_sig(p_forced) == plan_sig(p_e)
        forced_ok += ok; forced_bad += (not ok)
        rec[f"{m}_ppc"] = cost_forced / cost_star
        rec[f"{m}_pinned"] = ok
    rows.append(rec)
    if TEST:
        print(f"  q{q}: cost*={cost_star:.0f}  " +
              " ".join(f"{m}={rec.get(f'{m}_ppc')!s:.6}/{ 'pin' if rec.get(f'{m}_pinned') else 'NOPIN'}" for m in METHODS), flush=True)
    elif (n + 1) % 20 == 0:
        print(f"  {n+1}/{len(qs)}  pinned ok={forced_ok} bad={forced_bad}", flush=True)

out = SCRATCH / "ppc_results.csv"
cols = ["qid", "n_tables"] + [f"{m}_ppc" for m in METHODS] + [f"{m}_pinned" for m in METHODS]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} queries); plan-pin ok={forced_ok} bad={forced_bad}", flush=True)

"""HYPOTHESIS-5 driver: measure real PostgreSQL plan regret (runtime + plan cost) on STATS-CEB,
by injecting each estimator's join cardinalities into the patched PG (ml_joinest), per query.

For each complete query q and method in {bayescard, deepdb, flat, neurocard, true}:
  - write q's join sub-plan estimate slice (in sub_queries.sql order = PG's request order),
  - fresh psql connection: SET ml_joinest_enabled + fname; EXPLAIN (ANALYZE, FORMAT JSON) <full query>,
  - record Execution Time (ms) and Total Cost.
runtime-regret(q,method) = time(method)/time(true).  Output: costmodel-scratch/pg_results.csv
"""
import json, subprocess, csv, sys, time, os
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("CE_BASE", str(Path(__file__).resolve().parent.parent / "data")))
BENCH = BASE / "End-to-End-CardEst-Benchmark"
SCRATCH = BASE / "costmodel-scratch"
SUBQ = BENCH / "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
FULLQ = BENCH / "workloads/stats_CEB/stats_CEB.sql"
ESTDIR = BENCH / "workloads/stats_CEB/sub_plan_queries/estimates"
TRUTH = BASE / "truth_cache.json"
METHODS = ["bayescard", "deepdb", "flat", "neurocard"]
TIMEOUT_S = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ONLY = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all; N>0 = first N complete queries (smoke)
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # timed reps per (q,method); median taken

# --- parse sub-plan lines (qid + truth-lookup key), per-qid ordered global indices ---
lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
key = []          # per global line: truth_cache key
qid_of = []
for l in lines:
    body, qid = l.rsplit("||", 1)
    key.append(body.strip().rstrip(";").strip())
    qid_of.append(int(qid))
qid_lines = defaultdict(list)
for i, q in enumerate(qid_of):
    qid_lines[q].append(i)

# --- estimator values per line, truth per line ---
est = {m: [float(x) for x in (ESTDIR / f"stats_CEB_sub_queries_{m}.txt").read_text().split()] for m in METHODS}
truth_cache = json.loads(TRUTH.read_text())
true_val = [truth_cache.get(k, -1.0) for k in key]

# --- full queries: qid -> SQL ---
full = {}
for i, l in enumerate(l for l in FULLQ.read_text().splitlines() if l.strip()):
    _, sql = l.split("||", 1)
    full[i] = sql.strip()

# complete query = all its sub-plan lines have a positive true value
complete = [q for q in sorted(qid_lines) if all(true_val[i] > 0 for i in qid_lines[q]) and q in full]
if ONLY:
    complete = complete[:ONLY]
print(f"complete queries: {len(complete)} (timeout {TIMEOUT_S}s)", flush=True)


def run(slice_vals, sql):
    (SCRATCH / "slice.txt").write_text("\n".join(repr(float(v)) for v in slice_vals) + "\n")
    cmd = ["docker", "exec", "ce-pg", "psql", "-U", "postgres", "-d", "stats", "-t", "-A", "-c",
           f"SET ml_joinest_enabled=true; SET ml_joinest_fname='/scratch/slice.txt'; "
           f"SET statement_timeout='{TIMEOUT_S}s'; "
           f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING ON, SUMMARY ON) {sql}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S + 30)
    except subprocess.TimeoutExpired:
        return None, None, "proc_timeout"
    if "statement timeout" in r.stderr:
        return None, None, "stmt_timeout"
    out = r.stdout.strip()
    if not out or "ERROR" in r.stderr:
        return None, None, (r.stderr.strip()[:80] or "no_output")
    try:
        plan = json.loads(out)
        p = plan[0]
        return float(p["Execution Time"]), float(p["Plan"]["Total Cost"]), "ok"
    except Exception as e:
        return None, None, f"parse:{e}"


rows = []
t0 = time.time()
for n, q in enumerate(complete):
    sql = full[q]
    rec = {"qid": q, "n_subplans": len(qid_lines[q])}
    for m in METHODS + ["true"]:
        sl = [(true_val[i] if m == "true" else est[m][i]) for i in qid_lines[q]]
        times = []; cost = None; timed_out = False
        for _ in range(REPS):
            ms, c, st = run(sl, sql)
            if ms is not None:
                times.append(ms); cost = c if cost is None else c
            elif "timeout" in str(st):
                timed_out = True; break        # slow plan: record as >=timeout, no point retrying
            else:
                break                          # hard error
        if times:
            med = sorted(times)[len(times) // 2]; status = f"ok{len(times)}"
        elif timed_out:
            med = float(TIMEOUT_S * 1000); status = "timeout"   # lower-bound runtime (>= timeout)
        else:
            med = None; status = "err"
        rec[f"{m}_ms"] = med
        rec[f"{m}_cost"] = cost
        rec[f"{m}_status"] = status
    rows.append(rec)
    if (n + 1) % 10 == 0 or ONLY:
        ok = sum(1 for r in rows if r["true_ms"] is not None)
        print(f"  {n+1}/{len(complete)} done, {ok} with true runtime, {time.time()-t0:.0f}s", flush=True)

out_csv = SCRATCH / "pg_results.csv"
cols = ["qid", "n_subplans"] + [f"{m}_{x}" for m in METHODS + ["true"] for x in ("ms", "cost", "status")]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"wrote {out_csv}  ({len(rows)} queries, {time.time()-t0:.0f}s)", flush=True)

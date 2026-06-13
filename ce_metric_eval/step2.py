"""ACS Step 2 = FROZEN HYPOTHESIS-4. Robustness (bootstrap CI), generalization (held-out
split), independence (DuckDB-native 5th estimator), and a cheap analytic proxy for ACS_inf.

Reproduce: python -m ce_metric_eval.step2
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .experiment import prep_from, tier2
from .workload import (load, load_joblight, build_db, parse_line, SUBQ)
from .geometry import regret, oracle
from .metrics import qerror
from .optimizer import enumerate_plans, cost_of_tree
from .acs import acs_at

OUT = Path(__file__).resolve().parent.parent / "data" / "results"
DUCK_CACHE = Path(__file__).resolve().parent.parent / "data" / "duckdb_est.json"
SEED = 0xAC5E


# ---------- DuckDB native estimator (5th, independent) ----------
def _root_estcard(node):
    ei = node.get("extra_info", {})
    c = ei.get("Estimated Cardinality") if isinstance(ei, dict) else None
    if c is None:
        c = node.get("cardinality")
    return float(c) if c is not None else None


def build_duckdb_estimates():
    if DUCK_CACHE.exists():
        raw = json.loads(DUCK_CACHE.read_text())
        return {int(q): {frozenset(k.split(",")): v for k, v in d.items()} for q, d in raw.items()}
    con = build_db()
    out = defaultdict(dict)
    for line in SUBQ.read_text().splitlines():
        if not line.strip():
            continue
        sql, qid, aset, _ = parse_line(line)
        try:
            res = con.execute("EXPLAIN (FORMAT json) " + sql.replace("COUNT(*)", "*")).fetchone()
            plan = json.loads(res[-1])
            root = plan[0] if isinstance(plan, list) else plan
            est = _root_estcard(root)
        except Exception:
            est = None
        if est and est > 0:
            out[qid][aset] = est
    DUCK_CACHE.write_text(json.dumps(
        {str(q): {",".join(sorted(k)): v for k, v in d.items()} for q, d in out.items()}))
    return out


# ---------- per-query measures ----------
def acs_inf_map(prepared, draws=800):
    rng = np.random.default_rng(0xAC5E)
    return {P["qid"]: acs_at(P, 5.0, draws, rng) for P in prepared}


def analytic_proxies(P):
    true = P["true"]
    rs = sorted(cost_of_tree(t, oracle(true)) for t in enumerate_plans(P["q"]))
    opt = rs[0]
    ratios = np.array([r / opt for r in rs])
    return {
        "mean": float(ratios.mean()),
        "median": float(np.median(ratios)),
        "geomean": float(np.exp(np.log(ratios).mean())),
        "2nd": float(ratios[1]) if len(ratios) > 1 else 1.0,
        "top10": float(ratios[:10].mean()),
    }


def est_regret_q(P, est_dict):
    true = P["true"]; subs = P["subsets"]
    est = {s: max(float(est_dict[s]), 1.0) for s in subs if s in est_dict}
    if len(est) < len(subs):
        return None, None
    return regret(P["q"], oracle(true), oracle(est)), max(qerror(est[s], true[s]) for s in subs)


def margin(reg, acs, q):
    return spearmanr(reg, acs).statistic - spearmanr(reg, q).statistic


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = prep_from(load())
    jl = prep_from(load_joblight())
    print(f"prepared: STATS={len(stats)} job-light={len(jl)}")

    acs_s = acs_inf_map(stats)
    acs_j = acs_inf_map(jl)

    def per_query(prepared, acsmap):
        rows = tier2(prepared); by = defaultdict(list)
        for r in rows:
            by[r["qid"]].append(r)
        qids = [P["qid"] for P in prepared]
        reg = np.array([np.mean([x["regret"] for x in by[q]]) for q in qids])
        qq = np.array([np.mean([x["maxq"] for x in by[q]]) for q in qids])
        acs = np.array([acsmap[q] for q in qids])
        return np.array(qids), reg, qq, acs

    qids, reg, qq, acs = per_query(stats, acs_s)

    # H4a bootstrap margin (STATS)
    rng = np.random.default_rng(7)
    n = len(qids); mlist = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        mlist.append(margin(reg[idx], acs[idx], qq[idx]))
    mlist = np.array(mlist)
    ci = (float(np.percentile(mlist, 2.5)), float(np.percentile(mlist, 97.5)))
    rho_acs = spearmanr(reg, acs).statistic
    H4a = ci[0] > 0.15

    # H4c held-out split
    rng2 = np.random.default_rng(11); perm = rng2.permutation(n); half = n // 2
    A, B = perm[:half], perm[half:]
    mA, mB = margin(reg[A], acs[A], qq[A]), margin(reg[B], acs[B], qq[B])
    H4c = (mA > 0.15) and (mB > 0.15)

    # H4d independent estimator (DuckDB native)
    duck = build_duckdb_estimates()
    dreg, dq, dacs = [], [], []
    for P in stats:
        if P["qid"] in duck:
            r, q = est_regret_q(P, duck[P["qid"]])
            if r is not None:
                dreg.append(r); dq.append(q); dacs.append(acs_s[P["qid"]])
    dreg, dq, dacs = map(np.array, (dreg, dq, dacs))
    if len(dreg) >= 10:
        d_margin = spearmanr(dreg, dacs).statistic - spearmanr(dreg, dq).statistic
        H4d = d_margin > 0.10
    else:
        d_margin, H4d = float("nan"), False

    # H4e analytic proxy
    proxA = {k: [] for k in ["mean", "median", "geomean", "2nd", "top10"]}
    for P in stats:
        pp = analytic_proxies(P)
        for k in proxA:
            proxA[k].append(pp[k])
    prox_tbl = {}
    for k, vals in proxA.items():
        vals = np.array(vals)
        prox_tbl[k] = {"rho_vs_MC_acs": float(spearmanr(vals, acs).statistic),
                       "rho_vs_regret": float(spearmanr(reg, vals).statistic)}
    H4e = any(d["rho_vs_MC_acs"] >= 0.70 and abs(d["rho_vs_regret"] - rho_acs) <= 0.10
              for d in prox_tbl.values())

    # H4f regime guard (job-light)
    jq, jreg, jqq, jacs = per_query(jl, acs_j)
    jl_margin = margin(jreg, jacs, jqq)
    H4f = not (jl_margin > 0.15)

    confirm = H4a and H4c and H4d and H4f
    print(f"\n=== STATS-CEB ===  n={n}")
    print(f"  rho(regret, ACS_inf) = {rho_acs:.3f}   rho(regret, q) = {spearmanr(reg,qq).statistic:.3f}")
    print(f"  H4a margin 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]  (>0.15? {H4a})")
    print(f"  H4c held-out margins: A={mA:.3f} B={mB:.3f}  ({H4c})")
    print(f"  H4d DuckDB-native estimator: margin={d_margin:.3f} (n={len(dreg)})  (>0.10? {H4d})")
    print(f"  H4e analytic proxies (rho vs MC-ACS / vs regret):")
    for k, d in prox_tbl.items():
        print(f"        {k:7s}: {d['rho_vs_MC_acs']:+.3f} / {d['rho_vs_regret']:+.3f}")
    print(f"      H4e usable proxy? {H4e}")
    print(f"  H4f job-light margin = {jl_margin:+.3f}  (NOT >0.15? {H4f})")
    print(f"\n>>> STEP-2 VERDICT: {'CONFIRM' if confirm else 'NOT CONFIRMED'}")
    summary = {"rho_acs": rho_acs, "margin_ci": ci, "heldout": [mA, mB],
               "duckdb_margin": d_margin, "duckdb_n": len(dreg),
               "proxies": prox_tbl, "jl_margin": jl_margin,
               "H4a": H4a, "H4c": H4c, "H4d": H4d, "H4e": H4e, "H4f": H4f,
               "verdict": "CONFIRM" if confirm else "NOT CONFIRMED"}
    (OUT / "step2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT}/step2_summary.json")


if __name__ == "__main__":
    main()

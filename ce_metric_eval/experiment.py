"""Day-1 headline experiment = the kill-test from HYPOTHESIS.md.

Tier-1 (controlled mechanism): inject lognormal noise on REAL STATS-CEB geometry, sweep
  sigma, and test whether conditioning on the plan-condition-number kappa collapses the
  q-error -> regret scatter, per the FROZEN decision rule.
Tier-2 (in-the-wild): the 4 released estimators (bayescard/deepdb/flat/neurocard).

Decision rule (frozen): H1 supported iff pooled |r| < 0.5 AND (r_cond - r_pool) >= 0.2 AND
  r_cond >= 0.7, under BOTH operationalizations (kappa-quartile-stratified weighted Spearman
  AND partial Spearman controlling for log kappa), reported for kappa_flip (primary) and
  kappa_margin (secondary). Reported transparently; no post-hoc threshold edits.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from .geometry import regret, kappa_flip, kappa_margin, oracle
from .metrics import qerror
from .optimizer import connected_subsets, count_plans
from .workload import load

SEED = 0xC0FFEE
SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]   # lognormal log-space std (controls q-error)
DRAWS = 20
OUT = Path(__file__).resolve().parent.parent / "data" / "results"


def prep_queries():
    return prep_from(load())


def prep_from(qs):
    prepared = []
    skip = {"incomplete": 0, "plan_explosion": 0, "degenerate": 0, "single_table": 0}
    for qid, lq in sorted(qs.items()):
        if len(lq.aliases) < 2:
            skip["single_table"] += 1
            continue
        ok, have, need = lq.complete()
        if not ok:
            skip["incomplete"] += 1
            continue
        q = lq.to_query()
        if count_plans(q) > 200_000:
            skip["plan_explosion"] += 1
            continue
        subsets = connected_subsets(q, min_size=2)
        true = dict(lq.true)
        kf, _ = kappa_flip(q, oracle(true))
        km = kappa_margin(q, oracle(true))
        if not math.isfinite(kf) or not math.isfinite(km) or kf <= 0:
            skip["degenerate"] += 1
            continue
        prepared.append(dict(qid=qid, q=q, subsets=subsets, true=true,
                             est=lq.est, kf=kf, km=km, n=len(q.tables)))
    print(f"  dropped: {skip}  (of {len(qs)} queries)")
    return prepared


def tier1(prepared) -> List[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for P in prepared:
        true = P["true"]
        subsets = P["subsets"]
        tvals = np.array([true[s] for s in subsets])
        for sigma in SIGMAS:
            for d in range(DRAWS):
                noise = rng.normal(0.0, sigma, size=len(subsets))
                evals = np.maximum(tvals * np.exp(noise), 1.0)
                est = {s: float(v) for s, v in zip(subsets, evals)}
                r = regret(P["q"], oracle(true), oracle(est))
                maxq = max(qerror(est[s], true[s]) for s in subsets)
                rows.append(dict(tier="t1", qid=P["qid"], sigma=sigma, draw=d,
                                 maxq=maxq, regret=r, kf=P["kf"], km=P["km"], n=P["n"]))
    return rows


def tier2(prepared) -> List[dict]:
    rows = []
    for P in prepared:
        true = P["true"]
        subsets = P["subsets"]
        for method, edict in P["est"].items():
            est = {s: max(float(edict[s]), 1.0) for s in subsets}
            r = regret(P["q"], oracle(true), oracle(est))
            maxq = max(qerror(est[s], true[s]) for s in subsets)
            rows.append(dict(tier="t2", qid=P["qid"], method=method,
                             maxq=maxq, regret=r, kf=P["kf"], km=P["km"], n=P["n"]))
    return rows


def _spear(x, y):
    if len(x) < 3:
        return float("nan")
    return spearmanr(x, y).statistic


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residual method)."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    def resid(a, b):  # residual of a after linear fit on b
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def stratified_spearman(logq, logr, kappa, n_strata=4):
    """Within-kappa-quartile Spearman(logq,logr), sample-size weighted."""
    qcut = np.quantile(kappa, np.linspace(0, 1, n_strata + 1))
    qcut[0], qcut[-1] = -np.inf, np.inf
    num = den = 0.0
    parts = []
    for i in range(n_strata):
        m = (kappa >= qcut[i]) & (kappa < qcut[i + 1])
        if m.sum() < 3:
            continue
        r = _spear(logq[m], logr[m])
        if math.isnan(r):
            continue
        parts.append((int(m.sum()), r))
        num += m.sum() * r
        den += m.sum()
    return (num / den if den else float("nan")), parts


def analyse(rows, label):
    logq = np.log(np.array([r["maxq"] for r in rows]))
    logr = np.log(np.array([max(r["regret"], 1.0) for r in rows]))
    out = {"label": label, "n_points": len(rows),
           "frac_regret_gt1": float(np.mean(logr > 1e-9)),
           "r_pool": _spear(logq, logr)}
    for kname in ("kf", "km"):
        kappa = np.array([r[kname] for r in rows])
        logk = np.log(kappa)
        r_strat, parts = stratified_spearman(logq, logr, kappa)
        r_part = partial_spearman(logq, logr, logk)
        out[f"{kname}_r_strat"] = r_strat
        out[f"{kname}_r_partial"] = r_part
        out[f"{kname}_strata"] = parts
    return out


def verdict(a):
    """Apply the frozen decision rule; return per-proxy/op verdict strings."""
    rp = a["r_pool"]
    res = {}
    for kname in ("kf", "km"):
        for op in ("strat", "partial"):
            rc = a[f"{kname}_r_{op}"]
            lift = rc - rp
            if math.isnan(rc):
                v = "n/a"
            elif abs(rp) < 0.5 and lift >= 0.2 and rc >= 0.7:
                v = "SUPPORT"
            elif lift < 0.1:
                v = "REJECT"
            else:
                v = "weak"
            res[f"{kname}_{op}"] = (round(rc, 3), round(lift, 3), v)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = prep_queries()
    print(f"prepared complete queries: {len(prepared)}")
    print(f"  kappa_flip range: {min(P['kf'] for P in prepared):.3f} .. "
          f"{max(P['kf'] for P in prepared):.3f}")

    r1, r2 = tier1(prepared), tier2(prepared)
    for rows, name in [(r1, "tier1_points"), (r2, "tier2_points")]:
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    summary = {}
    for rows, label in [(r1, "TIER-1 synthetic (controlled)"),
                        (r2, "TIER-2 real estimators")]:
        a = analyse(rows, label)
        a["verdict"] = verdict(a)
        summary[label] = a
        print(f"\n=== {label} ===")
        print(f"  points={a['n_points']}  frac(regret>1)={a['frac_regret_gt1']:.2f}  "
              f"r_pool={a['r_pool']:.3f}")
        for kname, pretty in [("kf", "kappa_flip (primary)"), ("km", "kappa_margin")]:
            print(f"  {pretty}: r_strat={a[f'{kname}_r_strat']:.3f}  "
                  f"r_partial={a[f'{kname}_r_partial']:.3f}")
            for op in ("strat", "partial"):
                rc, lift, v = a["verdict"][f"{kname}_{op}"]
                print(f"      [{op:7s}] r_cond={rc:+.3f} lift={lift:+.3f} -> {v}")
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {OUT}/tier1_points.csv, tier2_points.csv, summary.json")


if __name__ == "__main__":
    main()

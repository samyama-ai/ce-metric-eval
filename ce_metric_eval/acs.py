"""Average-case Sub-Optimality (ACS) — Experiment 1 (frozen HYPOTHESIS-3).

ACS_inf(q) = E[P-error] under a wide (sigma->large) lognormal error model: a GEOMETRY-ONLY
query constant (uses no real estimates, no true-point locality). Tests whether intrinsic
query geometry predicts large-error regret of REAL estimators better than realized q-error.

Reproduce: python -m ce_metric_eval.acs
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .experiment import prep_from, tier2
from .workload import load, load_joblight
from .geometry import regret, oracle

OUT = Path(__file__).resolve().parent.parent / "data" / "results"
SEED = 0xAC50


def acs_at(P, sigma, draws, rng):
    true = P["true"]; subs = P["subsets"]
    tv = np.array([true[s] for s in subs])
    tot = 0.0
    for _ in range(draws):
        ev = np.maximum(tv * np.exp(rng.normal(0.0, sigma, size=len(subs))), 1.0)
        est = {s: float(v) for s, v in zip(subs, ev)}
        tot += regret(P["q"], oracle(true), oracle(est))
    return tot / draws


def partial(x, y, z):
    rx, ry, rz = (np.argsort(np.argsort(np.asarray(v, float))).astype(float) for v in (x, y, z))
    def res(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        c, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ c
    ex, ey = res(rx, rz), res(ry, rz)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def analyse(prepared, label, draws=300):
    rng = np.random.default_rng(SEED)
    qids, acs_inf, acs4, acs6, kf, nt = [], [], [], [], [], []
    for P in prepared:
        qids.append(P["qid"])
        acs_inf.append(acs_at(P, 5.0, draws, rng))
        acs4.append(acs_at(P, 4.0, draws, rng))
        acs6.append(acs_at(P, 6.0, draws, rng))
        kf.append(P["kf"]); nt.append(P["n"])
    acs_inf = np.array(acs_inf); acs4 = np.array(acs4); acs6 = np.array(acs6)
    # real estimators: per-query mean regret + mean max-q-error, and pooled points
    rows = tier2(prepared)
    by = defaultdict(list)
    for r in rows:
        by[r["qid"]].append(r)
    reg_mean = np.array([np.mean([x["regret"] for x in by[q]]) for q in qids])
    q_mean = np.array([np.mean([x["maxq"] for x in by[q]]) for q in qids])
    pooled_reg = np.array([r["regret"] for r in rows])
    pooled_q = np.array([r["maxq"] for r in rows])
    qid_index = {q: i for i, q in enumerate(qids)}
    pooled_acs = np.array([acs_inf[qid_index[r["qid"]]] for r in rows])

    plateau = float(np.mean(np.abs(acs6 - acs4) / np.maximum(acs4, 1e-9)))
    res = {
        "label": label, "nq": len(qids),
        "rho_regret_acs": float(spearmanr(reg_mean, acs_inf).statistic),
        "rho_regret_qmean": float(spearmanr(reg_mean, q_mean).statistic),
        "partial_acs_given_n": partial(reg_mean, acs_inf, nt),
        "rho_regret_kappa": float(spearmanr(reg_mean, kf).statistic),
        "pooled_rho_acs": float(spearmanr(pooled_reg, pooled_acs).statistic),
        "pooled_rho_q": float(spearmanr(pooled_reg, pooled_q).statistic),
        "plateau_rel_change": plateau,
        "acs_inf_range": [float(acs_inf.min()), float(np.median(acs_inf)), float(acs_inf.max())],
    }
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Preparing workloads (this runs Monte-Carlo ACS per query)...")
    stats = prep_from(load())
    jl = prep_from(load_joblight())
    R = {}
    for name, prep in [("STATS-CEB (large-error)", stats), ("job-light (small-error)", jl)]:
        a = analyse(prep, name)
        R[name] = a
        print(f"\n=== {name} ===  (n={a['nq']})")
        print(f"  ACS_inf range [min,med,max] = {[round(x,3) for x in a['acs_inf_range']]}")
        print(f"  per-query: rho(regret, ACS_inf)   = {a['rho_regret_acs']:+.3f}")
        print(f"             rho(regret, q-error)   = {a['rho_regret_qmean']:+.3f}")
        print(f"             rho(regret, kappa)     = {a['rho_regret_kappa']:+.3f}  (true-point, Day-1)")
        print(f"             partial(ACS|#tables)   = {a['partial_acs_given_n']:+.3f}")
        print(f"  pooled(query,est): rho(P-err,ACS)={a['pooled_rho_acs']:+.3f}  "
              f"rho(P-err,qerr)={a['pooled_rho_q']:+.3f}")
        print(f"  plateau |ACS(6)-ACS(4)|/ACS(4) = {a['plateau_rel_change']:.3f}")

    s = R["STATS-CEB (large-error)"]; j = R["job-light (small-error)"]
    H3a = (s["rho_regret_acs"] >= 0.55) and (s["rho_regret_acs"] - s["rho_regret_qmean"] >= 0.10)
    H3b = s["pooled_rho_acs"] >= s["pooled_rho_q"]
    H3c = s["plateau_rel_change"] <= 0.15
    H3d_margin_stats = s["rho_regret_acs"] - s["rho_regret_qmean"]
    H3d_margin_jl = j["rho_regret_acs"] - j["rho_regret_qmean"]
    H3d = H3d_margin_jl < H3d_margin_stats  # ACS dominance is regime-specific
    verdict = "SIGNAL" if (H3a and H3c) else "NO SIGNAL"
    print("\n" + "=" * 60)
    print(f"H3a (ACS>=.55 & beats q by>=.10 on STATS): {H3a}  "
          f"[acs={s['rho_regret_acs']:.3f} q={s['rho_regret_qmean']:.3f} margin={H3d_margin_stats:+.3f}]")
    print(f"H3b (ACS>=q pooled per-estimator): {H3b}")
    print(f"H3c (plateau<=.15): {H3c}  [{s['plateau_rel_change']:.3f}]")
    print(f"H3d (regime-specific; jl margin {H3d_margin_jl:+.3f} < stats {H3d_margin_stats:+.3f}): {H3d}")
    print(f"\n>>> EXPERIMENT-1 VERDICT: {verdict}")
    R["verdict"] = verdict
    with open(OUT / "acs_summary.json", "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"wrote {OUT}/acs_summary.json")


if __name__ == "__main__":
    main()

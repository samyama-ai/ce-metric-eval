"""Day-2 confirmatory analysis = the FROZEN HYPOTHESIS-2 regime test.

Runs the per-query, regime-split analysis on:
  - STATS-CEB (123 complete queries) -- hypothesis-GENERATING; reported as 'consistency' only.
  - job-light (70 queries, IMDB schema) -- INDEPENDENT workload -> the H2e confirmation.
  - real estimators restricted to their in-regime (small q-error) subset -> H2f.

Thresholds are read from HYPOTHESIS-2.md and NOT edited after seeing numbers.
Reproduce: python -m ce_metric_eval.confirm
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .experiment import prep_from, tier1, tier2, SIGMAS
from .workload import load, load_joblight
from .metrics import qerror

OUT = Path(__file__).resolve().parent.parent / "data" / "results"


def partial(x, y, z):
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    def res(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        c, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ c
    ex, ey = res(rx, rz), res(ry, rz)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def regime_curve(prepared):
    """Per-query mean regret vs kappa across the sigma sweep (tier-1 synthetic)."""
    rows = tier1(prepared)
    by = {}
    for r in rows:
        by.setdefault(r["sigma"], {}).setdefault(r["qid"], []).append(r)
    res = {}
    for sig in sorted(by):
        qs = by[sig]
        mr = np.array([np.mean([x["regret"] for x in v]) for v in qs.values()])
        kf = np.array([next(iter(v))["kf"] for v in qs.values()])
        mq = np.array([np.mean([x["maxq"] for x in v]) for v in qs.values()])
        nt = np.array([next(iter(v))["n"] for v in qs.values()])
        res[sig] = {
            "nq": len(qs),
            "rho_kf": float(spearmanr(mr, kf).statistic),
            "rho_q": float(spearmanr(mr, mq).statistic),
            "partial_kf_given_n": partial(mr, kf, nt),
        }
    return res


def tier2_in_regime(prepared, qcap=3.0):
    """H2f: per estimator, restrict to queries with that estimator's per-query max q-error
    <= qcap (in-regime), then Spearman(regret, kappa_flip)."""
    rows = tier2(prepared)
    out = {}
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    for m, rs in by.items():
        sub = [r for r in rs if r["maxq"] <= qcap]
        if len(sub) >= 8:
            reg = [r["regret"] for r in sub]
            kf = [r["kf"] for r in sub]
            out[m] = {"n_in_regime": len(sub),
                      "rho": float(spearmanr(reg, kf).statistic)}
        else:
            out[m] = {"n_in_regime": len(sub), "rho": float("nan")}
    return out


def evaluate(curve):
    """Apply frozen H2a-d thresholds to a regime curve."""
    g = lambda s: curve.get(s, {})
    r025 = g(0.25).get("rho_kf", float("nan"))
    r050 = g(0.5).get("rho_kf", float("nan"))
    r150 = g(1.5).get("rho_kf", float("nan"))
    q025 = g(0.25).get("rho_q", float("nan"))
    pn025 = g(0.25).get("partial_kf_given_n", float("nan"))
    seq = [g(s).get("rho_kf", float("nan")) for s in SIGMAS]
    # non-increasing allowing one inversion <= 0.05
    inversions = sum(1 for a, b in zip(seq, seq[1:]) if b - a > 0.05)
    H2a = (r025 >= 0.50) and (r050 >= 0.40)
    H2b = (r025 - q025) >= 0.20
    H2c = (inversions == 0) and (r150 < 0.20)
    H2d = pn025 >= 0.40
    return {"H2a": H2a, "H2b": H2b, "H2c": H2c, "H2d": H2d,
            "r025": round(r025, 3), "r050": round(r050, 3), "r150": round(r150, 3),
            "q025": round(q025, 3), "partial025": round(pn025, 3),
            "inversions": inversions}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Preparing workloads...")
    stats = prep_from(load())
    jl = prep_from(load_joblight())
    print(f"  STATS-CEB complete+usable: {len(stats)}   job-light: {len(jl)}")

    report = {}
    for name, prepared in [("STATS-CEB (generating)", stats), ("job-light (INDEPENDENT)", jl)]:
        curve = regime_curve(prepared)
        ev = evaluate(curve)
        t2 = tier2_in_regime(prepared)
        report[name] = {"curve": curve, "eval": ev, "tier2_in_regime": t2}
        print(f"\n=== {name} ===  (n={len(prepared)})")
        print("  sigma:  " + "  ".join(f"{s:>5}" for s in SIGMAS))
        print("  rho_kf: " + "  ".join(f"{curve[s]['rho_kf']:+5.2f}" for s in SIGMAS))
        print("  rho_q : " + "  ".join(f"{curve[s]['rho_q']:+5.2f}" for s in SIGMAS))
        print(f"  H2a(r025>=.5 & r050>=.4)={ev['H2a']}  [r025={ev['r025']} r050={ev['r050']}]")
        print(f"  H2b(kappa-q>=.2)={ev['H2b']}  [r025-q025={round(ev['r025']-ev['q025'],3)}]")
        print(f"  H2c(decay & r150<.2)={ev['H2c']}  [r150={ev['r150']} inversions={ev['inversions']}]")
        print(f"  H2d(partial|n>=.4)={ev['H2d']}  [partial={ev['partial025']}]")
        print(f"  H2f tier-2 in-regime (q<=3): " +
              ", ".join(f"{m}:{d['rho']:+.2f}(n={d['n_in_regime']})" for m, d in t2.items()))

    # Frozen decision rule
    s_ev = report["STATS-CEB (generating)"]["eval"]
    j_ev = report["job-light (INDEPENDENT)"]["eval"]
    j_t2 = report["job-light (INDEPENDENT)"]["tier2_in_regime"]
    core = all([j_ev["H2a"], j_ev["H2b"], j_ev["H2c"], j_ev["H2d"]])  # on independent set
    h2e = j_ev["H2a"]  # independence: H2a holds on job-light
    h2f = any(d["rho"] >= 0.35 for d in j_t2.values() if d["rho"] == d["rho"])
    print("\n" + "=" * 60)
    print(f"INDEPENDENT (job-light) core H2a-d all pass: {core}")
    print(f"H2e (independence, H2a on job-light): {h2e}    H2f (real est in-regime>=.35): {h2f}")
    if core and h2e:
        verdict = "CONFIRM"
    elif j_ev["H2a"] is False:
        verdict = "REFUTE"
    else:
        verdict = "PARTIAL"
    print(f"\n>>> DAY-2 VERDICT: {verdict}")
    report["verdict"] = verdict
    with open(OUT / "confirm_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {OUT}/confirm_summary.json")


if __name__ == "__main__":
    main()

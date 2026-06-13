"""ACS_inf via the limit law (THEORY-3):  ACS_inf(q) = sum_k r_k * pi_k, where r_k are
true plan cost-ratios and pi_k are CARDINALITY-FREE minimax-rank selection probabilities
(k = argmin_k max_{S in I_k} eps_S, eps iid). Computable without simulating the error model.

Reproduce / validate:  python -m ce_metric_eval.acs_limit
"""
from __future__ import annotations
import numpy as np

from .optimizer import enumerate_plans, cost_of_tree
from .geometry import oracle, internal_subsets


def acs_inf_limit(P, draws=2000, rng=None):
    """ACS_inf by the limit law: cost-ratios r_k weighted by minimax-rank probabilities pi_k."""
    rng = rng or np.random.default_rng(0)
    true = P["true"]; allsub = list(P["subsets"]); idx = {s: i for i, s in enumerate(allsub)}
    plans = enumerate_plans(P["q"])
    Ik = [[idx[s] for s in internal_subsets(t) if s in idx] for t in plans]
    costs = np.array([cost_of_tree(t, oracle(true)) for t in plans])
    r = costs / costs.min()
    sel = np.zeros(len(plans))
    for _ in range(draws):
        eps = rng.normal(0.0, 1.0, len(allsub))
        # cardinality-free minimax-rank selection (ties: lexicographic via sorted-desc compare)
        best, bk = None, 0
        for k, I in enumerate(Ik):
            key = tuple(sorted((eps[j] for j in I), reverse=True)) if I else (np.inf,)
            if best is None or key < best:
                best, bk = key, k
        sel[bk] += 1
    pi = sel / draws
    return float((r * pi).sum())


if __name__ == "__main__":
    from scipy.stats import spearmanr
    from .experiment import prep_from
    from .workload import load
    from .acs import acs_at
    prepared = prep_from(load())
    rng = np.random.default_rng(3)
    lim, mc = [], []
    for P in prepared[:80]:
        lim.append(acs_inf_limit(P, draws=2000, rng=rng))
        mc.append(acs_at(P, 5.0, 400, rng))
    lim, mc = np.array(lim), np.array(mc)
    print(f"limit-law ACS_inf vs MC-ACS(sigma=5): Spearman={spearmanr(lim, mc).statistic:.4f}, "
          f"median |lim-mc|/mc={np.median(np.abs(lim-mc)/mc):.4f}  (n={len(lim)})")

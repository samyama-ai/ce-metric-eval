r"""Plan regret and the plan-condition-number proxies (the first-principles core).

regret      : forward error  -- how much worse the estimate-chosen plan is, under truth.
kappa_flip  : PRIMARY condition number = 1 / (input-space L_inf log-margin to the nearest
              plan-switch boundary). Derived in closed form below.
kappa_margin: SECONDARY (cheap, cost-space) = log(2nd-best / best) full-plan cost.

--- Closed form for the flip margin (the tropical-cell wall distance) ---
Plan cost is C_out = sum of cardinalities over a plan's internal nodes. Perturb each
subset S's cardinality multiplicatively by e^{u_S} with |u_S| <= delta (an L_inf ball of
radius delta in log-cardinality space; delta = log q-error). For an alternative plan P'
to beat the optimum P*, we need cost(P') <= cost(P*). Internal nodes shared by both plans
cancel, leaving

    cost(P') - cost(P*) = sum_{S in P'\P*} c_S e^{u_S}  -  sum_{S in P*\P'} c_S e^{u_S}.

The adversary minimizes this by setting u_S = -delta on P'\P* and u_S = +delta on P*\P*,
giving e^{-delta} A - e^{+delta} B with A = sum_{P'\P*} c_S, B = sum_{P*\P'} c_S. Setting
<= 0 yields  delta(P') = 1/2 * ln(A / B).  Since P* is optimal, A >= B, so delta >= 0.
The query's flip margin is delta_q = min over alternative plans P' of delta(P').
"""
from __future__ import annotations
import math
from typing import Callable, Dict, FrozenSet, List, Tuple

from .optimizer import Query, Tree, optimize, cost_of_tree, enumerate_plans, leaves


def internal_subsets(tree: Tree) -> List[FrozenSet[str]]:
    """Leaf-sets of every internal (join) node of a plan tree."""
    if isinstance(tree, str):
        return []
    out = [leaves(tree)]
    out += internal_subsets(tree[0])
    out += internal_subsets(tree[1])
    return out


def regret(q: Query,
           true_card: Callable[[FrozenSet[str]], float],
           est_card: Callable[[FrozenSet[str]], float]) -> float:
    """C_out(plan chosen under estimates ; truth) / C_out(plan chosen under truth ; truth)."""
    _, p_true = optimize(q, true_card)
    _, p_est = optimize(q, est_card)
    num = cost_of_tree(p_est, true_card)
    den = cost_of_tree(p_true, true_card)
    return num / den


def kappa_margin(q: Query, true_card: Callable[[FrozenSet[str]], float]) -> float:
    """log(second-best distinct full-plan cost / best). +inf if only one cost exists."""
    costs = sorted({round(cost_of_tree(t, true_card), 9) for t in enumerate_plans(q)})
    if len(costs) < 2:
        return float("inf")
    return math.log(costs[1] / costs[0])


def kappa_flip(q: Query, true_card: Callable[[FrozenSet[str]], float]
               ) -> Tuple[float, float]:
    """Return (kappa_flip, delta_q). delta_q = min L_inf log-perturbation that flips the
    optimal plan; kappa_flip = 1/delta_q (inf on an exact tie)."""
    _, p_star = optimize(q, true_card)
    star = set(internal_subsets(p_star))
    best_delta = float("inf")
    for t in enumerate_plans(q):
        cur = set(internal_subsets(t))
        if cur == star:
            continue  # same plan
        only_alt = cur - star          # P' \ P*
        only_star = star - cur         # P* \ P'
        A = sum(true_card(s) for s in only_alt)
        B = sum(true_card(s) for s in only_star)
        if A <= 0 or B <= 0:
            continue
        delta = 0.5 * math.log(A / B)
        if delta < best_delta:
            best_delta = delta
    if best_delta == float("inf"):
        return 0.0, float("inf")
    if best_delta <= 0.0:
        return float("inf"), 0.0
    return 1.0 / best_delta, best_delta


def oracle(d: Dict[FrozenSet[str], float]) -> Callable[[FrozenSet[str]], float]:
    """Wrap a {frozenset: cardinality} dict as a card() oracle."""
    return lambda s: d[s]

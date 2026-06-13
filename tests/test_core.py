"""Correctness layer (C1-C4). C1 is fully hand-computed; see the table below.

3-table chain  A - B - C  (edges A-B, B-C; A-C NOT joinable).
Two valid plans:
  T1 = (A,(B,C))  internal subsets {ABC},{BC}   cost = card(BC)+card(ABC)
  T2 = ((A,B),C)  internal subsets {ABC},{AB}   cost = card(AB)+card(ABC)

TRUE: AB=100, BC=1000, ABC=5000  ->  cost T1=6000, T2=5100  -> optimal = T2 (5100)
EST : AB=100, BC=50,   ABC=5000  ->  cost T1=5050, T2=5100  -> optimal = T1 (5050)
  regret      = cost_true(T1)/cost_true(T2) = 6000/5100 = 1.176470588
  q-error(BC) = max(1000/50, 50/1000) = 20
  kappa_flip  : P*=T2, alt=T1 -> A=card(BC)=1000, B=card(AB)=100
                delta = 0.5*ln(1000/100) = 0.5*ln10 = 1.151292546 ; kappa=1/delta=0.868589
  kappa_margin= ln(6000/5100) = 0.162518929
All numbers below are independently hand-derived, not copied from a run.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_metric_eval.optimizer import Query, optimize, cost_of_tree, enumerate_plans
from ce_metric_eval.geometry import regret, kappa_flip, kappa_margin, oracle

A, B, C = "A", "B", "C"
fs = frozenset
CHAIN = Query([A, B, C], [(A, B), (B, C)])
TRUE = oracle({fs([A, B]): 100, fs([B, C]): 1000, fs([A, B, C]): 5000,
               fs([A]): 1, fs([B]): 1, fs([C]): 1})
EST = oracle({fs([A, B]): 100, fs([B, C]): 50, fs([A, B, C]): 5000,
              fs([A]): 1, fs([B]): 1, fs([C]): 1})


def approx(x, y, tol=1e-6):
    return abs(x - y) <= tol * max(1.0, abs(y))


def test_c1_optimal_cost_and_plan():
    cost, plan = optimize(CHAIN, TRUE)
    assert approx(cost, 5100.0), cost
    # optimal true plan must join A,B first (internal subset {A,B} present)
    from ce_metric_eval.geometry import internal_subsets
    assert fs([A, B]) in set(internal_subsets(plan))


def test_c1_regret():
    assert approx(regret(CHAIN, TRUE, EST), 6000.0 / 5100.0)


def test_c1_kappa_flip():
    k, delta = kappa_flip(CHAIN, TRUE)
    assert approx(delta, 0.5 * math.log(10.0)), delta
    assert approx(k, 1.0 / (0.5 * math.log(10.0))), k


def test_c1_kappa_margin():
    assert approx(kappa_margin(CHAIN, TRUE), math.log(6000.0 / 5100.0))


def test_c2_dp_equals_bruteforce():
    # DP optimum must equal the cheapest plan found by brute-force enumeration.
    dp_cost, _ = optimize(CHAIN, TRUE)
    brute = min(cost_of_tree(t, TRUE) for t in enumerate_plans(CHAIN))
    assert approx(dp_cost, brute)
    # also on a 4-table cycle
    q4 = Query(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    import random
    rnd = random.Random(7)
    card = {}
    for n in range(1, 5):
        from itertools import combinations
        for combo in combinations(["A", "B", "C", "D"], n):
            card[frozenset(combo)] = rnd.uniform(1, 1e4)
    orc = oracle(card)
    # restrict to connected subsets the optimizer will actually touch is unnecessary:
    # enumerate_plans only uses connected ones; ensure all connected subsets have a value (they do).
    dp_cost, _ = optimize(q4, orc)
    brute = min(cost_of_tree(t, orc) for t in enumerate_plans(q4))
    assert approx(dp_cost, brute), (dp_cost, brute)


def test_c3_regret_ge_one_and_true_invariant():
    # R1 invariant: true estimates => regret == 1 exactly
    assert approx(regret(CHAIN, TRUE, TRUE), 1.0)
    # regret always >= 1
    assert regret(CHAIN, TRUE, EST) >= 1.0 - 1e-12


def test_c4_kappa_flip_monotonic():
    # Bigger best/2nd gap => harder to flip => smaller kappa_flip (larger delta).
    near_tie = oracle({fs([A, B]): 100, fs([B, C]): 101, fs([A, B, C]): 5000,
                       fs([A]): 1, fs([B]): 1, fs([C]): 1})
    wide_gap = oracle({fs([A, B]): 100, fs([B, C]): 100000, fs([A, B, C]): 5000,
                       fs([A]): 1, fs([B]): 1, fs([C]): 1})
    k_tie, d_tie = kappa_flip(CHAIN, near_tie)
    k_wide, d_wide = kappa_flip(CHAIN, wide_gap)
    assert d_tie < d_wide       # near-tie flips with a smaller perturbation
    assert k_tie > k_wide       # ... i.e. higher condition number


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)

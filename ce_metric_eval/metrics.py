"""q-error battery and aggregation statistics."""
from __future__ import annotations
import math
from typing import Dict, FrozenSet, List


def qerror(est: float, true: float) -> float:
    """max(est/true, true/est); guards against non-positive values (clamp to 1.0)."""
    e = max(float(est), 1.0)
    t = max(float(true), 1.0)
    return max(e / t, t / e)


def qerror_vector(est: Dict[FrozenSet[str], float],
                  true: Dict[FrozenSet[str], float],
                  subsets: List[FrozenSet[str]]) -> List[float]:
    return [qerror(est[s], true[s]) for s in subsets]


def aggregate(qs: List[float]) -> Dict[str, float]:
    """Aggregation statistics over a query's per-subquery q-errors."""
    xs = sorted(qs)
    n = len(xs)

    def pct(p):
        if n == 0:
            return float("nan")
        k = min(n - 1, int(math.ceil(p / 100.0 * n)) - 1)
        return xs[max(0, k)]

    return {
        "median": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "max": xs[-1] if n else float("nan"),
        "gmean": math.exp(sum(math.log(x) for x in xs) / n) if n else float("nan"),
    }

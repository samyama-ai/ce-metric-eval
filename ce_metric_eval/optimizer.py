"""Selinger-style DP join optimizer over connected subsets, with the C_out cost model.

C_out(plan) = sum over the plan's internal (join) nodes of the cardinality of that
intermediate result. Base-table scans are free. The optimizer is parameterized by a
*cardinality oracle* `card(frozenset_of_tables) -> float` so the same code finds the
optimal plan under TRUE cardinalities or under ESTIMATES — the difference between those
two plans (costed under truth) is plan regret (see geometry.py).

The plan space is the full bushy space (all connected binary join trees), so the result
is the genuine optimum, not a left-deep heuristic. Queries here are small (STATS-CEB
templates are <= ~8 tables), so exact DP over connected subsets is fine.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Tuple

# A plan tree is either a table name (leaf, str) or a 2-tuple (left_tree, right_tree).
Tree = object


class Query:
    """A join query: tables + an undirected join graph (which pairs are joinable)."""

    def __init__(self, tables: List[str], edges: List[Tuple[str, str]]):
        self.tables = list(tables)
        self.adj: Dict[str, set] = {t: set() for t in self.tables}
        for a, b in edges:
            self.adj[a].add(b)
            self.adj[b].add(a)

    def is_connected(self, s: FrozenSet[str]) -> bool:
        if not s:
            return False
        start = next(iter(s))
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in self.adj[x]:
                if y in s and y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen == set(s)

    def has_cross_edge(self, s1: FrozenSet[str], s2: FrozenSet[str]) -> bool:
        """True if some table in s1 is join-connected to some table in s2 (valid join)."""
        for a in s1:
            if self.adj[a] & s2:
                return True
        return False


def count_plans(q: Query, cap: int = 200_000) -> int:
    """Number of valid bushy plan trees, with early bail-out at `cap` (returns cap+1)."""
    @lru_cache(maxsize=None)
    def cnt(s: FrozenSet[str]) -> int:
        if len(s) == 1:
            return 1
        total = 0
        elems = sorted(s)
        pivot, others = elems[0], elems[1:]
        for r in range(len(others)):
            for combo in combinations(others, r):
                s1 = frozenset((pivot,) + combo)
                s2 = s - s1
                if not s2 or not (q.is_connected(s1) and q.is_connected(s2)):
                    continue
                if not q.has_cross_edge(s1, s2):
                    continue
                total += cnt(s1) * cnt(s2)
                if total > cap:
                    return cap + 1
        return total
    return cnt(frozenset(q.tables))


def connected_subsets(q: Query, min_size: int = 2) -> List[FrozenSet[str]]:
    """All connected subsets of the join graph with at least `min_size` tables — exactly
    the subsets whose cardinality the DP optimizer will request."""
    from itertools import combinations as _c
    out = []
    for n in range(min_size, len(q.tables) + 1):
        for combo in _c(q.tables, n):
            s = frozenset(combo)
            if q.is_connected(s):
                out.append(s)
    return out


def leaves(tree: Tree) -> FrozenSet[str]:
    if isinstance(tree, str):
        return frozenset([tree])
    return leaves(tree[0]) | leaves(tree[1])


def cost_of_tree(tree: Tree, card: Callable[[FrozenSet[str]], float]) -> float:
    """C_out of a fixed plan tree under an arbitrary oracle (sum of internal-node sizes)."""
    if isinstance(tree, str):
        return 0.0
    left, right = tree
    return cost_of_tree(left, card) + cost_of_tree(right, card) + card(leaves(tree))


def optimize(q: Query, card: Callable[[FrozenSet[str]], float]) -> Tuple[float, Tree]:
    """Return (min C_out, optimal plan tree) over the full bushy space, under `card`."""

    @lru_cache(maxsize=None)
    def best(s: FrozenSet[str]) -> Tuple[float, Tree]:
        if len(s) == 1:
            return 0.0, next(iter(s))
        out = card(s)
        best_cost = float("inf")
        best_tree: Tree = None
        elems = sorted(s)
        pivot = elems[0]  # fix a pivot to avoid enumerating each split twice
        others = elems[1:]
        # left subset must contain pivot; iterate over all such non-empty proper subsets
        for r in range(0, len(others)):
            for combo in combinations(others, r):
                s1 = frozenset((pivot,) + combo)
                s2 = s - s1
                if not s2:
                    continue
                if not (q.is_connected(s1) and q.is_connected(s2)):
                    continue
                if not q.has_cross_edge(s1, s2):
                    continue
                c1, t1 = best(s1)
                c2, t2 = best(s2)
                c = c1 + c2 + out
                if c < best_cost:
                    best_cost = c
                    best_tree = (t1, t2)
        return best_cost, best_tree

    full = frozenset(q.tables)
    return best(full)


def enumerate_plans(q: Query) -> List[Tree]:
    """All valid bushy plan trees (for brute-force optimality checks on small queries)."""

    @lru_cache(maxsize=None)
    def gen(s: FrozenSet[str]) -> Tuple[Tree, ...]:
        if len(s) == 1:
            return (next(iter(s)),)
        trees: List[Tree] = []
        elems = sorted(s)
        pivot = elems[0]
        others = elems[1:]
        for r in range(0, len(others)):
            for combo in combinations(others, r):
                s1 = frozenset((pivot,) + combo)
                s2 = s - s1
                if not s2:
                    continue
                if not (q.is_connected(s1) and q.is_connected(s2)):
                    continue
                if not q.has_cross_edge(s1, s2):
                    continue
                for t1 in gen(s1):
                    for t2 in gen(s2):
                        trees.append((t1, t2))
        return tuple(trees)

    return list(gen(frozenset(q.tables)))

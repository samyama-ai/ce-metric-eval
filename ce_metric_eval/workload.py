"""Load STATS-CEB: compute TRUE sub-plan cardinalities with DuckDB, parse join structure
from the sub-plan SQL, and align the 4 released estimator outputs (tier-2).

Each line of stats_CEB_sub_queries.sql is:  <COUNT(*) SQL> ;|| <top_query_id>
The estimate files are line-aligned (one float per sub-plan). True cardinalities are NOT
shipped — we compute them by executing the COUNT(*) on the simplified STATS database.
"""
from __future__ import annotations
import json
import re
import threading
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import duckdb

from .optimizer import Query, connected_subsets

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "End-to-End-CardEst-Benchmark"
CSVDIR = DATA / "datasets" / "stats_simplified"
SUBQ = DATA / "workloads" / "stats_CEB" / "sub_plan_queries" / "stats_CEB_sub_queries.sql"
ESTDIR = DATA / "workloads" / "stats_CEB" / "sub_plan_queries" / "estimates"
ESTIMATORS = ["bayescard", "deepdb", "flat", "neurocard"]
CSVMAP = {
    "users": "users.csv", "posts": "posts.csv", "postlinks": "postLinks.csv",
    "posthistory": "postHistory.csv", "comments": "comments.csv",
    "votes": "votes.csv", "badges": "badges.csv", "tags": "tags.csv",
}

_FROM_RE = re.compile(r"\bFROM\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)
# handles both "users as u" (STATS-CEB) and "title t" (job-light, implicit alias)
_ALIAS_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:as\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*",
                       re.IGNORECASE)
_JOIN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z0-9_]+)\s*=\s*"
                      r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z0-9_]+)")


CACHE = HERE.parent / "data" / "truth_cache.json"
DBFILE = HERE.parent / "data" / "stats.duckdb"


def build_db(persist: str | None = None, mem_gb: int = 8, threads: int = 8
             ) -> duckdb.DuckDBPyConnection:
    path = persist if persist else str(DBFILE)
    con = duckdb.connect(path)
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem_gb}GB'")
    con.execute("SET preserve_insertion_order=false")
    if con.execute("SELECT count(*) FROM information_schema.tables").fetchone()[0] >= 8:
        return con  # already built (persisted)
    schema = (CSVDIR / "stats.sql").read_text()
    schema = re.sub(r"SERIAL PRIMARY KEY", "INTEGER", schema, flags=re.IGNORECASE)
    con.execute(schema)
    for tbl, fname in CSVMAP.items():
        con.execute(f"COPY {tbl} FROM '{CSVDIR/fname}' (FORMAT CSV, HEADER TRUE, "
                    f"TIMESTAMPFORMAT '%Y-%m-%d %H:%M:%S')")
    return con


def _run_with_timeout(con, sql: str, timeout_s: float) -> Optional[float]:
    """Execute COUNT(*) with a wall-clock timeout; interrupt + return None on overrun."""
    box: Dict[str, object] = {}

    def work():
        try:
            box["v"] = float(con.execute(sql).fetchone()[0])
        except Exception as e:  # interrupted or error
            box["e"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        con.interrupt()
        t.join(15)
        return None
    return box.get("v")  # type: ignore[return-value]


def load_truth_cache() -> Dict[str, float]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def build_truth_cache(timeout_s: float = 90.0, save_every: int = 100) -> Dict[str, float]:
    """Compute TRUE cardinalities for every distinct sub-plan COUNT(*), with a per-query
    timeout, persisting to data/truth_cache.json so it is computed exactly once (resumable).
    Timed-out queries are stored as -1 (sentinel) so we don't retry them every run."""
    con = build_db()
    cache = load_truth_cache()
    lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
    sqls = []
    seen = set()
    for line in lines:
        sql, *_ = parse_line(line)
        if sql not in seen:
            seen.add(sql)
            sqls.append(sql)
    todo = [s for s in sqls if s not in cache]
    print(f"distinct sub-plan SQL: {len(sqls)}  cached: {len(sqls)-len(todo)}  todo: {len(todo)}")
    done = 0
    for sql in todo:
        v = _run_with_timeout(con, sql, timeout_s)
        cache[sql] = v if v is not None else -1.0
        done += 1
        if v is None:
            print(f"  TIMEOUT (>{timeout_s}s): {sql[:90]}...")
        if done % save_every == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  ...{done}/{len(todo)} computed, cache saved", flush=True)
    CACHE.write_text(json.dumps(cache))
    n_timeout = sum(1 for v in cache.values() if v is not None and v < 0)
    print(f"cache complete: {len(cache)} entries, {n_timeout} timeouts")
    return cache


def parse_line(sql_line: str) -> Tuple[str, int, FrozenSet[str], List[Tuple[str, str]]]:
    """-> (count_sql, query_id, alias_set, join_edges)."""
    sql, qid = sql_line.rsplit("||", 1)
    sql = sql.strip().rstrip(";").strip()
    qid = int(qid.strip())
    m = _FROM_RE.search(sql)
    from_clause = m.group(1)
    aliases = {}
    for part in from_clause.split(","):
        am = _ALIAS_RE.fullmatch(part) or _ALIAS_RE.search(part)
        if am:
            aliases[am.group(2)] = am.group(1).lower()
    alias_set = frozenset(aliases)
    edges = []
    for jm in _JOIN_RE.finditer(sql):
        a, b = jm.group(1), jm.group(3)
        if a in aliases and b in aliases and a != b:
            edges.append((a, b))
    return sql, qid, alias_set, edges


class LoadedQuery:
    def __init__(self, qid: int):
        self.qid = qid
        self.aliases: set = set()
        self.edges: set = set()
        self.true: Dict[FrozenSet[str], float] = {}
        self.est: Dict[str, Dict[FrozenSet[str], float]] = {m: {} for m in ESTIMATORS}

    def to_query(self) -> Query:
        return Query(sorted(self.aliases), [tuple(sorted(e)) for e in self.edges])

    def complete(self) -> Tuple[bool, int, int]:
        """True iff every connected subset (size>=2) has a true cardinality."""
        q = self.to_query()
        needed = connected_subsets(q, min_size=2)
        have = sum(1 for s in needed if s in self.true and self.true[s] > 0)
        return have == len(needed), have, len(needed)


def load() -> Dict[int, LoadedQuery]:
    """Assemble LoadedQuery objects from the precomputed truth cache + estimate files.
    Requires build_truth_cache() to have been run. Sub-plans whose true cardinality is
    missing or a timeout sentinel (-1) are left out (-> query reported incomplete)."""
    cache = load_truth_cache()
    if not cache:
        raise RuntimeError("truth cache empty: run `python -m ce_metric_eval.workload` first")
    lines = [l for l in SUBQ.read_text().splitlines() if l.strip()]
    ests = {m: [float(x) for x in (ESTDIR / f"stats_CEB_sub_queries_{m}.txt")
                .read_text().splitlines() if x.strip()] for m in ESTIMATORS}
    for m in ESTIMATORS:
        assert len(ests[m]) == len(lines), (m, len(ests[m]), len(lines))

    queries: Dict[int, LoadedQuery] = {}
    for i, line in enumerate(lines):
        sql, qid, alias_set, edges = parse_line(line)
        lq = queries.setdefault(qid, LoadedQuery(qid))
        lq.aliases |= set(alias_set)
        lq.edges |= {tuple(sorted(e)) for e in edges}
        card = cache.get(sql)
        if card is not None and card > 0:
            lq.true[alias_set] = card
        for m in ESTIMATORS:
            lq.est[m][alias_set] = ests[m][i]
    return queries


JL_SUBQ = (DATA / "workloads" / "job-light" / "sub_plan_queries"
           / "job_light_sub_query_with_star_join.sql")  # has inline true cards (||qid||true)
JL_ESTDIR = DATA / "workloads" / "job-light" / "sub_plan_queries" / "estimates"


def load_joblight() -> Dict[int, LoadedQuery]:
    """Independent workload (IMDB schema). True cardinalities are INLINE as the 3rd
    ||-field: `<SQL> ;|| <query_id> || <true_card>`. No database needed."""
    lines = [l for l in JL_SUBQ.read_text().splitlines() if l.strip()]
    ests = {m: [float(x) for x in (JL_ESTDIR / f"job_light_sub_queries_{m}.txt")
                .read_text().splitlines() if x.strip()] for m in ESTIMATORS}
    for m in ESTIMATORS:
        assert len(ests[m]) == len(lines), (m, len(ests[m]), len(lines))
    queries: Dict[int, LoadedQuery] = {}
    for i, line in enumerate(lines):
        body, qid_s, truec_s = line.rsplit("||", 2)
        qid = int(qid_s.strip())
        true_card = float(truec_s.strip())
        sql = body.strip().rstrip(";").strip()
        _, _, alias_set, edges = parse_line(sql + ";||0")  # reuse FROM/WHERE parser
        lq = queries.setdefault(qid, LoadedQuery(qid))
        lq.aliases |= set(alias_set)
        lq.edges |= {tuple(sorted(e)) for e in edges}
        if true_card > 0:
            lq.true[alias_set] = true_card
        for m in ESTIMATORS:
            lq.est[m][alias_set] = ests[m][i]
    return queries


if __name__ == "__main__":
    import sys
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    build_truth_cache(timeout_s=timeout)
    qs = load()
    comp = {qid: lq.complete() for qid, lq in qs.items()}
    n_complete = sum(1 for c in comp.values() if c[0])
    print(f"\nqueries={len(qs)} complete(full plan lattice)={n_complete}")
    sizes = sorted(len(lq.aliases) for lq in qs.values())
    print(f"table-count per query: min={sizes[0]} max={sizes[-1]} "
          f"median={sizes[len(sizes)//2]}")
    # show a few incomplete ones
    for qid, (ok, have, need) in sorted(comp.items()):
        if not ok:
            print(f"  incomplete q{qid}: {have}/{need} connected subsets covered")

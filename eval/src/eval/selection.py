from __future__ import annotations

from collections import defaultdict
from typing import Callable

from eval.claims import Claim


def _by_source(pool: list[Claim]) -> dict[str, list[Claim]]:
    out: dict[str, list[Claim]] = defaultdict(list)
    for c in pool:
        out[c.source_id].append(c)
    return out


def _entity_index(pool: list[Claim]) -> dict[str, list[Claim]]:
    out: dict[str, list[Claim]] = defaultdict(list)
    for c in pool:
        for e in c.entities:
            out[e].append(c)
    return out


def _take(predicate: Callable[[object], object], items, count: int) -> list:
    """Collect up to `count` predicate-return values (not the inputs). The
    predicate returns either the desired output (a list[Claim] or other) or
    a falsy value to skip the input. Centralizes the count-bounded loop."""
    out: list = []
    for it in items:
        result = predicate(it)
        if result:
            out.append(result)
            if len(out) >= count:
                break
    return out


def _single_fact(pool, count, _by_s, _by_e):
    return [[c] for c in pool[:count]]


def _locate(pool, count, _by_s, _by_e):
    # any claim; the case's gold span IS its location — phrasing asks "where/which episode"
    return [[c] for c in pool[:count]]


def _coverage(pool, count, by_s, _by_e):
    def _ok(claims):
        return list(claims) if len(claims) >= 2 else False
    return _take(_ok, by_s.values(), count)


def _enumeration(pool, count, by_s, _by_e):
    def _ok(claims):
        named = [c for c in claims if c.entities]
        distinct = {e for c in named for e in c.entities}
        return named if len(named) >= 3 and len(distinct) >= 3 else False
    return _take(_ok, by_s.values(), count)


def _comparison(pool, count, _by_s, by_e):
    def _ok(claims):
        srcs = {c.source_id for c in claims}
        if len(srcs) < 2:
            return False
        return [next(c for c in claims if c.source_id == min(srcs)),
                next(c for c in claims if c.source_id == max(srcs))]
    return _take(_ok, by_e.values(), count)


def _multi_hop(pool, count, _by_s, by_e):
    def _ok(claims):
        spans = {(c.source_id, c.section_seq) for c in claims}
        return [claims[0], claims[1]] if len(spans) >= 2 else False
    return _take(_ok, by_e.values(), count)


def _entity_profile(pool, count, _by_s, by_e):
    def _ok(claims):
        spans = {(c.source_id, c.section_seq) for c in claims}
        return list(claims) if len(claims) >= 2 and len(spans) >= 2 else False
    return _take(_ok, by_e.values(), count)


def _temporal(pool, count, by_s, _by_e):
    def _ok(claims):
        timed = [c for c in claims if c.has_time]
        return timed if len(timed) >= 2 else False
    return _take(_ok, by_s.values(), count)


def _causal_absent(pool, count, _by_s, _by_e):
    cause_entities = {e for c in pool if c.is_cause for e in c.entities}
    def _ok(c):
        if c.is_cause or not c.entities:
            return False
        return [c] if not any(e in cause_entities for e in c.entities) else False
    return _take(_ok, pool, count)


_SELECTORS = {
    "single_fact": _single_fact, "locate": _locate, "coverage": _coverage,
    "enumeration": _enumeration, "comparison": _comparison, "multi_hop": _multi_hop,
    "entity_profile": _entity_profile, "temporal": _temporal, "causal_absent": _causal_absent,
}


def select(category: str, pool: list[Claim], count: int) -> list[list[Claim]]:
    """Up to `count` claim-sets for `category`. Each set's spans become the case's
    gold evidence. Returns [] when the pool can't satisfy the type's structure."""
    fn = _SELECTORS.get(category)
    if fn is None:
        return []
    by_s = _by_source(pool)
    by_e = _entity_index(pool)
    return fn(pool, count, by_s, by_e)

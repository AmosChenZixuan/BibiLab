from __future__ import annotations

from collections import defaultdict

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


def _single_fact(pool, count):
    return [[c] for c in pool[:count]]


def _locate(pool, count):
    # any claim; the case's gold span IS its location — phrasing asks "where/which episode"
    return [[c] for c in pool[:count]]


def _coverage(pool, count):
    sets = []
    for _sid, claims in _by_source(pool).items():
        if len(claims) >= 2:
            sets.append(list(claims))
        if len(sets) >= count:
            break
    return sets


def _enumeration(pool, count):
    # ≥3 claims in one source naming distinct entities → the entities are the list
    sets = []
    for _sid, claims in _by_source(pool).items():
        named = [c for c in claims if c.entities]
        distinct = {e for c in named for e in c.entities}
        if len(named) >= 3 and len(distinct) >= 3:
            sets.append(named)
        if len(sets) >= count:
            break
    return sets


def _comparison(pool, count):
    idx = _entity_index(pool)
    sets = []
    for _e, claims in idx.items():
        srcs = {c.source_id for c in claims}
        if len(srcs) >= 2:
            a = next(c for c in claims if c.source_id == min(srcs))
            b = next(c for c in claims if c.source_id == max(srcs))
            sets.append([a, b])
        if len(sets) >= count:
            break
    return sets


def _multi_hop(pool, count):
    # entity bridge: claim A names E (A not about E only), claim B in a DIFFERENT
    # span is about E → answer of hop 1 (E) feeds hop 2
    idx = _entity_index(pool)
    sets = []
    for _e, claims in idx.items():
        spans = {(c.source_id, c.section_seq) for c in claims}
        if len(spans) >= 2:
            a, b = claims[0], claims[1]
            sets.append([a, b])
        if len(sets) >= count:
            break
    return sets


def _entity_profile(pool, count):
    sets = []
    for _e, claims in _entity_index(pool).items():
        spans = {(c.source_id, c.section_seq) for c in claims}
        if len(claims) >= 2 and len(spans) >= 2:
            sets.append(list(claims))
        if len(sets) >= count:
            break
    return sets


def _temporal(pool, count):
    sets = []
    for _sid, claims in _by_source(pool).items():
        timed = [c for c in claims if c.has_time]
        if len(timed) >= 2:
            sets.append(timed)
        if len(sets) >= count:
            break
    return sets


def _causal_absent(pool, count):
    # event claim about entity E with NO is_cause claim mentioning E anywhere → provable absence
    cause_entities = {e for c in pool if c.is_cause for e in c.entities}
    sets = []
    for c in pool:
        if c.is_cause or not c.entities:
            continue
        if not any(e in cause_entities for e in c.entities):
            sets.append([c])
        if len(sets) >= count:
            break
    return sets


_SELECTORS = {
    "single_fact": _single_fact, "locate": _locate, "coverage": _coverage,
    "enumeration": _enumeration, "comparison": _comparison, "multi_hop": _multi_hop,
    "entity_profile": _entity_profile, "temporal": _temporal, "causal_absent": _causal_absent,
}


def select(category: str, pool: list[Claim], count: int) -> list[list[Claim]]:
    """Up to `count` claim-sets for `category`. Each set's spans become the case's
    gold evidence. Returns [] when the pool can't satisfy the type's structure."""
    fn = _SELECTORS.get(category)
    return fn(pool, count) if fn else []

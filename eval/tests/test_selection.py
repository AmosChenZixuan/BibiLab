from eval.claims import Claim
from eval.selection import select


def _c(sid, seq, text, ents=(), cause=False, time=False):
    return Claim(source_id=sid, section_seq=seq, text=text, snippet=text,
                 entities=list(ents), is_cause=cause, has_time=time)


def test_single_fact_picks_one_claim_per_set():
    pool = [_c("s1", 0, "fact one"), _c("s1", 0, "fact two")]
    sets = select("single_fact", pool, count=2)
    assert len(sets) == 2 and all(len(s) == 1 for s in sets)


def test_coverage_groups_all_claims_of_one_source():
    pool = [_c("s1", 0, "a"), _c("s1", 1, "b"), _c("s2", 0, "c")]
    sets = select("coverage", pool, count=5)
    assert any(sorted(cl.section_seq for cl in s) == [0, 1] and {cl.source_id for cl in s} == {"s1"} for s in sets)


def test_comparison_pairs_two_different_sources_sharing_entity():
    pool = [_c("s1", 0, "E fights", ents=["E"]), _c("s2", 0, "E flees", ents=["E"])]
    sets = select("comparison", pool, count=1)
    assert len(sets) == 1
    s = sets[0]
    assert {cl.source_id for cl in s} == {"s1", "s2"}


def test_entity_profile_aggregates_one_entity_across_spans():
    pool = [_c("s1", 0, "E is brave", ents=["E"]), _c("s2", 0, "E owns a sword", ents=["E"]),
            _c("s1", 0, "F sleeps", ents=["F"])]
    sets = select("entity_profile", pool, count=5)
    assert any(all("E" in cl.entities for cl in s) and len(s) >= 2 for s in sets)


def test_temporal_needs_two_time_marked_claims_one_source():
    pool = [_c("s1", 0, "first this", time=True), _c("s1", 1, "then that", time=True),
            _c("s1", 0, "no marker")]
    sets = select("temporal", pool, count=5)
    assert any(len(s) >= 2 and all(cl.has_time for cl in s) for s in sets)


def test_causal_absent_requires_event_with_no_cause_for_its_entity():
    # event about E with NO is_cause claim mentioning E → valid abstention candidate
    pool = [_c("s1", 0, "E was banished", ents=["E"])]
    sets = select("causal_absent", pool, count=5)
    assert len(sets) == 1 and sets[0][0].entities == ["E"]


def test_causal_absent_excludes_events_whose_cause_is_present():
    pool = [_c("s1", 0, "E was banished", ents=["E"]),
            _c("s1", 0, "E was banished because of treason", ents=["E"], cause=True)]
    sets = select("causal_absent", pool, count=5)
    assert sets == []  # cause for E exists → not a valid absence question


def test_unknown_or_empty_returns_empty():
    assert select("single_fact", [], count=3) == []

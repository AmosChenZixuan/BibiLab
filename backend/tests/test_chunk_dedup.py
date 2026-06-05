"""Intra-turn chunk dedup: pure partition helper + execute_find_passages wiring."""

from types import SimpleNamespace

from bibilab.pipeline.chat_tools import _partition_unseen_chunks


def _chunk(source_id: str, start: float, end: float):
    # Mirrors the fields execute_find_passages reads off a retrieve chunk.
    return SimpleNamespace(
        source_id=source_id,
        timestamp_start=start,
        timestamp_end=end,
        content=f"{source_id}@{int(start)}",
    )


def test_partition_keeps_first_occurrence_and_records_id():
    seen: set[str] = set()
    chunks = [_chunk("s1", 10, 20), _chunk("s1", 30, 40)]
    new = _partition_unseen_chunks(chunks, seen)
    assert [c.content for c in new] == ["s1@10", "s1@30"]
    assert seen == {"s1_10_20", "s1_30_40"}


def test_partition_drops_ids_already_seen():
    seen = {"s1_10_20"}
    chunks = [_chunk("s1", 10, 20), _chunk("s2", 5, 9)]
    new = _partition_unseen_chunks(chunks, seen)
    assert [c.source_id for c in new] == ["s2"]
    assert seen == {"s1_10_20", "s2_5_9"}


def test_partition_dedups_within_a_single_call():
    seen: set[str] = set()
    chunks = [_chunk("s1", 10, 20), _chunk("s1", 10, 20)]
    new = _partition_unseen_chunks(chunks, seen)
    assert len(new) == 1


def test_partition_truncates_subsecond_timestamps_to_int():
    # chunk_id uses int(ts) so 20.4 and 20.9 collide with 20 — matches the
    # existing chunk_id format at chat_tools.py:419/441.
    seen: set[str] = set()
    _partition_unseen_chunks([_chunk("s1", 10.4, 20.9)], seen)
    assert seen == {"s1_10_20"}

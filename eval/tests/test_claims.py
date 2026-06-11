import asyncio
from eval.claims import Claim, load_spans


def test_claim_defaults():
    c = Claim(source_id="s1", section_seq=0, text="X happened", snippet="X happened")
    assert c.entities == [] and c.is_cause is False and c.has_time is False


def test_load_spans_builds_section_text(monkeypatch):
    # one source, two sections; each section's text is the joined segment texts
    async def fake_sections(sid):
        return [
            {"seq": 0, "seg_start": 0, "seg_end": 1},
            {"seq": 1, "seg_start": 2, "seg_end": 2},
        ]
    async def fake_segs(ranges):
        sid, a, b = ranges[0]
        pool = {0: "hello", 1: "world", 2: "again"}
        return [{"seq": i, "text": pool[i]} for i in range(a, b + 1)]
    monkeypatch.setattr("eval.claims.get_sections", fake_sections)
    monkeypatch.setattr("eval.claims.get_segments_for_ranges", fake_segs)
    spans = asyncio.run(load_spans("s1"))
    assert [(s["section_seq"], s["text"]) for s in spans] == [(0, "hello world"), (1, "again")]

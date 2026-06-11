import asyncio
from eval.claims import Claim, build_claim_pool, extract_claims_for_span, load_spans


def test_claim_defaults():
    c = Claim(source_id="s1", section_seq=0, text="X happened", snippet="X happened")
    assert c.entities == [] and c.is_cause is False and c.has_time is False


def test_load_spans_builds_section_text(monkeypatch):
    # one source, two sections; load_spans issues ONE batched ranges query
    # then joins per-section segment text by seq.
    async def fake_sections(sid):
        return [
            {"seq": 0, "seg_start": 0, "seg_end": 1},
            {"seq": 1, "seg_start": 2, "seg_end": 2},
        ]
    async def fake_segs(ranges):
        # one call, returns all rows for both sections
        pool = {0: "hello", 1: "world", 2: "again"}
        return [{"seq": s, "text": pool[s]} for s in (0, 1, 2)]
    monkeypatch.setattr("eval.claims.get_sections", fake_sections)
    monkeypatch.setattr("eval.claims.get_segments_for_ranges", fake_segs)
    spans = asyncio.run(load_spans("s1"))
    assert [(s["section_seq"], s["text"]) for s in spans] == [(0, "hello world"), (1, "again")]


def test_load_spans_empty_source_returns_empty(monkeypatch):
    async def fake_sections(sid):
        return []
    monkeypatch.setattr("eval.claims.get_sections", fake_sections)
    assert asyncio.run(load_spans("s1")) == []


def test_extract_claims_tags_and_provenance(monkeypatch):
    raw = (
        '{"claims": ['
        '{"text": "A defeated B", "entities": ["A","B"], "is_cause": false, "has_time": true},'
        '{"text": "B did it out of revenge", "entities": ["B"], "is_cause": true, "has_time": false}'
        ']}'
    )
    monkeypatch.setattr("eval._utils._call_llm", lambda p, *a, **k: raw)
    span = {"source_id": "s1", "section_seq": 0, "text": "A defeated B. B did it out of revenge."}
    claims, err = extract_claims_for_span(span, ai_cfg=None, language="zh")
    assert err is None
    assert len(claims) == 2
    assert claims[0].source_id == "s1" and claims[0].section_seq == 0   # provenance injected, not from LLM
    assert claims[0].has_time is True and claims[1].is_cause is True
    assert claims[0].snippet  # snippet now tracks the claim text, not the section opening


def test_extract_claims_call_failure_is_error_not_crash(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("slow")
    monkeypatch.setattr("eval._utils._call_llm", boom)
    span = {"source_id": "s1", "section_seq": 0, "text": "x"}
    claims, err = extract_claims_for_span(span, ai_cfg=None)
    assert claims == [] and "TimeoutError" in err


def test_build_claim_pool_partial_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("eval.claims._cache_dir", lambda: tmp_path)
    spans = [
        {"source_id": "s1", "section_seq": 0, "text": "good"},
        {"source_id": "s2", "section_seq": 0, "text": "boom"},
    ]
    def fake(prompt, *a, **k):
        if "boom" in prompt:
            raise RuntimeError("api error")
        return '{"claims":[{"text":"f","entities":[],"is_cause":false,"has_time":false}]}'
    monkeypatch.setattr("eval._utils._call_llm", fake)
    pool, errors = build_claim_pool(spans, ai_cfg=None)
    assert len(pool) == 1 and pool[0].source_id == "s1"
    assert len(errors) == 1 and "s2" in errors[0]


def test_build_claim_pool_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("eval.claims._cache_dir", lambda: tmp_path)
    spans = [{"source_id": "s1", "section_seq": 0, "text": "same"}]
    calls = []
    def fake(prompt, *a, **k):
        calls.append(1)
        return '{"claims":[{"text":"f","entities":[],"is_cause":false,"has_time":false}]}'
    monkeypatch.setattr("eval._utils._call_llm", fake)
    build_claim_pool(spans, ai_cfg=None)
    build_claim_pool(spans, ai_cfg=None)   # second run hits cache
    assert len(calls) == 1

from eval.generate import (
    CATEGORY_PROMPTS,
    MAX_SOURCES,
    MAX_WORDS_PER_SOURCE,
    _extract_facts,
    _extract_one_source,
    _load_per_source,
    _read_transcript,
)


def test_category_prompts_exist():
    from eval.models import ALL_CATEGORIES

    for cat in ALL_CATEGORIES:
        assert cat in CATEGORY_PROMPTS
    # the prompt set covers exactly the taxonomy, no orphans
    assert set(CATEGORY_PROMPTS) == set(ALL_CATEGORIES)


def test_max_constants():
    assert MAX_SOURCES > 0
    assert MAX_WORDS_PER_SOURCE > 0


def test_resolve_counts_floor_and_weights():
    from eval.generate import DEFAULT_WEIGHTS, resolve_counts

    cats = ["single_fact", "enumeration", "multi_hop"]
    counts = resolve_counts(cats, floor=3)
    # every selected category gets at least the floor
    assert all(v >= 3 for v in counts.values())
    # unweighted category sits exactly at the floor
    assert counts["single_fact"] == 3
    # failure-prone shapes get a surplus on top of the floor
    assert counts["enumeration"] == 3 + DEFAULT_WEIGHTS["enumeration"]
    assert counts["multi_hop"] == 3 + DEFAULT_WEIGHTS["multi_hop"]


def test_read_transcript(monkeypatch):
    async def fake(sid, include_time=False):
        return "hello world"

    monkeypatch.setattr("bibilab.pipeline.transcribe.load_transcript_text", fake)
    assert _read_transcript("any-id") == "hello world"


def test_read_transcript_missing(monkeypatch):
    async def fake(sid, include_time=False):
        return ""

    monkeypatch.setattr("bibilab.pipeline.transcribe.load_transcript_text", fake)
    assert _read_transcript("missing-id") == ""


def test_load_per_source_truncation(monkeypatch):
    monkeypatch.setattr("eval.generate.MAX_WORDS_PER_SOURCE", 500)
    transcripts = {"s1": "word " * 600, "s2": "hello world"}
    monkeypatch.setattr(
        "eval.generate._read_transcript",
        lambda sid: transcripts.get(sid, ""),
    )
    sources = [
        {"id": "s1", "title": "Long"},
        {"id": "s2", "title": "Short"},
    ]
    loaded = _load_per_source(sources)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "s1"
    assert "[...truncated...]" in loaded[0]["transcript"]
    assert loaded[1]["transcript"] == "hello world"


def test_load_per_source_skips_missing(monkeypatch):
    transcripts = {"s1": "only one"}
    monkeypatch.setattr(
        "eval.generate._read_transcript",
        lambda sid: transcripts.get(sid, ""),
    )
    sources = [
        {"id": "s1", "title": "T"},
        {"id": "s2", "title": "Gone"},
    ]
    loaded = _load_per_source(sources)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "s1"


def test_extract_one_source_succeeds(monkeypatch):
    monkeypatch.setattr(
        "eval.generate._call_llm",
        lambda p, *a, **k: '{"facts":["t"]}',
    )
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert err is None
    assert fact["id"] == "s1"  # id supplied by caller, not LLM
    assert fact["facts"] == ["t"]


def test_extract_one_source_retries_on_malformed(monkeypatch):
    calls = []

    def fake_call(prompt, *a, **k):
        calls.append(prompt)
        return "[[[broken" if len(calls) == 1 else '{"facts":["t"]}'

    monkeypatch.setattr("eval.generate._call_llm", fake_call)
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert err is None
    assert fact["facts"] == ["t"]
    assert len(calls) == 2


def test_extract_one_source_persists_on_final_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("eval.generate._call_llm", lambda p, *a, **k: "[[[bad")
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert fact is None
    assert "s1" in err
    failed_dir = tmp_path / "evals" / "_failed"
    artifacts = list(failed_dir.glob("facts_s1_*.txt"))
    assert len(artifacts) == 1
    assert artifacts[0].name in err


def test_extract_one_source_rejects_non_object(monkeypatch, tmp_path):
    # Malformed shape: returns a JSON array instead of the expected object.
    monkeypatch.setattr("eval.generate._call_llm", lambda p, *a, **k: '[1, 2, 3]')
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert fact is None
    assert "Expected JSON object" in err or "object" in err


def test_extract_facts_all_succeed(monkeypatch):
    monkeypatch.setattr(
        "eval.generate._call_llm",
        lambda p, *a, **k: '{"facts":["t"]}',
    )
    sources = [
        {"id": "s1", "transcript": "hello"},
        {"id": "s2", "transcript": "world"},
    ]
    facts, errors = _extract_facts(sources, ai_cfg=None)
    assert errors == []
    assert len(facts) == 2
    assert {f["id"] for f in facts} == {"s1", "s2"}


def test_extract_facts_partial_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)

    def fake_call(prompt, *a, **k):
        # source s2's transcript is the only one containing "world"
        if "world" in prompt:
            return "[[[broken"
        return '{"facts":["t"]}'

    monkeypatch.setattr("eval.generate._call_llm", fake_call)
    sources = [
        {"id": "s1", "transcript": "hello"},
        {"id": "s2", "transcript": "world"},
    ]
    facts, errors = _extract_facts(sources, ai_cfg=None)
    assert len(facts) == 1
    assert facts[0]["id"] == "s1"
    assert len(errors) == 1
    assert "s2" in errors[0]


def test_extract_facts_reports_progress(monkeypatch):
    monkeypatch.setattr("eval.generate._call_llm", lambda p, *a, **k: '{"facts":[]}')
    sources = [{"id": f"s{i}", "transcript": f"text {i}"} for i in range(3)]
    progress = []
    _extract_facts(sources, ai_cfg=None, on_progress=lambda d, t, e: progress.append((d, t, e)))
    assert len(progress) == 3
    assert progress[-1] == (3, 3, 0)

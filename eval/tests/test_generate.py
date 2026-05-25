from eval.generate import (
    CATEGORY_PROMPTS,
    MAX_SOURCES,
    MAX_WORDS,
    _extract_facts,
    _extract_one_source,
    _load_per_source,
    _read_transcript,
)


def test_category_prompts_exist():
    for cat in ("narrow", "broad", "cross_ref", "ambiguous", "absence", "temporal"):
        assert cat in CATEGORY_PROMPTS


def test_max_constants():
    assert MAX_SOURCES > 0
    assert MAX_WORDS > 0


def test_read_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "t1.txt").write_text("hello world")
    assert _read_transcript("transcripts/t1.txt") == "hello world"


def test_read_transcript_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    assert _read_transcript("transcripts/nope.txt") == ""


def test_load_per_source_truncation(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    monkeypatch.setattr("eval.generate.MAX_WORDS", 1000)
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "t1.txt").write_text("word " * 600)
    (d / "t2.txt").write_text("hello world")

    sources = [
        {"id": "s1", "transcript_path": "transcripts/t1.txt", "title": "Long"},
        {"id": "s2", "transcript_path": "transcripts/t2.txt", "title": "Short"},
    ]
    loaded = _load_per_source(sources)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "s1"
    assert "[...truncated...]" in loaded[0]["transcript"]
    assert loaded[1]["transcript"] == "hello world"


def test_load_per_source_skips_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "t1.txt").write_text("only one")

    sources = [
        {"id": "s1", "transcript_path": "transcripts/t1.txt", "title": "T"},
        {"id": "s2", "transcript_path": "transcripts/missing.txt", "title": "Gone"},
    ]
    loaded = _load_per_source(sources)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "s1"


def test_extract_one_source_succeeds(monkeypatch):
    monkeypatch.setattr(
        "eval.generate._call_llm",
        lambda p, *a, **k: '{"topics":["t"],"claims":[],"entities":[],"contrasts":[],"temporal":[]}',
    )
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert err is None
    assert fact["id"] == "s1"  # id supplied by caller, not LLM
    assert fact["topics"] == ["t"]


def test_extract_one_source_retries_on_malformed(monkeypatch):
    calls = []

    def fake_call(prompt, *a, **k):
        calls.append(prompt)
        return "[[[broken" if len(calls) == 1 else '{"topics":["t"]}'

    monkeypatch.setattr("eval.generate._call_llm", fake_call)
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert err is None
    assert fact["topics"] == ["t"]
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
    # The #355 shape: returns an array instead of an object.
    monkeypatch.setattr("eval.generate._call_llm", lambda p, *a, **k: '[1, 2, 3]')
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    fact, err = _extract_one_source({"id": "s1", "transcript": "hello"}, ai_cfg=None)
    assert fact is None
    assert "Expected JSON object" in err or "object" in err


def test_extract_facts_all_succeed(monkeypatch):
    monkeypatch.setattr(
        "eval.generate._call_llm",
        lambda p, *a, **k: '{"topics":["t"]}',
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
        return '{"topics":["t"]}'

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
    monkeypatch.setattr("eval.generate._call_llm", lambda p, *a, **k: '{"topics":[]}')
    sources = [{"id": f"s{i}", "transcript": f"text {i}"} for i in range(3)]
    progress = []
    _extract_facts(sources, ai_cfg=None, on_progress=lambda d, t, e: progress.append((d, t, e)))
    assert len(progress) == 3
    assert progress[-1] == (3, 3, 0)

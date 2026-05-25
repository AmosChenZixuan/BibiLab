from eval.generate import (
    CATEGORY_PROMPTS,
    MAX_SOURCES,
    MAX_WORDS,
    _read_transcript,
    _load_sources,
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


def test_load_sources_truncation(monkeypatch, tmp_path):
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
    block = _load_sources(sources)
    assert "[s1]" in block
    assert "[s2]" in block
    assert "hello world" in block
    assert "[...truncated...]" in block


def test_load_sources_single(monkeypatch, tmp_path):
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "t1.txt").write_text("only one transcript")

    sources = [{"id": "s1", "transcript_path": "transcripts/t1.txt", "title": "T"}]
    block = _load_sources(sources)
    assert "only one transcript" in block
    assert "=== [s1]" in block

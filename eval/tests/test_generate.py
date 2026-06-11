from eval.generate import (
    DEFAULT_WEIGHTS,
    MAX_SOURCES,
    phrase_question,
    resolve_counts,
)


def test_resolve_counts_floor_and_weights():
    cats = ["single_fact", "enumeration", "multi_hop", "coverage", "causal_absent"]
    counts = resolve_counts(cats, floor=3)
    # every selected category gets at least the floor
    assert all(v >= 3 for v in counts.values())
    # unweighted category sits exactly at the floor
    assert counts["single_fact"] == 3
    # all weighted categories in DEFAULT_WEIGHTS get their surplus on top of the floor
    for cat in DEFAULT_WEIGHTS:
        assert counts[cat] == 3 + DEFAULT_WEIGHTS[cat]


def test_max_sources_positive():
    assert MAX_SOURCES > 0


def test_phrase_question_parses_qa(monkeypatch):
    from eval.claims import Claim
    monkeypatch.setattr("eval._utils._call_llm",
        lambda p, *a, **k: '{"question":"谁打败了B？","expected_answer_draft":"A。"}')
    claims = [Claim(source_id="s1", section_seq=0, text="A defeated B", snippet="A defeated B")]
    q, a, err = phrase_question("single_fact", claims, ai_cfg=None, language="zh")
    assert err is None and q == "谁打败了B？" and a == "A。"


def test_phrase_question_call_failure(monkeypatch):
    from eval.claims import Claim
    def boom(*a, **k):
        raise TimeoutError("slow")
    monkeypatch.setattr("eval._utils._call_llm", boom)
    claims = [Claim(source_id="s1", section_seq=0, text="x", snippet="x")]
    q, a, err = phrase_question("single_fact", claims, ai_cfg=None)
    assert q == "" and "TimeoutError" in err


def test_generate_eval_set_produces_evidence_anchored_cases(monkeypatch):
    from pathlib import Path
    import tempfile
    from eval.generate import generate_eval_set

    # two sources, one span each; claim pool then drives selection + phrasing
    async def fake_spans(sid):
        return [{"source_id": sid, "section_seq": 0, "text": f"text of {sid}"}]
    monkeypatch.setattr("eval.generate.load_spans", fake_spans)

    def fake_call(prompt, *a, **k):
        if "内容概括" in prompt:  # extraction (summary-into-N-points)
            return '{"claims":[{"text":"E acts","entities":["E"],"is_cause":false,"has_time":false}]}'
        return '{"question":"q?","expected_answer_draft":"a"}'  # phrasing

    monkeypatch.setattr("eval._utils._call_llm", fake_call)
    monkeypatch.setattr("eval.claims._cache_dir", lambda: Path(tempfile.mkdtemp()))

    sources = [{"id": "s1", "title": "A"}, {"id": "s2", "title": "B"}]
    es = generate_eval_set("list-1", sources, {"single_fact": 2, "comparison": 1}, ai_cfg=None)
    assert es.cases, "should produce cases"
    sf = [c for c in es.cases if c.category == "single_fact"]
    assert sf and sf[0].evidence and sf[0].evidence[0].source_id in {"s1", "s2"}
    # comparison case (if produced) spans two sources
    comp = [c for c in es.cases if c.category == "comparison"]
    if comp:
        assert len({e.source_id for e in comp[0].evidence}) == 2

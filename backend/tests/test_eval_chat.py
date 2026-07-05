"""Integration tests for POST /eval/run_chat.

Grader-shaped contract: full retrieved-section set per tool call (not
narrowed to only what the LLM cited), full evidence text, cited flags, no
persistence.
"""

from unittest.mock import AsyncMock, patch

import pytest

from bibilab.pipeline._shared import LLMEmptyResponseError, StreamEvent, ToolCall
from bibilab.pipeline.chat_tools import FIND_PASSAGES_TOOL
from tests import create_list

pytestmark = pytest.mark.integration


async def _fake_no_tool_stream(messages, cfg, tools=None, system=None):
    yield StreamEvent(type="delta", content="Hi there!")
    yield StreamEvent(type="done")


@pytest.mark.asyncio
async def test_eval_endpoint_trivial_query_returns_single_json_response(client, mock_stream_llm):
    """A trivial query needing no retrieval returns 200 with a single JSON
    body (not SSE), empty tool_calls, and a non-empty answer."""
    list_id = await create_list(client, "L")
    mock_stream_llm.side_effect = _fake_no_tool_stream

    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})

    assert resp.status_code == 200
    assert "text/event-stream" not in resp.headers.get("content-type", "")
    body = resp.json()
    assert body["answer"] == "Hi there!"
    assert body["tool_calls"] == []
    assert body["iterations_used"] == 1
    assert body["synthesis_forced"] is False
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_eval_endpoint_no_db_writes(client, mock_stream_llm):
    """Zero rows added to messages/conversations by a call — stateless."""
    from bibilab.db import get_db

    list_id = await create_list(client, "L")
    mock_stream_llm.side_effect = _fake_no_tool_stream

    async def count_rows(table: str) -> int:
        async with get_db() as db:
            cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            return row[0]

    before = {t: await count_rows(t) for t in ("messages", "conversations")}
    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})
    assert resp.status_code == 200
    after = {t: await count_rows(t) for t in ("messages", "conversations")}
    assert before == after


@pytest.mark.asyncio
async def test_eval_endpoint_llm_override_partial_merge(client, mock_stream_llm):
    """An llm override with only `model` set leaves api_key/base_url/
    protocol at the backend's configured (default, in this test env) values."""
    list_id = await create_list(client, "L")
    captured: dict = {}

    async def fake_stream(messages, cfg, tools=None, system=None):
        captured["cfg"] = cfg
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream

    resp = await client.post(
        "/eval/run_chat",
        json={"query": "hi", "list_id": list_id, "llm": {"model": "gpt-4o-mini"}},
    )

    assert resp.status_code == 200
    cfg = captured["cfg"]
    assert cfg.model == "gpt-4o-mini"
    assert cfg.protocol == "openai"
    assert cfg.api_key == ""
    assert cfg.base_url == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_eval_endpoint_language_null_defaults_en(client, mock_stream_llm):
    """language=null resolves to "en" regardless of backend
    cfg.ai.output_language, driven directly against the router function so
    the backend config can be pinned to a non-default output_language."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.models.eval import EvalChatRequest
    from bibilab.routers.eval import run_chat_eval

    list_id = await create_list(client, "L")
    captured_system: list[str] = []

    async def fake_stream(messages, cfg, tools=None, system=None):
        captured_system.append(system or "")
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url="", output_language="zh"),
        backend=BackendConfig(),
    )
    request = EvalChatRequest(query="hi", list_id=list_id, language=None)
    await run_chat_eval(request, cfg)

    assert captured_system[0].rstrip().endswith("Respond in English.")


@pytest.mark.asyncio
async def test_eval_endpoint_missing_models_412(client):
    """Missing required models -> 412 with {error, missing}."""
    list_id = await create_list(client, "L")
    with patch("bibilab.routers._model_gate.missing_required_models", return_value=["reranker"]):
        resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})

    assert resp.status_code == 412
    detail = resp.json()["detail"]
    assert detail["missing"] == ["reranker"]


@pytest.mark.asyncio
async def test_eval_endpoint_unknown_list_404(client):
    """Unknown list_id -> 404."""
    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": "does-not-exist"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_eval_endpoint_pipeline_error_classified_500(client, mock_stream_llm):
    """A pipeline exception surfaces as 500 with a classified error code."""
    list_id = await create_list(client, "L")
    mock_stream_llm.side_effect = LLMEmptyResponseError("boom")

    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})

    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "llm_empty_response"


@pytest.mark.asyncio
async def test_eval_endpoint_retrieval_keeps_full_section_set_with_full_text_and_cited(
    client, mock_stream_llm, monkeypatch
):
    """Core grader contract in one scenario: find_passages surfaces two sections,
    only one gets cited. tool_calls[].sections[] must include BOTH (cited
    flag distinguishing them, not narrowed to only the cited one), the cited
    section's full_text must join both of its chunks (not just the first,
    which is all `preview` would carry), and `answer` must carry the raw
    [1] marker matching the cited section's index."""
    from bibilab.db import get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.embed import RetrievalResult, SourceHit
    from bibilab.pipeline.section import Section
    from tests.factories import SourceFactory, make_retrieved_chunk, make_seg_rows, make_whisper_segments

    list_id = await create_list(client, "L")

    segs = make_whisper_segments()
    sections = [
        Section(seg_start=0, seg_end=14, token_count=100, timestamp_start=0.0, timestamp_end=15.0),
        Section(seg_start=15, seg_end=29, token_count=100, timestamp_start=15.0, timestamp_end=30.0),
    ]
    digests = [
        SectionDigest(summary="sec1 summary", keywords=["k1"]),
        SectionDigest(summary="sec2 summary", keywords=["k2"]),
    ]
    source_id = await SourceFactory.build(
        list_id,
        video_id="BVevalfull",
        title="Eval Full Text Video",
        segments=segs,
        sections=sections,
        section_digests=digests,
    )
    section_rows = await get_sections(source_id)

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        return RetrievalResult(
            chunks=[
                # Two chunks in section 1 (index 1) — full_text must join both.
                make_retrieved_chunk(source_id, 0, 1, score=0.9, title="Eval Full Text Video"),
                make_retrieved_chunk(source_id, 10, 12, score=0.8, title="Eval Full Text Video"),
                # One chunk in section 2 (index 2) — never cited by the LLM.
                make_retrieved_chunk(source_id, 16, 17, score=0.7, title="Eval Full Text Video"),
            ],
            candidates_evaluated=3,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id=source_id, video_title="Eval Full Text Video", best_score=-0.9)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=make_seg_rows(source_id)))
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))

    call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "the topic"}),
            )
        else:
            yield StreamEvent(type="delta", content="Section one covers it [1].")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream

    resp = await client.post("/eval/run_chat", json={"query": "the topic", "list_id": list_id})

    assert resp.status_code == 200
    body = resp.json()
    assert "[1]" in body["answer"]

    assert len(body["tool_calls"]) == 1
    call = body["tool_calls"][0]
    assert call["tool_name"] == "find_passages"
    sections_out = {s["index"]: s for s in call["sections"]}
    assert set(sections_out) == {1, 2}, "both sections must be present, not narrowed to only the cited one"

    cited = sections_out[1]
    uncited = sections_out[2]
    assert cited["cited"] is True
    assert uncited["cited"] is False

    assert "sentence 0" in cited["full_text"]
    assert "sentence 10" in cited["full_text"], "full_text must join every chunk, not just the first"

    # llm_context: the exact LLM-visible tool message per call, captured before
    # the SSE payload strips it — fence headers included, so a grader can bind
    # [N] answer markers to source titles. full_text alone is header-less.
    assert len(body["llm_context"]) == 1
    assert body["llm_context"][0].startswith('===== [1] "Eval Full Text Video"')
    assert "sentence 0" in body["llm_context"][0]
    assert "sentence 16" in body["llm_context"][0], "uncited section is part of what the LLM read"


@pytest.mark.asyncio
async def test_eval_endpoint_llm_override_invalid_merge_422(client):
    """An override violating AIConfig's cross-field constraint
    (max_output_tokens >= the non-overridable context_window) is caller
    error -> 422 with a message, not an unclassified bare 500."""
    list_id = await create_list(client, "L")
    resp = await client.post(
        "/eval/run_chat",
        json={"query": "hi", "list_id": list_id, "llm": {"max_output_tokens": 10_000_000}},
    )
    assert resp.status_code == 422
    detail = str(resp.json()["detail"])
    assert "max_output_tokens" in detail
    # Pydantic's default error rendering embeds the merged input dict —
    # including the backend's api_key. The detail must carry messages only.
    assert "input_value" not in detail
    assert "api_key" not in detail


@pytest.mark.asyncio
async def test_eval_endpoint_unknown_protocol_422(client):
    """protocol is a closed enum on the wire: a typo ('antropic') must fail
    validation loudly instead of silently selecting the OpenAI wire branch."""
    resp = await client.post(
        "/eval/run_chat",
        json={"query": "hi", "list_id": "x", "llm": {"protocol": "antropic"}},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_eval_endpoint_tool_failure_maps_to_tool_error(client, mock_stream_llm, monkeypatch):
    """A tool-execution failure surfaces as the same "tool_error" code
    production records — an eval harness must be able to tell a retrieval
    failure from a backend bug (internal_error)."""
    from bibilab.pipeline import chat_tools

    list_id = await create_list(client, "L")

    async def boom(*args, **kwargs):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(chat_tools, "retrieve", boom)

    async def fake_stream(messages, cfg, tools=None, system=None):
        yield StreamEvent(
            type="tool_call",
            tool_call=ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "q"}),
        )

    mock_stream_llm.side_effect = fake_stream

    resp = await client.post("/eval/run_chat", json={"query": "q", "list_id": list_id})

    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "tool_error"


@pytest.mark.asyncio
async def test_eval_endpoint_multi_call_turn_keeps_per_call_evidence(client, mock_stream_llm, monkeypatch):
    """Two find_passages calls hitting the SAME section in one turn: the
    first call's sections[] row must keep the evidence THAT call showed the
    LLM, even though the second call overwrites the shared registry's
    full_text (rows are snapshotted at tool_result time, not read from
    final registry state). Also: the preamble streamed before the first tool
    call is separated from the synthesis by a paragraph break, matching what
    production renders around a tool call."""
    from bibilab.db import get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.embed import RetrievalResult, SourceHit
    from bibilab.pipeline.section import Section
    from tests.factories import SourceFactory, make_retrieved_chunk, make_seg_rows, make_whisper_segments

    list_id = await create_list(client, "L")

    segs = make_whisper_segments()
    source_id = await SourceFactory.build(
        list_id,
        video_id="BVevalmulti",
        title="Multi Call Video",
        segments=segs,
        sections=[Section(seg_start=0, seg_end=29, token_count=100, timestamp_start=0.0, timestamp_end=30.0)],
        section_digests=[SectionDigest(summary="sec1 summary", keywords=["k1"])],
    )
    section_rows = await get_sections(source_id)

    retrieve_count = 0

    async def fake_retrieve(query_text, source_ids, cfg, top_k, **kwargs):
        nonlocal retrieve_count
        retrieve_count += 1
        # Call 1 surfaces two chunks; call 2 re-hits the same section with a
        # DIFFERENT (unseen) chunk, which rewrites the registry's full_text.
        chunks = (
            [
                make_retrieved_chunk(source_id, 0, 1, score=0.9, title="Multi Call Video"),
                make_retrieved_chunk(source_id, 10, 12, score=0.8, title="Multi Call Video"),
            ]
            if retrieve_count == 1
            else [make_retrieved_chunk(source_id, 5, 6, score=0.7, title="Multi Call Video")]
        )
        return RetrievalResult(
            chunks=chunks,
            candidates_evaluated=len(chunks),
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id=source_id, video_title="Multi Call Video", best_score=-0.9)],
        )

    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=make_seg_rows(source_id)))
    monkeypatch.setattr(chat_tools, "get_sections", AsyncMock(return_value=section_rows))

    llm_call_count = 0

    async def fake_stream(messages, cfg, tools=None, system=None):
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count == 1:
            yield StreamEvent(type="delta", content="Let me check.")
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="c1", name=FIND_PASSAGES_TOOL.name, arguments={"query": "topic a"}),
            )
        elif llm_call_count == 2:
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id="c2", name=FIND_PASSAGES_TOOL.name, arguments={"query": "topic b"}),
            )
        else:
            yield StreamEvent(type="delta", content="Covered here [1].")
            yield StreamEvent(type="done")

    mock_stream_llm.side_effect = fake_stream

    resp = await client.post("/eval/run_chat", json={"query": "the topic", "list_id": list_id})

    assert resp.status_code == 200
    body = resp.json()

    # Preamble and synthesis carry the production paragraph break, not fused text.
    assert body["answer"] == "Let me check.\n\nCovered here [1]."

    assert len(body["tool_calls"]) == 2
    first, second = body["tool_calls"]
    first_s1 = {s["index"]: s for s in first["sections"]}[1]
    second_s1 = {s["index"]: s for s in second["sections"]}[1]

    # Call 1's row keeps call 1's evidence (chunks at segs 0-1 and 10-12)...
    assert "sentence 0" in first_s1["full_text"]
    assert "sentence 10" in first_s1["full_text"]
    # ...and call 2's chunk (segs 5-6) must NOT bleed into call 1's row.
    assert "sentence 5 " not in first_s1["full_text"]
    # Call 2's row carries call 2's chunk.
    assert "sentence 5 " in second_s1["full_text"]
    # The section summary the LLM saw above the fragments is part of the evidence.
    assert "sec1 summary" in first_s1["full_text"]
    # cited comes from the citation events; both rows describe the same section.
    assert first_s1["cited"] is True
    assert second_s1["cited"] is True

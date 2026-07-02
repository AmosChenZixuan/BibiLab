"""Integration tests for POST /eval/run_chat (issue #581).

Grader-shaped contract: full retrieved-section set per tool call (not
narrowed to only what the LLM cited), full evidence text, cited flags, no
persistence. See shipit-spec-eval-run-chat.md / issue #581 design notes.
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
    """AC1 + AC8: a trivial query needing no retrieval returns 200 with a
    single JSON body (not SSE), empty tool_calls, and a non-empty answer."""
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
    assert body["detected_language"] == "en"
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_eval_endpoint_no_db_writes(client, mock_stream_llm):
    """AC2: zero rows added to messages/conversations by a call."""
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
    """AC6: an llm override with only `model` set leaves api_key/base_url/
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
    """AC7: language=null resolves to "en" regardless of backend
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
    response = await run_chat_eval(request, cfg)

    assert response.detected_language == "en"
    assert captured_system[0].rstrip().endswith("Respond in English.")


@pytest.mark.asyncio
async def test_eval_endpoint_missing_models_412(client):
    """AC9: missing required models -> 412 with {error, missing}."""
    list_id = await create_list(client, "L")
    with patch("bibilab.routers._model_gate.missing_required_models", return_value=["reranker"]):
        resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})

    assert resp.status_code == 412
    detail = resp.json()["detail"]
    assert detail["missing"] == ["reranker"]


@pytest.mark.asyncio
async def test_eval_endpoint_unknown_list_404(client):
    """AC10: unknown list_id -> 404."""
    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": "does-not-exist"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_eval_endpoint_pipeline_error_classified_500(client, mock_stream_llm):
    """AC11: a pipeline exception surfaces as 500 with a classified error code."""
    list_id = await create_list(client, "L")
    mock_stream_llm.side_effect = LLMEmptyResponseError("boom")

    resp = await client.post("/eval/run_chat", json={"query": "hi", "list_id": list_id})

    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "llm_empty_response"


@pytest.mark.asyncio
async def test_eval_endpoint_retrieval_keeps_full_section_set_with_full_text_and_cited(
    client, mock_stream_llm, monkeypatch
):
    """AC3 + AC4 + AC5, one scenario: find_passages surfaces two sections,
    only one gets cited. tool_calls[].sections[] must include BOTH (cited
    flag distinguishing them, not narrowed to only the cited one), the cited
    section's full_text must join both of its chunks (not just the first,
    which is all `preview` would carry), and `answer` must carry the raw
    [1] marker matching the cited section's index."""
    from bibilab.db import get_sections
    from bibilab.pipeline import chat_tools
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.embed import RetrievalResult, RetrievedChunk, SourceHit
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests.factories import SourceFactory

    list_id = await create_list(client, "L")

    segs = [
        WhisperSegment(start=float(i), end=float(i + 1), text=f"sentence {i} about the topic", speaker="SPK_0")
        for i in range(30)
    ]
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
                RetrievedChunk(
                    content="raw-a",
                    video_title="Eval Full Text Video",
                    timestamp_start=0.0,
                    timestamp_end=1.0,
                    source_id=source_id,
                    distance=0.1,
                    score=0.9,
                    seg_start=0,
                    seg_end=1,
                ),
                RetrievedChunk(
                    content="raw-b",
                    video_title="Eval Full Text Video",
                    timestamp_start=10.0,
                    timestamp_end=12.0,
                    source_id=source_id,
                    distance=0.2,
                    score=0.8,
                    seg_start=10,
                    seg_end=12,
                ),
                # One chunk in section 2 (index 2) — never cited by the LLM.
                RetrievedChunk(
                    content="raw-c",
                    video_title="Eval Full Text Video",
                    timestamp_start=16.0,
                    timestamp_end=17.0,
                    source_id=source_id,
                    distance=0.3,
                    score=0.7,
                    seg_start=16,
                    seg_end=17,
                ),
            ],
            candidates_evaluated=3,
            sources_with_hits=1,
            sources_total=1,
            source_coverage=[SourceHit(source_id=source_id, video_title="Eval Full Text Video", best_score=-0.9)],
        )

    seg_rows = [
        {
            "source_id": source_id,
            "seq": i,
            "start_s": float(i),
            "end_s": float(i + 1),
            "speaker": "SPK_0",
            "text": f"sentence {i} about the topic",
        }
        for i in range(30)
    ]
    monkeypatch.setattr(chat_tools, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_tools, "get_segments_for_ranges", AsyncMock(return_value=seg_rows))
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

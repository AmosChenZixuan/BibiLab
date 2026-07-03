import asyncio
import json

import httpx

from eval import api
from eval.models import EvalCase, RunCaseResult
from eval.reporter import _mean
from eval.runner import map_response, run_single_case


def test_mean():
    scores = [4.0, 5.0, 3.0, 4.0, 5.0]
    result = _mean(scores)
    assert result == 4.2


def test_mean_empty():
    assert _mean([]) == 0.0


def test_mean_single():
    assert _mean([3.7]) == 3.7


def test_run_case_result_defaults():
    r = RunCaseResult(
        case_id="c1",
        answer="test",
        citations=[],
        rag_calls=[],
        tool_blocks=[],
        llm_duration_ms=0,
        error=None,
    )
    assert r.error is None
    assert r.case_id == "c1"


_RUN_CHAT_BODY = {
    "answer": "Alice explains it in episode 2. [1]",
    "tool_calls": [
        {
            "tool_name": "find_passages",
            "query": "who explains it",
            "sections": [
                {
                    "index": 1,
                    "section_id": "sec-1",
                    "source_id": "src-a",
                    "source_title": "EP2",
                    "full_text": "===== [1] fence =====\n[S1] Alice explains it.",
                    "cited": True,
                },
                {
                    "index": 2,
                    "section_id": "sec-2",
                    "source_id": "src-b",
                    "source_title": "EP3",
                    "full_text": "===== [2] fence =====\n[S2] unrelated.",
                    "cited": False,
                },
            ],
            "candidates_evaluated": 12,
            "sources_with_hits": 2,
            "sources_total": 3,
            "reranked": True,
            "scoped_pool_size": 3,
            "facet_scope": None,
        },
        {
            "tool_name": "read_section",
            "index": 1,
            "section_id": "sec-1",
            "source_id": "src-a",
            "source_title": "EP2",
            "full_text": "[S1] Alice explains it at length.",
            "cited": True,
        },
    ],
    "iterations_used": 3,
    "synthesis_forced": False,
    "latency_ms": 1234,
    "llm_context": [
        '===== [1] "EP2" · Section 1 (0:00–0:15) =====\n[S1] Alice explains it.\n\n===== [2] "EP3" · Section 1 =====\n[S2] unrelated.',
        '[1] "EP2" · Section 1 — full text:\n[S1] Alice explains it at length.',
    ],
}


def test_map_response_full_mapping():
    r = map_response("c1", _RUN_CHAT_BODY)

    assert r.answer == "Alice explains it in episode 2. [1]"
    # llm_context passes through verbatim from the endpoint — the exact
    # LLM-bound tool messages, fence headers included (grading parity).
    assert r.llm_context == _RUN_CHAT_BODY["llm_context"]
    # citations: only cited sections, from the pipeline's cited flags.
    assert r.citations == [
        {"index": 1, "source_id": "src-a", "section_id": "sec-1"},
        {"index": 1, "source_id": "src-a", "section_id": "sec-1"},
    ]
    assert r.rag_calls == [
        {
            "query": "who explains it",
            "candidates_evaluated": 12,
            "sources_with_hits": 2,
            "sources_total": 3,
            "reranked": True,
            "scoped_pool_size": 3,
            "facet_scope": None,
        }
    ]
    assert r.tool_blocks == _RUN_CHAT_BODY["tool_calls"]
    assert r.llm_duration_ms == 1234
    assert r.error is None


def test_map_response_no_tools():
    r = map_response("c1", {"answer": "hi", "tool_calls": [], "latency_ms": 5})
    assert r.answer == "hi"
    assert r.llm_context == []
    assert r.citations == []


def test_run_single_case_maps_http_error_to_error_string(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": {"error": "llm_rate_limit_error"}})

    monkeypatch.setattr(api, "async_transport", httpx.MockTransport(handler))
    case = EvalCase(id="c1", category="single_fact", question="q?")
    result = asyncio.run(run_single_case(case, "list-1", llm=None, language="en"))

    assert result.error == "llm_rate_limit_error"
    assert result.answer == ""


def test_run_single_case_sends_llm_override_and_language(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok", "tool_calls": [], "latency_ms": 1})

    monkeypatch.setattr(api, "async_transport", httpx.MockTransport(handler))
    case = EvalCase(id="c1", category="single_fact", question="q?")
    result = asyncio.run(run_single_case(case, "list-1", llm={"model": "m-x"}, language="zh"))

    assert result.error is None
    assert seen["query"] == "q?"
    assert seen["list_id"] == "list-1"
    assert seen["llm"] == {"model": "m-x"}
    assert seen["language"] == "zh"


def test_run_single_case_null_profile_omits_llm_key(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok", "tool_calls": [], "latency_ms": 1})

    monkeypatch.setattr(api, "async_transport", httpx.MockTransport(handler))
    case = EvalCase(id="c1", category="single_fact", question="q?")
    asyncio.run(run_single_case(case, "list-1", llm=None, language="en"))

    assert "llm" not in seen

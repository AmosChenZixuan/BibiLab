"""Tests for rewriter integration in chat.py — _build_prior_user_turns
plus the synthetic-injection / fallback-telemetry / retrieve-failure paths
in run_chat_turn."""

import pytest

from bibilab.pipeline.rewriter import PriorUserTurn, RewriterIntent, build_rewriter_prompt
from bibilab.routers.chat import _build_prior_user_turns


class TestBuildPriorUserTurns:
    def test_empty_history(self):
        assert _build_prior_user_turns([]) == []

    def test_no_user_messages(self):
        history = [{"role": "assistant", "content": "hi"}]
        assert _build_prior_user_turns(history) == []

    def test_single_turn_no_assistant_response(self):
        result = _build_prior_user_turns([{"role": "user", "content": "hello"}])
        assert result == [PriorUserTurn(text="hello", retrieved=False)]

    def test_single_turn_with_retrieve(self):
        history = [
            {"role": "user", "content": "第一集讲了什么"},
            {"role": "assistant", "content": "...", "tool_blocks": [{"name": "retrieve", "result": {}}]},
        ]
        result = _build_prior_user_turns(history)
        assert result == [PriorUserTurn(text="第一集讲了什么", retrieved=True)]

    def test_single_turn_without_retrieve(self):
        history = [
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": "好的", "tool_blocks": []},
        ]
        result = _build_prior_user_turns(history)
        assert result == [PriorUserTurn(text="嗯", retrieved=False)]

    def test_mixed_turns(self):
        history = [
            {"role": "user", "content": "第一集讲了什么"},
            {"role": "assistant", "content": "...", "tool_blocks": [{"name": "retrieve", "result": {}}]},
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": "好的", "tool_blocks": []},
            {"role": "user", "content": "第二集呢"},
            {
                "role": "assistant",
                "content": "...",
                "tool_blocks": [{"name": "retrieve", "result": {}}, {"name": "query_list_metadata", "result": {}}],
            },
        ]
        result = _build_prior_user_turns(history)
        assert result == [
            PriorUserTurn(text="第一集讲了什么", retrieved=True),
            PriorUserTurn(text="嗯", retrieved=False),
            PriorUserTurn(text="第二集呢", retrieved=True),
        ]

    def test_skips_tool_role_between_user_and_assistant(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "..."},
            {"role": "assistant", "content": "hello", "tool_blocks": []},
        ]
        result = _build_prior_user_turns(history)
        assert result == [PriorUserTurn(text="hi", retrieved=False)]

    def test_last_user_turn_no_following_assistant(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer", "tool_blocks": []},
            {"role": "user", "content": "last"},
        ]
        result = _build_prior_user_turns(history)
        assert len(result) == 2
        assert result[1] == PriorUserTurn(text="last", retrieved=False)

    def test_missing_or_empty_content_skipped(self):
        # Empty content would inject literal "" into the rewriter prompt; skip instead.
        result = _build_prior_user_turns([{"role": "user"}, {"role": "user", "content": "  "}])
        assert result == []

    def test_tool_blocks_is_none(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_blocks": None},
        ]
        result = _build_prior_user_turns(history)
        assert result == [PriorUserTurn(text="hi", retrieved=False)]


class TestRewriterPromptAsymmetricContext:
    """Bug 4 regression: rewriter prompt must exclude all assistant content."""

    def test_assistant_text_not_in_prompt(self):
        # Prior assistant replied with "第六集" content. Rewriter must not see it,
        # so sequence_number inference cannot bleed from assistant turn.
        assistant_leak = "ASSISTANT_LEAK_SENTINEL_第六集"
        prior = [PriorUserTurn(text="第六集发生了什么", retrieved=True)]
        prompt = build_rewriter_prompt(current="女巫的死期是哪一天", prior=prior)
        assert "第六集发生了什么" in prompt  # prior USER message is fine
        assert "女巫的死期是哪一天" in prompt
        assert assistant_leak not in prompt  # nothing from assistant role is injected


# --- run_chat_turn integration: synthetic injection + fallback + failure paths ---


def _retrieve_result_stub(query: str = "q", mode: str = "narrow") -> dict:
    return {
        "query": query,
        "mode": mode,
        "candidates_evaluated": 1,
        "sources_with_hits": 1,
        "sources_total": 1,
        "source_coverage": [{"source_id": "s1", "video_id": "v1", "title": "V1"}],
        "_chunks": "fmt",
        "_turn_indices": [1],
        "_raw_chunks": [
            {
                "source_id": "s1",
                "chunk_id": "v1_0_10",
                "content": "verbatim",
                "video_title": "V1",
                "timestamp_start": 0.0,
                "timestamp_end": 10.0,
                "citation_index": 1,
            }
        ],
        "dropped_by_gate": 0,
        "reranked": False,
        "scoped_pool_size": 1,
        "facet_scope": None,
        "gate_margin": None,
        "neighbors_pulled": 0,
    }


async def _run_with_fakes(monkeypatch, *, rewriter_return, retrieve_side_effect=None, retrieve_return=None):
    """Drive run_chat_turn with patched LLM/retrieve. Return captured kwargs from update_message_content."""
    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline._shared import StreamEvent
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    captured_messages: list = []

    def fake_rewriter(*, current, prior, cfg, **kw):
        return rewriter_return

    async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
        captured_messages.append(messages)
        yield StreamEvent(type="delta", content="ok")
        yield StreamEvent(type="done")

    async def fake_execute_retrieve(**kwargs):
        if retrieve_side_effect is not None:
            raise retrieve_side_effect
        return retrieve_return

    captured_update: dict = {}

    async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
        captured_update["metadata"] = metadata
        captured_update["status"] = status
        captured_update["error"] = error
        captured_update["tool_blocks"] = tool_blocks

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(chat_module, "run_rewriter", fake_rewriter)
    monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
    monkeypatch.setattr(chat_module, "execute_retrieve", fake_execute_retrieve)
    monkeypatch.setattr(chat_module, "update_message_content", capture_update)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    registry = ChatRunRegistry()
    msg_id = "msg-integ"
    registry.register(msg_id, task=None)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="q",
        history=[],
        summary=None,
        source_ids=["s1"],
        source_map={"v1": "s1"},
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )
    return captured_update, captured_messages


@pytest.mark.asyncio
async def test_synthetic_injection_uses_plain_text_excerpts(monkeypatch):
    """Excerpts inject into the user message as a <retrieved_excerpts> block —
    no synthetic tool_use/tool_result pair. The tool-pair shape would teach
    non-conformant providers that retrieve is callable, causing hallucinated
    tool calls."""
    intent = RewriterIntent(retrieve=True, query="q", mode="narrow")
    telemetry = {"retrieve": True, "mode": "narrow", "attempts": 1, "latency_ms": 5}
    _, captured_messages = await _run_with_fakes(
        monkeypatch,
        rewriter_return=(intent, telemetry),
        retrieve_return=_retrieve_result_stub(),
    )
    assert len(captured_messages) == 1
    msgs = captured_messages[0]
    # The trailing message is a single user message carrying both excerpts and question.
    last = msgs[-1]
    assert last["role"] == "user"
    content = last["content"]
    assert "<retrieved_excerpts>" in content
    assert "</retrieved_excerpts>" in content
    assert _retrieve_result_stub()["_chunks"] in content
    assert "q" in content  # user_message_text preserved
    # No assistant tool_calls / tool message anywhere in the sequence.
    for m in msgs:
        assert m.get("role") != "tool"
        assert "tool_calls" not in m or not m["tool_calls"]


@pytest.mark.asyncio
async def test_rewriter_attempts_telemetry_persisted(monkeypatch):
    """metadata.rag.rewriter records attempts + mode + latency."""
    intent = RewriterIntent(retrieve=True, query="q", mode="narrow")
    telemetry = {"retrieve": True, "mode": "narrow", "attempts": 2, "latency_ms": 12}
    captured, _ = await _run_with_fakes(
        monkeypatch,
        rewriter_return=(intent, telemetry),
        retrieve_return=_retrieve_result_stub(),
    )
    rewriter_meta = captured["metadata"]["rag"]["rewriter"]
    assert rewriter_meta["attempts"] == 2
    assert rewriter_meta["mode"] == "narrow"
    assert "fallback" not in rewriter_meta  # legacy field removed


@pytest.mark.asyncio
async def test_execute_retrieve_failure_completes_with_apology(monkeypatch):
    """When pre-retrieve raises (after rewriter succeeded), the turn must still
    finalize with a 'done' status — the answer LLM gets an apology directive
    and tool_blocks remain empty (no synthetic block to persist)."""
    intent = RewriterIntent(retrieve=True, query="q", mode="narrow")
    telemetry = {"retrieve": True, "mode": "narrow", "attempts": 1, "latency_ms": 1}
    captured, _ = await _run_with_fakes(
        monkeypatch,
        rewriter_return=(intent, telemetry),
        retrieve_side_effect=RuntimeError("reranker dead"),
    )
    assert captured["status"] == "done"
    assert captured["tool_blocks"] is None
    assert "calls" not in captured["metadata"].get("rag", {})


@pytest.mark.asyncio
async def test_rewriter_error_surfaces_as_sse_error(monkeypatch):
    """Rewriter exhausting its retry budget must surface as a failed turn,
    not a degraded answer. The user can then retry."""
    from bibilab.pipeline.rewriter import RewriterError

    def boom(*, current, prior, cfg, **kw):
        raise RewriterError("exhausted retries")

    from bibilab.config import AIConfig, BackendConfig, BibilabConfig
    from bibilab.pipeline.chat_runs import ChatRunRegistry
    from bibilab.routers import chat as chat_module

    captured: dict = {}

    async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
        captured["status"] = status
        captured["error"] = error

    async def noop(*a, **kw):
        return None

    async def fake_stream_llm(*a, **kw):
        if False:  # pragma: no cover - must not be called
            yield None

    monkeypatch.setattr(chat_module, "run_rewriter", boom)
    monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
    monkeypatch.setattr(chat_module, "update_message_content", capture_update)
    monkeypatch.setattr(chat_module, "set_active_stream", noop)

    registry = ChatRunRegistry()
    msg_id = "msg-rew-err"
    registry.register(msg_id, task=None)

    cfg = BibilabConfig(
        ai=AIConfig(protocol="openai", model="x", api_key="k", base_url=""),
        backend=BackendConfig(),
    )

    await chat_module.run_chat_turn(
        message_id=msg_id,
        conversation_id="c1",
        list_id="l1",
        user_message_text="q",
        history=[],
        summary=None,
        source_ids=["s1"],
        source_map={"v1": "s1"},
        ui_lang="en",
        cfg=cfg,
        registry=registry,
    )

    assert captured["status"] == "failed"
    assert captured["error"]  # classify_error populated it

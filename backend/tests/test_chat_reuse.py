"""Tests for tool_block replay and metadata in run_chat_turn."""

import pytest


class TestReuseCallerMechanics:
    """Prior tool_blocks replay into LLM messages; rag metadata integrity."""

    @pytest.mark.anyio
    async def test_non_trivial_keeps_tool_blocks(self, monkeypatch):
        """Non-trivial follow-up → tool_blocks replay into LLM messages."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.pipeline.chat_runs import ChatRunRegistry
        from bibilab.routers import chat as chat_module

        captured: list[list[dict]] = []

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            captured.append(list(messages))
            yield StreamEvent(type="delta", content="reusing [1]")
            yield StreamEvent(type="done")

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", noop)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        registry = ChatRunRegistry()
        registry.register("m1", task=None)
        history = [
            {"role": "user", "content": "tell me about quantum mechanics"},
            {
                "role": "assistant",
                "content": "first answer [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "toolu_a",
                        "name": "retrieve",
                        "arguments": {"query": "quantum"},
                        "result": {"chunks": [], "summary": {"sources_total": 1}},
                    }
                ],
            },
        ]
        cfg = BibilabConfig(
            ai=AIConfig(protocol="anthropic", model="x", api_key="k", base_url=""),
            backend=BackendConfig(),
        )
        await chat_module.run_chat_turn(
            message_id="m1",
            conversation_id="c1",
            list_id="l1",
            user_message_text="what else did it say about entanglement",
            history=history,
            summary=None,
            source_ids=[],
            source_map={},
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )
        has_tool_block = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") in ("tool_use", "tool_result") for b in m["content"])
            for m in captured[0]
        )
        assert has_tool_block, "prior tool_blocks must replay on the non-trivial path"

    @pytest.mark.anyio
    async def test_no_reused_from_prior_call_id_in_metadata(self, monkeypatch):
        """rag.calls never carries reuse marker."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.pipeline.chat_runs import ChatRunRegistry
        from bibilab.routers import chat as chat_module

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            yield StreamEvent(type="delta", content="follow-up answer")
            yield StreamEvent(type="done")

        captured: dict = {}

        async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
            captured["meta"] = metadata

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", capture_update)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        registry = ChatRunRegistry()
        registry.register("m3", task=None)
        history = [
            {"role": "user", "content": "tell me about quantum mechanics"},
            {
                "role": "assistant",
                "content": "first answer [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "toolu_reuse_1",
                        "name": "retrieve",
                        "arguments": {"query": "quantum"},
                        "result": {"chunks": [], "summary": {"sources_total": 1}},
                    }
                ],
            },
        ]
        cfg = BibilabConfig(
            ai=AIConfig(protocol="anthropic", model="x", api_key="k", base_url=""),
            backend=BackendConfig(),
        )
        await chat_module.run_chat_turn(
            message_id="m3",
            conversation_id="c3",
            list_id="l3",
            user_message_text="tell me more about quantum mechanics",
            history=history,
            summary=None,
            source_ids=[],
            source_map={},
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )
        calls = captured.get("meta", {}).get("rag", {}).get("calls", [])
        assert all("reused_from_prior_call_id" not in c for c in calls), calls

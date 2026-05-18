"""Tests for the trivial-acknowledgment guard (#308, was #280 cosine reuse)."""

import pytest

from bibilab.pipeline.chat_tools import trivial_ack_note

_SRC_LIST_INSTRUCTION = (
    "\n\nTo search, call retrieve. Include all source numbers except those clearly unrelated to the query."
)


class TestTrivialAckNote:
    """trivial_ack_note() returns the note for trivial acks, None otherwise."""

    @pytest.mark.parametrize("msg", ["1", "12", "99"])
    def test_pure_digits_are_trivial(self, msg):
        assert trivial_ack_note(msg) is not None

    @pytest.mark.parametrize("msg", ["", " ", "  ", "a", "ab", " a ", " . ", "是吗"])
    def test_short_messages_are_trivial(self, msg):
        assert trivial_ack_note(msg) is not None

    @pytest.mark.parametrize(
        "msg",
        ["否", "ok", "OK", "Ok", "好的", "好", "是", "对", "yes", "no", "thanks", "谢谢", "k", "kk"],
    )
    def test_stopwords_are_trivial(self, msg):
        assert trivial_ack_note(msg) is not None

    def test_trivial_note_text(self):
        note = trivial_ack_note("1")
        assert note is not None
        assert "acknowledgment" in note.lower()

    @pytest.mark.parametrize("msg", ["why", "what", "tell me more", "continue"])
    def test_non_trivial_messages_return_none(self, msg):
        assert trivial_ack_note(msg) is None

    def test_no_embedding_import_on_module(self):
        """Regression: the cosine path (embed_text) must be fully gone."""
        from bibilab.pipeline import chat_tools

        assert not hasattr(chat_tools, "embed_text")
        assert not hasattr(chat_tools, "_cosine_similarity")
        assert not hasattr(chat_tools, "ReuseAction")


class TestReuseCallerMechanics:
    """trivial_ack_note wired into run_chat_turn."""

    @pytest.mark.anyio
    async def test_non_trivial_keeps_tool_blocks(self, monkeypatch):
        """Non-trivial follow-up → tool_blocks NOT stripped → prior tool_use/
        tool_result replay into the LLM messages so it can reuse them."""
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
                        "arguments": {"query": "quantum", "expected_hits": "few"},
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
            source_list_str="Sources:\n" + _SRC_LIST_INSTRUCTION,
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
    async def test_trivial_with_no_prior_retrieve_injects_note(self, monkeypatch):
        """#308 behavior delta: a bare ack with NO prior retrieve in history
        still gets the no-retrieve note (loop/gate fully removed)."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.pipeline.chat_runs import ChatRunRegistry
        from bibilab.routers import chat as chat_module

        captured_system: list[str] = []

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            captured_system.append(system or "")
            yield StreamEvent(type="delta", content="you're welcome")
            yield StreamEvent(type="done")

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", noop)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        registry = ChatRunRegistry()
        registry.register("m2", task=None)
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        cfg = BibilabConfig(
            ai=AIConfig(protocol="anthropic", model="x", api_key="k", base_url=""),
            backend=BackendConfig(),
        )
        await chat_module.run_chat_turn(
            message_id="m2",
            conversation_id="c2",
            list_id="l2",
            user_message_text="ok",
            history=history,
            summary=None,
            source_ids=[],
            source_map={},
            source_list_str="Sources:\n" + _SRC_LIST_INSTRUCTION,
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )
        assert captured_system, "stream_llm never called"
        assert "acknowledgment" in captured_system[0].lower()

    @pytest.mark.anyio
    async def test_no_reused_from_prior_call_id_in_metadata(self, monkeypatch):
        """KEEP synthetic block deleted → rag.calls never carries reuse marker."""
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
                        "arguments": {"query": "quantum", "expected_hits": "few"},
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
            source_list_str="Sources:\n" + _SRC_LIST_INSTRUCTION,
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )
        calls = captured.get("meta", {}).get("rag", {}).get("calls", [])
        assert all("reused_from_prior_call_id" not in c for c in calls), calls

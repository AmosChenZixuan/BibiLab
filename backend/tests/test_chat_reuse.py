"""Tests for cross-turn retrieval reuse classifier (#280)."""

import pytest

from bibilab.pipeline.chat_tools import ReuseAction, decide_reuse

_SRC_LIST_INSTRUCTION = (
    "\n\nTo search, call retrieve. Include all source numbers except those clearly unrelated to the query."
)


class TestDecideReuse:
    """Unit tests for decide_reuse() decision matrix."""

    # --- Trivial message guard ---

    @pytest.mark.parametrize("msg", ["1", "12", "99"])
    def test_pure_digits_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            " ",
            "  ",
            "a",
            "ab",
            " a ",
            " . ",
            "是吗",  # 2 Chinese chars < 3 non-ws
        ],
    )
    def test_short_messages_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    @pytest.mark.parametrize(
        "msg",
        [
            "否",
            "ok",
            "OK",
            "Ok",
            "好的",
            "好",
            "是",
            "对",
            "yes",
            "no",
            "thanks",
            "谢谢",
            "k",
            "kk",
        ],
    )
    def test_stopwords_are_trivial(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action == ReuseAction.TRIVIAL

    def test_trivial_path_includes_note(self):
        decision = decide_reuse("1", "prior question")
        assert decision.note is not None
        assert "acknowledgment" in decision.note.lower()

    @pytest.mark.parametrize("msg", ["why", "what", "tell me more", "continue"])
    def test_non_trivial_short_messages_pass_guard(self, msg):
        decision = decide_reuse(msg, "prior question about physics")
        assert decision.action != ReuseAction.TRIVIAL

    # --- No prior context ---

    def test_no_prior_message_forces_fresh(self):
        decision = decide_reuse("tell me about quantum physics", None)
        assert decision.action == ReuseAction.FORCE_FRESH

    # --- Cosine similarity (requires embedding model) ---

    def test_identical_messages_keep(self):
        msg = "tell me about quantum mechanics in episode 3"
        decision = decide_reuse(msg, msg)
        assert decision.action == ReuseAction.KEEP

    def test_unrelated_messages_force_fresh(self):
        decision = decide_reuse(
            "what is the capital of France",
            "explain quantum mechanics in detail with equations",
        )
        assert decision.action == ReuseAction.FORCE_FRESH

    def test_paraphrase_same_topic_keep(self):
        decision = decide_reuse(
            "what did episode 3 say about quantum mechanics",
            "can you tell me about quantum mechanics in episode 3",
        )
        assert decision.action == ReuseAction.KEEP

    def test_deepening_followup_keep(self):
        decision = decide_reuse(
            "what are the key principles of quantum mechanics",
            "tell me about quantum mechanics in episode 3",
        )
        assert decision.action == ReuseAction.KEEP

    def test_topic_switch_force_fresh(self):
        decision = decide_reuse(
            "tell me about python programming",
            "explain the plot of episode 5 in detail",
        )
        assert decision.action == ReuseAction.FORCE_FRESH

    def test_embedding_failure_falls_back_to_force_fresh(self, monkeypatch):
        from bibilab.pipeline import chat_tools

        def boom(_):
            raise RuntimeError("ONNX session crashed")

        monkeypatch.setattr(chat_tools, "embed_text", boom)
        decision = decide_reuse("a non-trivial question", "another non-trivial question")
        assert decision.action == ReuseAction.FORCE_FRESH
        assert decision.note is None


class TestDecideReuseIntegration:
    """Integration tests: reuse classifier wired into run_chat_turn."""

    @pytest.mark.anyio
    async def test_force_fresh_strips_tool_blocks_from_expanded_messages(self, monkeypatch):
        """Topic switch → FORCE_FRESH → no tool_use/tool_result blocks in LLM messages."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.routers import chat as chat_module

        captured_messages: list[list[dict]] = []

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            captured_messages.append(list(messages))
            yield StreamEvent(type="delta", content="ok")
            yield StreamEvent(type="done")

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", noop)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        from bibilab.pipeline.chat_runs import ChatRunRegistry

        registry = ChatRunRegistry()
        msg_id = "msg-fresh-1"
        registry.register(msg_id, task=None)

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
            message_id=msg_id,
            conversation_id="c1",
            list_id="l1",
            user_message_text="what is the capital of France",
            history=history,
            summary=None,
            source_ids=[],
            source_map={},
            source_list_str="Sources:\n" + _SRC_LIST_INSTRUCTION,
            ui_lang="en",
            cfg=cfg,
            registry=registry,
        )

        assert captured_messages, "stream_llm not called"
        sent = captured_messages[0]
        # FORCE_FRESH: tool_blocks stripped → no assistant or tool_result blocks from history.
        # Expanded messages should be: [user (current)] only (no history replay).
        for m in sent:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    assert block.get("type") != "tool_use", f"tool_use block leaked into FORCE_FRESH path: {block}"
                    assert block.get("type") != "tool_result", (
                        f"tool_result block leaked into FORCE_FRESH path: {block}"
                    )

    @pytest.mark.anyio
    async def test_trivial_message_strips_tool_blocks_and_injects_note(self, monkeypatch):
        """Trivial acknowledgment → tool blocks stripped + system note injected."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.routers import chat as chat_module

        captured_system: list[str] = []
        captured_messages: list[list[dict]] = []

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            captured_messages.append(list(messages))
            captured_system.append(system or "")
            yield StreamEvent(type="delta", content="you're welcome")
            yield StreamEvent(type="done")

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", noop)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        from bibilab.pipeline.chat_runs import ChatRunRegistry

        registry = ChatRunRegistry()
        msg_id = "msg-trivial-1"
        registry.register(msg_id, task=None)

        history = [
            {"role": "user", "content": "tell me about quantum mechanics"},
            {
                "role": "assistant",
                "content": "first answer [1]",
                "tool_blocks": [
                    {
                        "tool_use_id": "toolu_b",
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
            message_id=msg_id,
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

        # Tool blocks must be stripped from expanded messages.
        for m in captured_messages[0]:
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    assert block.get("type") != "tool_use"
                    assert block.get("type") != "tool_result"

        # System note must be injected for TRIVIAL path.
        assert captured_system, "stream_llm never called"
        assert "acknowledgment" in captured_system[0].lower(), (
            f"Expected trivial-path note in system prompt, got: {captured_system[0][:200]}"
        )

    @pytest.mark.anyio
    async def test_keep_path_appends_reused_from_prior_call_id_in_metadata(self, monkeypatch):
        """Related follow-up → KEEP → synthetic reused_from_prior_call_id in rag.calls."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.routers import chat as chat_module

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            yield StreamEvent(type="delta", content="follow-up answer")
            yield StreamEvent(type="done")

        captured_metadata: dict = {}

        async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
            captured_metadata["meta"] = metadata

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", capture_update)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        from bibilab.pipeline.chat_runs import ChatRunRegistry

        registry = ChatRunRegistry()
        msg_id = "msg-keep-meta-1"
        registry.register(msg_id, task=None)

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
            message_id=msg_id,
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

        rag_calls = captured_metadata.get("meta", {}).get("rag", {}).get("calls", [])
        reused_entries = [c for c in rag_calls if c.get("reused_from_prior_call_id")]
        assert len(reused_entries) == 1, f"Expected 1 reused entry in rag.calls, got {len(reused_entries)}: {rag_calls}"
        entry = reused_entries[0]
        assert entry["query"] == "(reused)"
        assert entry["reused_from_prior_call_id"] == "toolu_reuse_1"

    @pytest.mark.anyio
    async def test_keep_streams_reuse_tool_result_during_stream(self, monkeypatch):
        """KEEP → a synthetic retrieve tool_result is streamed to the client so
        the reuse row shows mid-stream, not only after refresh."""
        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.routers import chat as chat_module

        async def fake_stream_llm(messages, cfg, tools=None, system=None, llm_max_tokens=2048):
            yield StreamEvent(type="delta", content="follow-up answer")
            yield StreamEvent(type="done")

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_llm", fake_stream_llm)
        monkeypatch.setattr(chat_module, "update_message_content", noop)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        from bibilab.pipeline.chat_runs import ChatRunRegistry

        registry = ChatRunRegistry()
        msg_id = "msg-keep-sse-1"
        buf = registry.register(msg_id, task=None)

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
            message_id=msg_id,
            conversation_id="c4",
            list_id="l4",
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

        reuse_frames = [
            e
            for e in buf.events
            if e.get("type") == "tool_result"
            and e.get("name") == "retrieve"
            and e.get("result", {}).get("reused_from_prior_call_id") == "toolu_reuse_1"
        ]
        assert len(reuse_frames) == 1, f"expected exactly one reuse tool_result frame, got {buf.events}"

    @pytest.mark.anyio
    async def test_keep_with_fresh_retrieve_streams_same_order_as_persisted(self, monkeypatch):
        """KEEP turn that also runs a fresh retrieve: reuse is the turn's first
        retrieval action, so it streams first AND is first in persisted
        rag.calls — streamed order matches post-refresh order."""
        import json

        from bibilab.config import AIConfig, BackendConfig, BibilabConfig
        from bibilab.pipeline._shared import StreamEvent
        from bibilab.routers import chat as chat_module

        async def fake_stream_with_tools(**kwargs):
            result = {
                "query": "quantum follow-up",
                "expected_hits": "few",
                "candidates_evaluated": 5,
                "sources_with_hits": 1,
                "sources_total": 1,
                "source_coverage": [],
                "dropped_by_gate": 0,
                "reranked": True,
            }
            yield StreamEvent(
                type="tool_result",
                content=json.dumps({"id": "r1", "name": "retrieve", "result": result}),
            )
            yield StreamEvent(type="delta", content="answer")
            yield StreamEvent(type="done")

        captured: dict = {}

        async def capture_update(message_id, content, metadata, status, error=None, tool_blocks=None):
            captured["meta"] = metadata

        async def noop(*a, **kw):
            return None

        monkeypatch.setattr(chat_module, "stream_with_tools", fake_stream_with_tools)
        monkeypatch.setattr(chat_module, "update_message_content", capture_update)
        monkeypatch.setattr(chat_module, "set_active_stream", noop)

        from bibilab.pipeline.chat_runs import ChatRunRegistry

        registry = ChatRunRegistry()
        msg_id = "msg-keep-order-1"
        buf = registry.register(msg_id, task=None)

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
            message_id=msg_id,
            conversation_id="c5",
            list_id="l5",
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

        # Streamed order: reuse tool_result first, then the fresh retrieve.
        tr = [e for e in buf.events if e.get("type") == "tool_result" and e.get("name") == "retrieve"]
        assert len(tr) == 2, f"expected reuse + real tool_result, got {buf.events}"
        assert tr[0]["result"]["reused_from_prior_call_id"] == "toolu_reuse_1", "reuse must stream first"
        assert "reused_from_prior_call_id" not in tr[1]["result"], "fresh retrieve streams after reuse"

        # Persisted order must match the streamed order.
        calls = captured["meta"]["rag"]["calls"]
        assert len(calls) == 2
        assert calls[0]["reused_from_prior_call_id"] == "toolu_reuse_1"
        assert calls[1]["query"] == "quantum follow-up"
        assert calls[1].get("reused_from_prior_call_id") is None

        # A final authoritative rag event is emitted just before the terminal
        # done, carrying the persisted-shape calls.
        types = [e.get("type") for e in buf.events]
        assert "rag" in types, f"no final rag event: {types}"
        rag_idx = types.index("rag")
        assert types[rag_idx + 1] == "done", "rag must immediately precede done"
        assert buf.events[rag_idx]["calls"] == calls

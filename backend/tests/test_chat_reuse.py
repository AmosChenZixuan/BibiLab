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


class TestExpandMessageForProviderCompaction:
    def test_expand_retrieve_block_replays_tag_not_chunks(self):
        """A prior-turn retrieve tool_block replays as a compact tag, not raw chunks."""
        from bibilab.pipeline.chat_tools import expand_message_for_provider

        msg = {
            "role": "assistant",
            "content": "决定主义是……[1]",
            "tool_blocks": [
                {
                    "tool_use_id": "tc1",
                    "name": "retrieve",
                    "arguments": {"query": "什么是决定主义"},
                    "result": {
                        "chunks": [
                            {
                                "source_id": "s1",
                                "chunk_id": "c1",
                                "content": "决定主义的原文段落",
                                "video_title": "血族",
                                "citation_index": 1,
                            },
                        ],
                        "summary": {
                            "query": "什么是决定主义",
                            "source_coverage": [{"source_id": "s1", "video_id": "v1", "title": "血族"}],
                        },
                    },
                }
            ],
        }

        out = expand_message_for_provider(msg, protocol="anthropic")
        tool_result_block = out[1]["content"][0]
        replayed = tool_result_block["content"]
        assert "Prior-turn retrieval" in replayed
        assert "什么是决定主义" in replayed
        assert "血族" in replayed
        assert "决定主义的原文段落" not in replayed

    def test_expand_retrieve_block_openai_protocol_also_tagged(self):
        """Same compaction under the openai protocol shape."""
        from bibilab.pipeline.chat_tools import expand_message_for_provider

        msg = {
            "role": "assistant",
            "content": "answer",
            "tool_blocks": [
                {
                    "tool_use_id": "tc1",
                    "name": "survey",
                    "arguments": {"query": "面食"},
                    "result": {
                        "chunks": [
                            {
                                "source_id": "s1",
                                "chunk_id": "c1",
                                "content": "牛肉面做法详解",
                                "video_title": "美食",
                                "citation_index": 1,
                            }
                        ],
                        "summary": {
                            "query": "面食",
                            "source_coverage": [{"source_id": "s1", "video_id": "v1", "title": "美食"}],
                        },
                    },
                }
            ],
        }

        out = expand_message_for_provider(msg, protocol="openai")
        tool_msg = next(m for m in out if m.get("role") == "tool")
        assert "Prior-turn retrieval" in tool_msg["content"]
        assert "牛肉面做法详解" not in tool_msg["content"]

    def test_expand_metadata_block_replays_tag_not_result(self):
        """query_list_metadata replays as a compact tag — result omitted to prevent
        stale-count contamination after the source list changes."""
        from bibilab.pipeline.chat_tools import expand_message_for_provider

        msg = {
            "role": "assistant",
            "content": "there are 5 sources",
            "tool_blocks": [
                {
                    "tool_use_id": "tc1",
                    "name": "query_list_metadata",
                    "arguments": {"query_type": "count"},
                    "result": {"count": 5},
                }
            ],
        }

        out = expand_message_for_provider(msg, protocol="anthropic")
        replayed = out[1]["content"][0]["content"]
        assert "Prior-turn query_list_metadata" in replayed
        assert "count" in replayed  # query_type echoed
        assert "5" not in replayed  # raw count not replayed

    def test_expand_generate_report_block_replays_tag_not_result(self):
        """generate_report replays as a compact tag — terminal-tool full replay
        anchors the LLM into re-calling generate_report on unrelated questions."""
        from bibilab.pipeline.chat_tools import expand_message_for_provider

        msg = {
            "role": "assistant",
            "content": "",
            "tool_blocks": [
                {
                    "tool_use_id": "tc1",
                    "name": "generate_report",
                    "arguments": {
                        "type": "custom_report",
                        "prompt": "compare ep1 and ep9",
                        "source_ids": ["s1", "s2"],
                    },
                    "result": {
                        "artifact_id": "a1",
                        "job_id": "j1",
                        "name": "custom_report",
                        "type": "custom_report",
                    },
                }
            ],
        }

        out = expand_message_for_provider(msg, protocol="anthropic")
        replayed = out[1]["content"][0]["content"]
        assert "Prior-turn generate_report" in replayed
        assert 'type="custom_report"' in replayed
        assert "a1" not in replayed  # artifact_id not replayed
        assert "j1" not in replayed  # job_id not replayed

    def test_expand_generate_report_block_openai_protocol_also_tagged(self):
        from bibilab.pipeline.chat_tools import expand_message_for_provider

        msg = {
            "role": "assistant",
            "content": "",
            "tool_blocks": [
                {
                    "tool_use_id": "tc1",
                    "name": "generate_report",
                    "arguments": {"type": "study_guide", "prompt": "p", "source_ids": []},
                    "result": {"artifact_id": "a1", "job_id": "j1", "name": "x", "type": "study_guide"},
                }
            ],
        }

        out = expand_message_for_provider(msg, protocol="openai")
        tool_msg = next(m for m in out if m.get("role") == "tool")
        assert "Prior-turn generate_report" in tool_msg["content"]
        assert "a1" not in tool_msg["content"]

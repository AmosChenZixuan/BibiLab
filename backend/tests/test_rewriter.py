import pytest
from pydantic import ValidationError

from bibilab.pipeline.rewriter import RewriterIntent


class TestRewriterIntentInvariants:
    def test_retrieve_false_with_all_null_fields_is_valid(self):
        intent = RewriterIntent(retrieve=False)
        assert intent.retrieve is False
        assert intent.query is None
        assert intent.mode is None

    def test_retrieve_false_with_query_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, query="hello")

    def test_retrieve_false_with_mode_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, mode="narrow")

    def test_retrieve_false_with_sequence_number_raises(self):
        with pytest.raises(ValidationError, match="retrieve=false requires"):
            RewriterIntent(retrieve=False, sequence_number=5)

    def test_retrieve_true_with_query_and_mode_is_valid(self):
        intent = RewriterIntent(retrieve=True, query="multiplexing", mode="narrow")
        assert intent.query == "multiplexing"
        assert intent.mode == "narrow"

    def test_retrieve_true_without_query_raises(self):
        with pytest.raises(ValidationError, match="retrieve=true requires"):
            RewriterIntent(retrieve=True, mode="narrow")

    def test_retrieve_true_without_mode_raises(self):
        with pytest.raises(ValidationError, match="retrieve=true requires"):
            RewriterIntent(retrieve=True, query="x")

    def test_invalid_mode_value_raises(self):
        with pytest.raises(ValidationError):
            RewriterIntent(retrieve=True, query="x", mode="few")  # legacy value


class TestBuildRewriterPrompt:
    def test_includes_current_message(self):
        from bibilab.pipeline.rewriter import build_rewriter_prompt

        prompt = build_rewriter_prompt(current="什么是多元主义", prior=[])
        assert "什么是多元主义" in prompt

    def test_includes_prior_user_messages_with_retrieve_tags(self):
        from bibilab.pipeline.rewriter import PriorUserTurn, build_rewriter_prompt

        prior = [
            PriorUserTurn(text="第五集讲了什么", retrieved=True),
            PriorUserTurn(text="嗯", retrieved=False),
        ]
        prompt = build_rewriter_prompt(current="继续", prior=prior)
        assert "第五集讲了什么" in prompt
        assert "嗯" in prompt
        assert "retrieve=true" in prompt
        assert "retrieve=false" in prompt

    def test_caps_window_at_five_prior_turns(self):
        from bibilab.pipeline.rewriter import PriorUserTurn, build_rewriter_prompt

        prior = [PriorUserTurn(text=f"msg{i}", retrieved=True) for i in range(10)]
        prompt = build_rewriter_prompt(current="继续", prior=prior)
        assert "msg9" in prompt  # most recent kept
        assert "msg4" not in prompt  # 5th-from-last dropped

    def test_no_assistant_content_in_prompt(self):
        from bibilab.pipeline.rewriter import PriorUserTurn, build_rewriter_prompt

        prior = [PriorUserTurn(text="第五集讲了什么", retrieved=True)]
        prompt = build_rewriter_prompt(current="继续", prior=prior)
        lower = prompt.lower()
        assert "assistant:" not in lower
        assert "tool_result" not in lower


class TestParseRewriterResponse:
    def test_parses_valid_json(self):
        from bibilab.pipeline.rewriter import parse_rewriter_response

        raw = '{"retrieve": true, "query": "x", "mode": "narrow", "sequence_number": null, "season_number": null}'
        intent = parse_rewriter_response(raw)
        assert intent.retrieve is True
        assert intent.query == "x"
        assert intent.mode == "narrow"

    def test_strips_markdown_fences(self):
        from bibilab.pipeline.rewriter import parse_rewriter_response

        raw = '```json\n{"retrieve": false}\n```'
        intent = parse_rewriter_response(raw)
        assert intent.retrieve is False

    def test_returns_none_on_invalid_json(self):
        from bibilab.pipeline.rewriter import parse_rewriter_response

        assert parse_rewriter_response("not json at all") is None

    def test_returns_none_on_invariant_violation(self):
        from bibilab.pipeline.rewriter import parse_rewriter_response

        raw = '{"retrieve": false, "query": "x"}'
        assert parse_rewriter_response(raw) is None


class TestFallbackIntent:
    def test_uses_raw_message_and_narrow_mode(self):
        from bibilab.pipeline.rewriter import fallback_intent

        intent = fallback_intent("hello there")
        assert intent.retrieve is True
        assert intent.query == "hello there"
        assert intent.mode == "narrow"


class TestRunRewriter:
    def _cfg(self):
        from bibilab.config import AIConfig

        return AIConfig(protocol="anthropic", model="claude-haiku-4-5-20251001", api_key="test")

    def test_happy_path_returns_intent_and_telemetry(self):
        from unittest.mock import patch

        from bibilab.pipeline.rewriter import run_rewriter

        raw = '{"retrieve": true, "query": "x", "mode": "narrow", "sequence_number": null, "season_number": null}'
        with patch("bibilab.pipeline.rewriter._call_llm", return_value=raw):
            intent, tel = run_rewriter(current="x?", prior=[], cfg=self._cfg())
        assert intent.retrieve is True
        assert tel["fallback"] is False
        assert tel["mode"] == "narrow"
        assert tel["latency_ms"] >= 0

    def test_invalid_json_falls_back(self):
        from unittest.mock import patch

        from bibilab.pipeline.rewriter import run_rewriter

        with patch("bibilab.pipeline.rewriter._call_llm", return_value="not json"):
            intent, tel = run_rewriter(current="hello", prior=[], cfg=self._cfg())
        assert tel["fallback"] is True
        assert intent.query == "hello"
        assert intent.mode == "narrow"

    def test_llm_exception_falls_back(self):
        from unittest.mock import patch

        from bibilab.pipeline.rewriter import run_rewriter

        with patch("bibilab.pipeline.rewriter._call_llm", side_effect=RuntimeError("timeout")):
            intent, tel = run_rewriter(current="hello", prior=[], cfg=self._cfg())
        assert tel["fallback"] is True
        assert intent.retrieve is True

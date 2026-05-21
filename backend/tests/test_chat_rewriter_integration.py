"""Tests for rewriter integration in chat.py — _build_prior_user_turns."""

from bibilab.pipeline.rewriter import PriorUserTurn
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

    def test_missing_content_key_defaults_to_empty(self):
        result = _build_prior_user_turns([{"role": "user"}])
        assert result == [PriorUserTurn(text="", retrieved=False)]

    def test_tool_blocks_is_none(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_blocks": None},
        ]
        result = _build_prior_user_turns(history)
        assert result == [PriorUserTurn(text="hi", retrieved=False)]

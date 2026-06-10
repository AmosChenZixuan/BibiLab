"""Tests for the per-LLM-call output budget (resolve_max_tokens + max_output_tokens)."""

import pytest
from pydantic import ValidationError

from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    ContextWindowExceededError,
    LLMEmptyResponseError,
    LLMOutputBudgetExceededError,
    _no_text_error,
    _serialize_messages,
    resolve_max_tokens,
)
from bibilab.pipeline.chat_tools import TOOL_NAME_FIND_PASSAGES


def _cfg(context_window: int = 128000, max_output_tokens: int = 16384) -> AIConfig:
    return AIConfig(
        protocol="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="sk-test",
        base_url="",
        output_language="en",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )


@pytest.mark.parametrize(
    "max_output_tokens",
    [16384, 32768, 65536, 102400],
    ids=["16K", "32K", "64K", "100K"],
)
def test_small_input_returns_user_ceiling(max_output_tokens: int) -> None:
    """The user-chosen output budget is returned verbatim for any input that
    fits in the context window. The 4 user-selectable values are the slot
    slider positions on the LLM tab — each must round-trip through the
    formula unchanged."""
    assert resolve_max_tokens(_cfg(max_output_tokens=max_output_tokens), "hello") == max_output_tokens


def test_input_filling_window_raises_instead_of_overflowing() -> None:
    """Input that overflows the window raises a factual error (no valid max_tokens
    exists) rather than emitting one that would overflow the provider's limit.

    The check is `input + max_output + margin > context_window`. Here the input
    alone exceeds the window, so no allocation is valid."""
    huge = "word " * 200000  # exceeds a 128K window outright
    with pytest.raises(ContextWindowExceededError, match="context window"):
        resolve_max_tokens(_cfg(context_window=128000), huge)


def test_input_at_window_margin_raises() -> None:
    """`input + max_output + margin > context_window` must raise rather than
    return 0 or a negative number — pins the boundary guard so a future
    `<=`/`>` flip doesn't silently return a value most providers reject.

    window=20000, margin=2048, max_output=16384, "hello"=1 token →
    1 + 16384 + 2048 = 18433, which is less than 20000, so this should
    succeed at the margin. Use a tighter window to force the raise.
    """
    with pytest.raises(ContextWindowExceededError, match="context window"):
        # window=20000, max_output=16384, margin=2048, input=~1600 tokens
        # → 1600 + 16384 + 2048 = 20032 > 20000 → raise
        resolve_max_tokens(
            _cfg(context_window=20000, max_output_tokens=16384),
            "word " * 1600,
        )


def test_user_ceiling_under_input_margin_returns_user_ceiling() -> None:
    """For any (context_window, max_output_tokens) pair where input fits
    under the available budget, the user's choice is returned unchanged —
    no dynamic scaling, no slow path."""
    cfg = _cfg(context_window=200000, max_output_tokens=100000)
    out = resolve_max_tokens(cfg, "word " * 1000)  # ~1K tokens
    assert out == 100000


def test_ai_config_rejects_max_output_greater_or_equal_context_window() -> None:
    """Cross-field validator: max_output_tokens must be strictly less than
    context_window, otherwise any LLM call would fail-loud on the first
    overflow check. Pydantic raises ValidationError on construction."""
    with pytest.raises(ValidationError, match="max_output_tokens"):
        AIConfig(
            protocol="anthropic",
            model="m",
            api_key="",
            base_url="",
            context_window=100000,
            max_output_tokens=100000,  # equal, not strictly less
        )
    with pytest.raises(ValidationError, match="max_output_tokens"):
        AIConfig(
            protocol="anthropic",
            model="m",
            api_key="",
            base_url="",
            context_window=100000,
            max_output_tokens=200000,  # greater than context
        )


def test_ai_config_accepts_strictly_less() -> None:
    """Counterpart: max_output_tokens < context_window is accepted."""
    cfg = AIConfig(
        protocol="anthropic",
        model="m",
        api_key="",
        base_url="",
        context_window=128000,
        max_output_tokens=16384,
    )
    assert cfg.max_output_tokens == 16384
    assert cfg.context_window == 128000


def test_llm_output_budget_exceeded_subclasses_value_error() -> None:
    """The new error must subclass ValueError so any legacy `except ValueError`
    in pipeline code (chat_summary, worker artifact, digest) continues to
    catch it. Error-class rollouts rely on this base-class contract."""
    assert issubclass(LLMOutputBudgetExceededError, ValueError)
    err = LLMOutputBudgetExceededError("test")
    assert isinstance(err, ValueError)
    assert str(err) == "test"


def test_llm_empty_response_subclasses_value_error() -> None:
    """LLMEmptyResponseError must also subclass ValueError (same legacy-catch
    reason). It is the non-budget no-text case so digest's retry loop catches
    it via the generic `except Exception` and retries — it is NOT in the
    immediate re-raise tuple."""
    assert issubclass(LLMEmptyResponseError, ValueError)
    assert isinstance(LLMEmptyResponseError("x"), ValueError)


@pytest.mark.parametrize("reason", ["max_tokens", "length"])
def test_no_text_error_length_cutoff_is_budget_error(reason: str) -> None:
    """A length/max_tokens cutoff is the ONLY no-text case that justifies the
    'raise max output tokens' hint → LLMOutputBudgetExceededError."""
    err = _no_text_error(reason, 16384)
    assert isinstance(err, LLMOutputBudgetExceededError)
    assert "output token limit" in str(err)


@pytest.mark.parametrize("reason", ["stop", "end_turn", "content_filter", "unknown", None])
def test_no_text_error_other_reasons_are_empty_response(reason) -> None:
    """Any non-length terminal reason (normal stop, refusal, unknown, missing)
    → LLMEmptyResponseError, so we never give false budget advice. Defaulting
    the unknown/None case here is deliberate: better a neutral 'try again' than
    a wrong 'raise max output tokens'."""
    err = _no_text_error(reason, 16384)
    assert isinstance(err, LLMEmptyResponseError)
    assert not isinstance(err, LLMOutputBudgetExceededError)


def test_serialize_messages_includes_tool_calls() -> None:
    """OpenAI tool-calling turns carry tool_calls / tool_call_id / name on the
    message envelope; the serializer must count them, not just content."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME_FIND_PASSAGES,
                        "arguments": '{"query": "a long query string here"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": TOOL_NAME_FIND_PASSAGES,
            "content": "tool result payload",
        },
    ]
    serialized = _serialize_messages(messages, system="sys", tools=None)
    # tool_calls JSON + tool_call_id + name all visible — not just "null\ntool result payload"
    assert "find_passages" in serialized
    assert "a long query string here" in serialized
    assert "c1" in serialized
    # Sanity: the content-only baseline (without the new fields) would be much shorter.
    content_only = "sys\nnull\ntool result payload"
    assert len(serialized) > len(content_only)


def test_serialize_messages_includes_anthropic_tool_blocks() -> None:
    """Anthropic tool-calling turns carry tool_use / tool_result inside the
    `content` list (no top-level tool_calls / tool_call_id). The serializer's
    list-content JSON-dump must capture the function name and arguments.

    Guards against future refactors that flatten content to string and would
    silently drop the tool-use data on the Anthropic path.
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look that up."},
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": TOOL_NAME_FIND_PASSAGES,
                    "input": {"query": "a long anthropic query string"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu1",
                    "content": '[{"chunk": "excerpt"}]',
                }
            ],
        },
    ]
    serialized = _serialize_messages(messages, system="sys", tools=None)
    # The function name, input JSON, and tool_result payload all ride inside
    # content — so a content-list JSON-dump is the path that counts them.
    assert "find_passages" in serialized
    assert "a long anthropic query string" in serialized
    assert "tu1" in serialized
    assert "excerpt" in serialized

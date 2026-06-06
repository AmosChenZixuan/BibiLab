"""Tests for the unified per-LLM-call output budget (resolve_max_tokens)."""

import pytest

from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _OUTPUT_CEILING,
    ContextWindowExceededError,
    _serialize_messages,
    resolve_max_tokens,
)


def _cfg(context_window: int = 128000) -> AIConfig:
    return AIConfig(
        protocol="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="sk-test",
        base_url="",
        output_language="en",
        context_window=context_window,
    )


def test_small_input_returns_ceiling():
    """A tiny prompt against a large window leaves the ceiling as the binding cap."""
    assert resolve_max_tokens(_cfg(), "hello") == _OUTPUT_CEILING


def test_large_input_shrinks_to_fit_below_ceiling():
    """When the window can't fit input + ceiling, output shrinks to exactly what fits
    (no floor override) — the read_source-carries-a-transcript case."""
    big = "word " * 120000  # ~120K tokens against a 128K window
    out = resolve_max_tokens(_cfg(context_window=128000), big)
    assert 0 < out < _OUTPUT_CEILING


def test_input_filling_window_raises_instead_of_overflowing():
    """Input that overflows the window raises a factual error (no valid max_tokens
    exists) rather than emitting one that would overflow the provider's limit."""
    huge = "word " * 200000  # exceeds a 128K window outright
    with pytest.raises(ContextWindowExceededError, match="context window"):
        resolve_max_tokens(_cfg(context_window=128000), huge)


def test_small_window_is_respected():
    """A smaller window scales the budget down accordingly (still no ceiling)."""
    out = resolve_max_tokens(_cfg(context_window=8192), "hello")
    assert 0 < out < _OUTPUT_CEILING


def test_input_at_window_margin_raises():
    """available == 0 must raise, not return 0 — pins the `<= 0` guard so a
    future `< 0` change doesn't silently return a value most providers reject.

    window=2049, margin=2048, "hello"=1 token → available = 2049-1-2048 = 0 → raise.
    """
    with pytest.raises(ContextWindowExceededError, match="context window"):
        resolve_max_tokens(_cfg(context_window=2049), "hello")


def test_serialize_messages_includes_tool_calls():
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
                        "name": "find_passages",
                        "arguments": '{"query": "a long query string here"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "find_passages",
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


def test_serialize_messages_includes_anthropic_tool_blocks():
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
                    "name": "find_passages",
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

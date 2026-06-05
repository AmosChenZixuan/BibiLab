"""Tests for the unified per-LLM-call output budget (resolve_max_tokens)."""

import pytest

from bibilab.config import AIConfig
from bibilab.pipeline._shared import (
    _OUTPUT_CEILING,
    ContextWindowExceededError,
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

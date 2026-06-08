"""Artifact section-batched refine.

The artifact job used to stuff every selected source's full transcript into
one _call_llm and raise ContextWindowExceededError on overflow. Now we
batch sections, feeding the running draft back to the LLM. Short inputs
(1 section per source) stay byte-identical to the old prompt.
"""

from __future__ import annotations

import pytest

from bibilab.worker import (
    _format_duration,
    _pack_sections,
    _render_multi_batch_section,
    _render_single_batch_text,
    _SectionView,
)

pytestmark = pytest.mark.integration


def _sv(source_id: str, seq: int, text: str, *, tokens: int | None = None) -> _SectionView:
    return _SectionView(
        source_id=source_id,
        source_title=f"Title-{source_id}",
        seq=seq,
        timestamp_start=seq * 60.0,
        timestamp_end=(seq + 1) * 60.0,
        text=text,
        token_count=tokens if tokens is not None else len(text.split()),
    )


# --- _pack_sections ---------------------------------------------------------


def test_pack_sections_empty():
    assert _pack_sections([], budget_tokens=100) == []


def test_pack_sections_single_section_under_budget():
    s1 = _sv("src-1", 0, "hello world", tokens=2)
    assert _pack_sections([s1], budget_tokens=10) == [[s1]]


def test_pack_sections_greedy_fill_two_batches():
    # Budget = 6 tokens. Sec A=3, B=3, C=1, D=1, E=1 → A+B=6 fits exactly;
    # C would push over (6+1=7>6), so the batch flushes; [C,D,E]=3 fits
    # in the second batch. Greedy trace: [A,B], then [C,D,E].
    a = _sv("src-1", 0, "aaa", tokens=3)
    b = _sv("src-1", 1, "bbb", tokens=3)
    c = _sv("src-1", 2, "c", tokens=1)
    d = _sv("src-1", 3, "d", tokens=1)
    e = _sv("src-1", 4, "e", tokens=1)
    batches = _pack_sections([a, b, c, d, e], budget_tokens=6)
    assert len(batches) == 2
    assert batches[0] == [a, b]
    assert batches[1] == [c, d, e]


def test_pack_sections_section_alone_overflows_returns_solo_batch():
    """A section that alone exceeds the budget stays alone in its batch.
    The caller checks this and raises PipelineError."""
    big = _sv("src-1", 0, "X" * 100, tokens=50)
    small = _sv("src-1", 1, "y", tokens=1)
    batches = _pack_sections([big, small], budget_tokens=10)
    # big goes alone (it alone exceeds the budget; greedy can't split it);
    # then small joins the next batch.
    assert batches == [[big], [small]]


def test_pack_sections_preserves_source_order():
    """Section order within a batch = source order (selected source ids)
    then seq within source. Packing must not reorder."""
    a = _sv("src-A", 0, "a0")
    b = _sv("src-A", 1, "a1")
    c = _sv("src-B", 0, "b0")
    d = _sv("src-B", 1, "b1")
    batches = _pack_sections([a, b, c, d], budget_tokens=10_000)
    flat = [s for batch in batches for s in batch]
    assert flat == [a, b, c, d]


# --- section text renderers --------------------------------------------------


def test_format_duration_minutes_seconds():
    assert _format_duration(0) == "00:00"
    assert _format_duration(5) == "00:05"
    assert _format_duration(65) == "01:05"
    assert _format_duration(3661) == "61:01"  # past 1h: keep counting minutes (no H:MM:SS)


def test_render_single_batch_text_groups_by_source_with_no_section_header():
    """Single-batch (byte-identical to today): `=== Source: {title} ===\n
    {concatenated section text per source, no per-section header)`. Sections
    for the same source are concatenated in seq order."""
    a0 = _sv("src-A", 0, "A0 text.", tokens=2)
    a1 = _sv("src-A", 1, "A1 text.", tokens=2)
    b0 = _sv("src-B", 0, "B0 text.", tokens=2)
    out = _render_single_batch_text([a0, a1, b0])
    # Source A: header + a0 + a1 (no per-section header, concatenated)
    # Source B: header + b0
    # Joined with the same separator today's _run_artifact_job uses ("\n\n").
    expected = "=== Source: Title-src-A ===\nA0 text.A1 text.\n\n=== Source: Title-src-B ===\nB0 text."
    assert out == expected


def test_render_multi_batch_section_uses_per_section_header():
    """Multi-batch path: each section gets `=== Source: {title} · Section
    {seq} (mm:ss-mm:ss) ===\n{text}` (no per-section blank lines; the
    caller controls inter-section spacing in the prompt)."""
    a0 = _sv("src-A", 0, "A0 text.", tokens=2)
    out = _render_multi_batch_section(a0)
    expected = "=== Source: Title-src-A · Section 0 (00:00-01:00) ===\nA0 text."
    assert out == expected

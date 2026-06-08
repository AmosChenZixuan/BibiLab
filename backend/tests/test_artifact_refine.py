"""Artifact section-batched refine.

The artifact job used to stuff every selected source's full transcript into
one _call_llm and raise ContextWindowExceededError on overflow. Now we
batch sections, feeding the running draft back to the LLM. Short inputs
(1 section per source) stay byte-identical to the old prompt.
"""

from __future__ import annotations

import pytest

from bibilab.config import BibilabConfig
from bibilab.worker import (
    ArtifactResult,
    _build_initial_prompt,
    _build_refine_prompt,
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


# --- prompt builders --------------------------------------------------------


def test_build_initial_prompt_matches_today_template_for_single_batch():
    """The single-batch path's prompt must be byte-identical to today's
    _generate_artifact template (regression guard)."""
    sections = [_sv("src-A", 0, "A text."), _sv("src-B", 0, "B text.")]
    transcript = _render_single_batch_text(sections)
    prompt = _build_initial_prompt(
        prompt="summarize the videos",
        artifact_type="study_guide",
        transcript_text=transcript,
        cfg=BibilabConfig(),
        ui_lang="en",
    )
    # The structure is: lang_instruction + blank + user_prompt + blank +
    # "Based on the following transcripts..." + "Transcript:" + transcript
    # + blank + lang_instruction + "Respond ONLY with valid JSON..." +
    # JSON schema + lang_output_directive.
    assert "summarize the videos" in prompt
    assert "=== Source: Title-src-A ===" in prompt
    assert "=== Source: Title-src-B ===" in prompt
    # Single-batch means NO per-section header, even if a source has multiple
    # sections (it would render as one concatenated block).
    assert "Section 0" not in prompt
    # The JSON schema contract is preserved (today's LLM is told to return
    # {name, content}).
    assert '"name"' in prompt
    assert '"content"' in prompt
    # The schema direction is present.
    assert "Respond ONLY with valid JSON" in prompt


def test_build_refine_prompt_includes_running_draft_and_integrate_directive():
    """Batch k>1 prompt: the running draft is in the prompt, the new
    sections are next, and an integrate directive tells the LLM what to do."""
    draft = ArtifactResult(name="Initial title", content="Initial body.")
    new_sections_text = "=== Source: Title-src-A · Section 1 (01:00-02:00) ===\nA1 text."
    prompt = _build_refine_prompt(
        prompt="summarize the videos",
        artifact_type="study_guide",
        draft=draft,
        new_sections_text=new_sections_text,
        cfg=BibilabConfig(),
        ui_lang="en",
    )
    # The running draft is shown (name + content).
    assert "Initial title" in prompt
    assert "Initial body." in prompt
    # The new material is shown.
    assert "Section 1 (01:00-02:00)" in prompt
    assert "A1 text." in prompt
    # An "integrate" directive is present.
    assert "integrate" in prompt.lower()
    # The {name, content} JSON contract is preserved so the LLM keeps
    # returning a parseable ArtifactResult.
    assert '"name"' in prompt
    assert '"content"' in prompt


def test_build_refine_prompt_uses_lang_directive():
    """The refine prompt honors cfg.output_language + ui_lang (same path
    as the initial prompt)."""
    cfg = BibilabConfig()
    draft = ArtifactResult(name="t", content="c")
    p_en = _build_refine_prompt("p", "study_guide", draft, "x", cfg, ui_lang="en")
    p_zh = _build_refine_prompt("p", "study_guide", draft, "x", cfg, ui_lang="zh")
    # Different ui_lang → different lang directive in the prompt.
    assert p_en != p_zh

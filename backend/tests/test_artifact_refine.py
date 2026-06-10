"""Artifact section-batched refine.

The artifact job used to stuff every selected source's full transcript into
one _call_llm and raise ContextWindowExceededError on overflow. Now we
batch sections, feeding the running draft back to the LLM. Short inputs
(1 section per source) stay byte-identical to the old prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from bibilab.config import BibilabConfig
from bibilab.db import bootstrap_db, create_list, get_artifact
from bibilab.pipeline.audio import PipelineError
from bibilab.pipeline.section import Section
from bibilab.pipeline.transcribe import WhisperSegment
from bibilab.worker import (
    ArtifactResult,
    WorkerLoop,
    _build_initial_prompt,
    _build_refine_prompt,
    _build_section_views,
    _format_duration,
    _pack_sections,
    _refine_artifact,
    _render_multi_batch_section,
    _render_single_batch_text,
    _SectionView,
)
from tests.factories import SourceFactory

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
    """Single-batch: `=== Source: {title} ===\\n{per-section text, joined
    with \\n so format_turns turns don't glue together, no per-section
    header)}`. Sources joined with the same `\\n\\n` separator the
    legacy artifact path used."""
    a0 = _sv("src-A", 0, "A0 text.", tokens=2)
    a1 = _sv("src-A", 1, "A1 text.", tokens=2)
    b0 = _sv("src-B", 0, "B0 text.", tokens=2)
    out = _render_single_batch_text([a0, a1, b0])
    # Source A: header + a0 + "\n" + a1 (no per-section header, sections
    # joined with \n so adjacent turns don't glue)
    # Source B: header + b0
    # Sources joined with the same "\n\n" separator today's _run_artifact_job used.
    expected = "=== Source: Title-src-A ===\nA0 text.\nA1 text.\n\n=== Source: Title-src-B ===\nB0 text."
    assert out == expected


def test_render_single_batch_text_multi_section_source_uses_newline_separator():
    """Regression: a single source with multiple sections must NOT glue
    adjacent sections together on one line. format_turns produces no
    trailing newline, so a ""-join produces corrupted lines like
    'world[SPK_0] start'. The newline-separator preserves readability."""
    sec0 = _sv("src-A", 0, "[SPK_0] hello")
    sec1 = _sv("src-A", 1, "[SPK_0] world")
    out = _render_single_batch_text([sec0, sec1])
    # Each section's text on its own line; the last turn of sec0 doesn't
    # abut the first turn of sec1.
    assert "hello\n[SPK_0] world" in out
    assert "hello[SPK_0] world" not in out  # the broken shape


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
    p_en = _build_refine_prompt("p", draft, "x", cfg, ui_lang="en")
    p_zh = _build_refine_prompt("p", draft, "x", cfg, ui_lang="zh")
    # Different ui_lang → different lang directive in the prompt.
    assert p_en != p_zh


# --- _build_section_views ---------------------------------------------------


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        yield tmp_path


@pytest.mark.asyncio
async def test_build_section_views_no_sections_raises_pipeline_error(tmp_bibilab_home):
    """A source with no section rows: _build_section_views raises
    PipelineError with the documented message (fail-loud contract)."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # SourceFactory.build without sections=... → no section rows.
    source_id = await SourceFactory.build("list-1", video_id="BVx")

    with pytest.raises(PipelineError, match="no sections; re-ingest required"):
        await _build_section_views([source_id])


@pytest.mark.asyncio
async def test_build_section_views_happy_path_returns_verbatim_text(tmp_bibilab_home):
    """1 source with 2 sections, 1 segment each: returns 2 _SectionView
    objects in seq order with verbatim text reconstructed from the segment
    slice. Source order across sources is also preserved."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        title="T",
        uploader="u",
        source_url="https://x",
        segments=[
            WhisperSegment(start=0.0, end=1.0, text="alpha", speaker="SPK_0"),
            WhisperSegment(start=1.0, end=2.0, text="beta", speaker="SPK_0"),
        ],
        sections=[
            Section(seg_start=0, seg_end=0, token_count=1, timestamp_start=0.0, timestamp_end=1.0),
            Section(seg_start=1, seg_end=1, token_count=1, timestamp_start=1.0, timestamp_end=2.0),
        ],
    )

    views = await _build_section_views([source_id])

    assert len(views) == 2
    # Ordered by source then seq (single source here, so seq order is the test).
    assert views[0].seq == 0
    assert views[1].seq == 1
    # Verbatim text reconstructed from the segment slice (format_turns
    # with include_time=False, same speaker → one line "[SPK_0] text").
    assert views[0].text == "[SPK_0] alpha"
    assert views[1].text == "[SPK_0] beta"
    # Token count is the precomputed value from the sections table.
    assert views[0].token_count == 1
    assert views[1].token_count == 1
    # Source title propagated from the source row.
    assert views[0].source_title == views[1].source_title
    # Timestamps propagated from the sections table.
    assert views[0].timestamp_start == 0.0
    assert views[1].timestamp_end == 2.0


# --- _refine_artifact (single-batch regression guard) ----------------------


@pytest.mark.asyncio
async def test_refine_artifact_single_batch_byte_identical_prompt():
    """Regression guard: when all sections fit in one batch, _refine_artifact
    calls _call_llm exactly once with the same prompt template today's
    _generate_artifact would build, and returns the parsed ArtifactResult.

    The "byte-identical" claim is structural (same template + same JSON
    schema + same lang directive + same transcript text shape), not a
    test-time string equality — LLM output is stochastic."""
    cfg = BibilabConfig()
    # Tiny context window that still fits 2 small sections easily.
    cfg.ai.context_window = 32_000
    cfg.ai.max_output_tokens = 4_000
    sections = [
        _SectionView(
            source_id="src-A",
            source_title="A",
            seq=0,
            timestamp_start=0.0,
            timestamp_end=1.0,
            text="A text.",
            token_count=2,
        ),
        _SectionView(
            source_id="src-B",
            source_title="B",
            seq=0,
            timestamp_start=0.0,
            timestamp_end=1.0,
            text="B text.",
            token_count=2,
        ),
    ]
    expected_response = '{"name": "My Title", "content": "My body."}'

    with patch("bibilab.pipeline._shared._call_llm", return_value=expected_response) as mock_llm:
        result = await _refine_artifact(
            prompt="summarize",
            sections=sections,
            cfg=cfg,
            ui_lang="en",
        )

    assert mock_llm.call_count == 1
    # The single call's prompt must contain both source headers, no
    # per-section header, and the JSON schema.
    prompt_arg = mock_llm.call_args[0][0]
    assert "=== Source: A ===" in prompt_arg
    assert "=== Source: B ===" in prompt_arg
    assert "Section 0" not in prompt_arg  # no per-section header on single-batch
    assert '"name"' in prompt_arg
    assert '"content"' in prompt_arg
    # The parsed result is returned.
    assert isinstance(result, ArtifactResult)
    assert result.name == "My Title"
    assert result.content == "My body."


# --- _refine_artifact (multi-batch path) -----------------------------------


@pytest.mark.asyncio
async def test_refine_artifact_multi_batch_calls_llm_per_batch():
    """2 sections that don't fit in one batch → 2 _call_llm calls. Batch
    1 uses the initial prompt template; batch 2 uses _build_refine_prompt
    with the running draft. The final ArtifactResult is the second call's
    parsed output."""
    cfg = BibilabConfig()
    # budget = 60: A=50, B=50 → 2 batches (each section alone).
    cfg.ai.context_window = 60 + 2 * 4000 + 500
    cfg.ai.max_output_tokens = 4000

    sections = [
        _SectionView("src-1", "A", 0, 0.0, 1.0, "A text.", token_count=50),
        _SectionView("src-1", "A", 1, 1.0, 2.0, "A1 text.", token_count=50),
    ]
    responses = [
        '{"name": "Initial", "content": "First pass."}',
        '{"name": "Refined", "content": "Second pass."}',
    ]
    with patch("bibilab.pipeline._shared._call_llm", side_effect=responses) as mock_llm:
        result = await _refine_artifact(
            prompt="summarize",
            sections=sections,
            cfg=cfg,
            ui_lang="en",
        )

    assert mock_llm.call_count == 2
    # First prompt: no "Current draft" section (it's the initial draft).
    p1 = mock_llm.call_args_list[0][0][0]
    assert "Current draft" not in p1
    # Second prompt: contains the first draft + integrate directive.
    p2 = mock_llm.call_args_list[1][0][0]
    assert "Current draft" in p2
    assert "First pass." in p2
    assert "integrate" in p2.lower()
    # Per-section header is on the multi-batch path.
    assert "Section 1" in p2
    # The final result is the LAST call's output (refined draft).
    assert result.name == "Refined"
    assert result.content == "Second pass."


@pytest.mark.asyncio
async def test_refine_artifact_single_section_overflow_fails_loud():
    """A single section whose token_count > budget → PipelineError
    (sections are atomic, never split). No _call_llm call is made."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 4_904
    cfg.ai.max_output_tokens = 1_000
    # budget = 4904 - 2*1000 - 500 = 2404

    sections = [
        _SectionView("src-1", "A", 0, 0.0, 1.0, "A text.", token_count=5_000),  # > 2404
    ]
    with patch("bibilab.pipeline._shared._call_llm") as mock_llm:
        with pytest.raises(PipelineError, match="alone"):
            await _refine_artifact(
                prompt="p",
                sections=sections,
                cfg=cfg,
                ui_lang="en",
            )
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_refine_artifact_batch_failure_raises_pipeline_error():
    """A batch whose _call_llm raises (not ContextWindowExceededError) is
    retried 3 times, then PipelineError. No partial artifact."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 60 + 2 * 4000 + 500
    cfg.ai.max_output_tokens = 4000

    sections = [
        _SectionView("src-1", "A", 0, 0.0, 1.0, "A text.", token_count=50),
        _SectionView("src-1", "A", 1, 1.0, 2.0, "A1 text.", token_count=50),
    ]
    with patch("bibilab.pipeline._shared._call_llm", side_effect=RuntimeError("boom")) as mock_llm:
        with pytest.raises(PipelineError, match="exhausted all retries"):
            await _refine_artifact(
                prompt="p",
                sections=sections,
                cfg=cfg,
                ui_lang="en",
            )
    # 3 attempts on the first (and only attempted) batch.
    assert mock_llm.call_count == 3


@pytest.mark.asyncio
async def test_refine_artifact_soft_cost_note_logs_warning_for_many_batches(caplog):
    """4+ batches → logger.warning fires (no schema/UI change)."""
    cfg = BibilabConfig()
    # budget = 75: 4 sections of 50 tokens each → 4 batches (50+50>75).
    cfg.ai.context_window = 75 + 2 * 4000 + 500
    cfg.ai.max_output_tokens = 4000
    sections = [_SectionView("src-1", "A", i, float(i), float(i + 1), f"text {i}", token_count=50) for i in range(4)]
    responses = [f'{{"name": "t{i}", "content": "c{i}"}}' for i in range(4)]
    with caplog.at_level(logging.WARNING, logger="bibilab.worker"):
        with patch("bibilab.pipeline._shared._call_llm", side_effect=responses):
            await _refine_artifact(
                prompt="p",
                sections=sections,
                cfg=cfg,
                ui_lang="en",
            )
    assert any("batches" in r.getMessage() and "threshold=3" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_refine_artifact_preserves_source_order_across_batches():
    """Source order in the input = source order across batches. Sections
    for src-A appear before sections for src-B even if they pack into
    different batches."""
    cfg = BibilabConfig()
    cfg.ai.context_window = 90 + 4000 + 4096 + 500
    cfg.ai.max_output_tokens = 4000
    # budget = 90: A=40, B=40 → 1 batch (40+40=80 ≤ 90)
    # Add a 3rd section to force a 2nd batch.
    cfg.ai.context_window = 90 + 2 * 4000 + 500
    cfg.ai.max_output_tokens = 4000
    sections = [
        _SectionView("src-A", "TA", 0, 0.0, 1.0, "a", token_count=40),
        _SectionView("src-B", "TB", 0, 0.0, 1.0, "b", token_count=40),
        _SectionView("src-B", "TB", 1, 1.0, 2.0, "b1", token_count=40),
    ]
    responses = [
        '{"name": "n1", "content": "c1"}',
        '{"name": "n2", "content": "c2"}',
    ]
    with patch("bibilab.pipeline._shared._call_llm", side_effect=responses) as mock_llm:
        await _refine_artifact(
            prompt="p",
            sections=sections,
            cfg=cfg,
            ui_lang="en",
        )
    p1 = mock_llm.call_args_list[0][0][0]
    p2 = mock_llm.call_args_list[1][0][0]
    # src-A header comes before src-B header in BOTH prompts.
    assert p1.index("Source: TA") < p1.index("Source: TB")
    # The second batch prompt (k=2) contains src-B (greedy pack: A+B in
    # batch 1, B1 in batch 2).
    assert "Source: TB" in p2


# --- end-to-end: _run_artifact_job uses _refine_artifact -------------------


@pytest.mark.asyncio
async def test_run_artifact_job_uses_refine_artifact_single_batch(tmp_bibilab_home):
    """End-to-end: an artifact job for 1 source (1 section) writes the
    artifact file + DB row. Regression guard: the LLM is called exactly
    once via _refine_artifact's single-batch path."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        segments=[
            WhisperSegment(start=0.0, end=1.0, text="hello", speaker="SPK_0"),
            WhisperSegment(start=1.0, end=2.0, text="world", speaker="SPK_0"),
        ],
        sections=[
            Section(seg_start=0, seg_end=1, token_count=2, timestamp_start=0.0, timestamp_end=2.0),
        ],
    )

    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job = {
        "id": "job-1",
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-1",
                "type": "study_guide",
                "prompt": "summarize",
                "source_ids": [source_id],
            }
        ),
    }

    async def _fake_refine(*, prompt, sections, cfg, ui_lang=None):
        return ArtifactResult(name="My Title", content="My body.")

    with patch("bibilab.worker._refine_artifact", side_effect=_fake_refine) as mock_refine:
        await worker._run_artifact_job(job)

    # _refine_artifact is the new entry point — it must be called once
    # with the section views. The OLD _run_artifact_job path (calling
    # _generate_artifact directly with a combined transcript) would not
    # touch _refine_artifact at all.
    assert mock_refine.call_count == 1
    # The artifact row is created in 'completed' status with the parsed
    # name + content from the LLM response.
    art = await get_artifact("art-1")
    assert art["status"] == "completed"
    assert art["name"] == "My Title"
    # The artifact content file is written.
    content_path = tmp_bibilab_home / "artifacts" / "list-1" / "art-1.md"
    assert content_path.exists()
    assert content_path.read_text() == "My body."


@pytest.mark.asyncio
async def test_run_artifact_job_batch_failure_writes_no_partial_artifact(tmp_bibilab_home):
    """End-to-end: when _refine_artifact raises PipelineError, the job
    ends in 'failed' status, the artifact row is not created, and the
    content file is not written (no partial artifact)."""
    from bibilab.db import create_job, get_artifact, get_job
    from bibilab.pipeline.audio import PipelineError as PE

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1",
        title="T",
        uploader="u",
        source_url="https://x",
        language="en",
        segments=[WhisperSegment(start=0.0, end=1.0, text="hello", speaker="SPK_0")],
        sections=[Section(seg_start=0, seg_end=0, token_count=1, timestamp_start=0.0, timestamp_end=1.0)],
    )

    job_id = await create_job(
        "artifact",
        {
            "list_id": "list-1",
            "artifact_id": "art-fail",
            "type": "study_guide",
            "prompt": "summarize",
            "source_ids": [source_id],
        },
    )
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job = {
        "id": job_id,
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-fail",
                "type": "study_guide",
                "prompt": "summarize",
                "source_ids": [source_id],
            }
        ),
    }

    async def _fake_refine_fail(*, prompt, sections, cfg, ui_lang=None):
        raise PE("simulated LLM failure")

    with patch("bibilab.worker._refine_artifact", side_effect=_fake_refine_fail):
        await worker._run_artifact_job(job)

    # Job is in failed status with the error message.
    job_row = await get_job(job_id)
    assert job_row is not None
    assert job_row["status"] == "failed"
    assert "simulated LLM failure" in (job_row["error"] or "")
    # No artifact row created, no content file written.
    assert await get_artifact("art-fail") is None
    content_path = tmp_bibilab_home / "artifacts" / "list-1" / "art-fail.md"
    assert not content_path.exists()


@pytest.mark.asyncio
async def test_run_artifact_job_no_sections_fails_loud(tmp_bibilab_home):
    """End-to-end: a source with no `sections` rows causes _run_artifact_job
    to fail the job with the documented message."""
    from bibilab.db import create_job, get_job

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # Source without sections.
    source_id = await SourceFactory.build("list-1", video_id="BV1")

    job_id = await create_job(
        "artifact",
        {
            "list_id": "list-1",
            "artifact_id": "art-nosec",
            "type": "study_guide",
            "prompt": "summarize",
            "source_ids": [source_id],
        },
    )
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    job = {
        "id": job_id,
        "meta": json.dumps(
            {
                "list_id": "list-1",
                "artifact_id": "art-nosec",
                "type": "study_guide",
                "prompt": "summarize",
                "source_ids": [source_id],
            }
        ),
    }

    await worker._run_artifact_job(job)

    # Job is failed with the documented message.
    job_row = await get_job(job_id)
    assert job_row is not None
    assert job_row["status"] == "failed"
    assert "no sections; re-ingest required" in (job_row["error"] or "")

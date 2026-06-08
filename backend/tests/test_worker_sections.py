"""Worker pipeline reorder: derive_sections + chunk_by_sections in _stage_process."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bibilab.config import BibilabConfig
from bibilab.pipeline.digest import DigestResult, SectionDigest
from bibilab.pipeline.section import Section
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


def _seg(start, end, text, speaker="S"):
    from bibilab.pipeline.transcribe import WhisperSegment

    return WhisperSegment(start=start, end=end, text=text, speaker=speaker)


def _video_meta():
    return SimpleNamespace(
        video_id="BV1x",
        platform="bilibili",
        title="T",
        cover_url=None,
        source_url="https://x",
        duration_seconds=0,
        uploader="u",
    )


def _empty_digest_sections(*args, **kwargs):
    """Bypass the LLM path entirely — return a valid empty (DigestResult, [SectionDigest]) tuple."""
    empty = DigestResult(
        summary="",
        keywords=[],
        series_name=None,
        sequence_number=None,
        season_number=None,
    )
    return empty, [SectionDigest(summary="", keywords=[])]


@pytest.mark.asyncio
async def test_stage_process_returns_sections_for_long_source(tmp_bibilab_home: Path):
    """A long synthetic source: _stage_process must return
    (extraction, sections) with sections spanning the segments."""
    from bibilab.db import bootstrap_db, create_list
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1x")
    # 180 long segments → ≥2 sections
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(180)]
    cfg = BibilabConfig()

    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)

    # Capture chunks passed to embed_chunks
    captured = {}

    def fake_embed_chunks(chunks, src, video_meta, list_id):
        captured["chunks"] = chunks
        captured["source_id"] = src

    with (
        patch("bibilab.worker.digest_sections", side_effect=_empty_digest_sections),
        patch("bibilab.worker.embed_chunks", side_effect=fake_embed_chunks),
    ):
        extraction, sections, _section_digests = await worker._stage_process(
            job_id="job-1",
            job={"id": "job-1", "meta": "{}"},
            sentence_segments=segs,
            source_id=source_id,
            video_meta=_video_meta(),
            list_id="list-1",
            cfg=cfg,
            effective_language="en",
        )

    assert isinstance(sections, list)
    assert len(sections) >= 2
    for s in sections:
        assert isinstance(s, Section)

    # Embed received the re-stamped chunks; verify nesting invariant for the
    # chunks actually passed to embed.
    chunks = captured["chunks"]
    assert len(chunks) >= 1
    for c in chunks:
        containing = [s for s in sections if s.seg_start <= c.seg_start and c.seg_end <= s.seg_end]
        assert len(containing) == 1, f"chunk [{c.seg_start}..{c.seg_end}] not contained in exactly one section"


@pytest.mark.asyncio
async def test_stage_process_short_video_returns_one_section(tmp_bibilab_home: Path):
    """A short source: 1 section, byte-identical chunks to pre-change."""
    from bibilab.db import bootstrap_db, create_list
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1y")
    segs = [_seg(float(i), float(i + 1), f"short {i}.") for i in range(20)]
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)

    captured = {}

    def fake_embed_chunks(chunks, src, video_meta, list_id):
        captured["chunks"] = chunks

    with (
        patch("bibilab.worker.digest_sections", side_effect=_empty_digest_sections),
        patch("bibilab.worker.embed_chunks", side_effect=fake_embed_chunks),
    ):
        extraction, sections, _section_digests = await worker._stage_process(
            job_id="job-2",
            job={"id": "job-2", "meta": "{}"},
            sentence_segments=segs,
            source_id=source_id,
            video_meta=_video_meta(),
            list_id="list-1",
            cfg=cfg,
            effective_language="en",
        )

    assert len(sections) == 1
    assert sections[0].seg_start == 0
    assert sections[0].seg_end == len(segs) - 1
    # Embed received the chunks; sequence_index must be 0..N-1
    chunks = captured["chunks"]
    assert sorted(c.sequence_index for c in chunks) == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Task 8: worker ingest uses digest_sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_one_section_byte_identical(tmp_bibilab_home: Path, mock_call_llm):
    """1-section regression: section_digests[0] mirrors the DigestResult.

    For sources that derive into a single section, the SectionDigest list
    must be byte-identical to the DigestResult's summary/keywords. The
    pre-change `digest()` path produced the same payload; the new
    `digest_sections` path must preserve that contract so chat [N] →
    section citations (Task #453) and rerun paths stay aligned.
    """
    from bibilab.db import bootstrap_db, create_list
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BVbyte")
    # 20 short segments → 1 section (well below SECTION_TARGET_TOKENS=12000).
    segs = [_seg(float(i), float(i + 1), f"short {i}.") for i in range(20)]

    # Mock the LLM to return a known digest JSON (one LLM call: section 1 digest).
    mock_call_llm.return_value = json.dumps(
        {
            "summary": "A short video.",
            "keywords": ["topic1", "topic2"],
            "series_name": "TestSeries",
            "sequence_number": 5,
        }
    )

    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)

    with patch("bibilab.worker.embed_chunks", side_effect=lambda *a, **kw: None):
        result = await worker._stage_process(
            job_id="job-byte",
            job={"id": "job-byte", "meta": "{}"},
            sentence_segments=segs,
            source_id=source_id,
            video_meta=_video_meta(),
            list_id="list-1",
            cfg=cfg,
            effective_language="en",
        )

    # 3-tuple: (extraction, sections, section_digests).
    assert result is not None
    extraction, sections, section_digests = result
    # 1 LLM call (digest() for section 1); no refine.
    assert mock_call_llm.call_count == 1
    # 1 section → 1 SectionDigest, mirroring the extraction exactly.
    assert len(sections) == 1
    assert len(section_digests) == 1
    assert isinstance(section_digests[0], SectionDigest)
    assert section_digests[0].summary == extraction.summary
    assert section_digests[0].keywords == extraction.keywords
    # Facets live on the extraction (digest runs once, on section 1).
    assert extraction.series_name == "TestSeries"
    assert extraction.sequence_number == 5


@pytest.mark.asyncio
async def test_ingest_n_sections_writes_summaries_atomically(tmp_bibilab_home: Path, mock_call_llm):
    """N>1 sections: digest_sections is called; section rows get populated.

    End-to-end through the worker stage:
      1. digest_sections runs 1 digest + (N-1) refines (mock_call_llm captures them).
      2. The returned section_digests length matches the derived section count.
      3. write_source_with_segments persists source row mirroring section[0] AND
         section rows with non-NULL summary/keywords, in one transaction.
    """
    from bibilab.db import (
        bootstrap_db,
        create_list,
        get_sections,
        get_source,
        write_source_with_segments,
    )
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # 300 long segments → ≥ 2 sections at SECTION_TARGET_TOKENS=12000.
    # Each segment is 100 tokens (100 words * 1 tok/word); 300 × 100 = 30K tokens.
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(300)]
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)

    # Pre-compute the section count by deriving against a known input.
    from bibilab.pipeline.section import derive_sections

    n_sections = len(derive_sections(segs))
    assert n_sections >= 2, f"expected >= 2 sections, got {n_sections}"

    # Build the LLM response sequence: 1 digest + (N-1) refines.
    digest_response = json.dumps(
        {
            "summary": "Sum 0",
            "keywords": ["k0"],
            "series_name": "S0",
            "sequence_number": 1,
        }
    )
    refine_responses = [json.dumps({"summary": f"Sum {i}", "keywords": [f"k{i}"]}) for i in range(1, n_sections)]
    mock_call_llm.side_effect = [digest_response] + refine_responses

    # Step 1: run _stage_process to derive sections + call digest_sections.
    with patch("bibilab.worker.embed_chunks", side_effect=lambda *a, **kw: None):
        result = await worker._stage_process(
            job_id="job-n",
            job={"id": "job-n", "meta": "{}"},
            sentence_segments=segs,
            source_id="src-n",
            video_meta=_video_meta(),
            list_id="list-1",
            cfg=cfg,
            effective_language="en",
        )

    assert result is not None
    extraction, sections, section_digests = result
    # 1 LLM call per section (1 digest + N-1 refines).
    assert mock_call_llm.call_count == n_sections
    # All N section digests present, in order.
    assert len(section_digests) == n_sections
    assert [sd.summary for sd in section_digests] == [f"Sum {i}" for i in range(n_sections)]
    # The DigestResult carries the section-1 digest (facets extracted once).
    assert extraction.summary == "Sum 0"
    assert extraction.keywords == ["k0"]
    assert extraction.series_name == "S0"
    assert extraction.sequence_number == 1

    # Step 2: persist the result through write_source_with_segments. Verify
    # the source row mirrors section[0] AND the section rows are populated
    # atomically (no NULL summary/keywords on a successful write).
    source_id = await SourceFactory.build("list-1", video_id="BVpersist")
    await write_source_with_segments(
        segments=segs,
        sections=sections,
        section_digests=section_digests,
        source_id=source_id,
        video_id="BVpersist",
        platform="bilibili",
        list_id="list-1",
        title="Persist Test",
        summary=extraction.summary,
        keywords=extraction.keywords,
        cover_url="",
        source_url="https://x",
        duration_seconds=0,
        uploader="u",
        language="en",
        whisper_model=cfg.transcription.model,
        ai_model=cfg.ai.model,
        settings_snapshot=cfg.model_dump(),
        series_name=extraction.series_name,
        sequence_number=extraction.sequence_number,
        season_number=extraction.season_number,
    )

    # Source row: summary/keywords mirror section_digests[0].
    src = await get_source(source_id)
    assert src["summary"] == section_digests[0].summary
    assert json.loads(src["keywords"]) == section_digests[0].keywords
    # Facets populated from the DigestResult (the digest() call).
    assert src["series_name"] == "S0"
    assert src["sequence_number"] == 1

    # Section rows: summary/keywords populated, in seq order, all N rows.
    rows = await get_sections(source_id)
    assert len(rows) == n_sections
    for i, row in enumerate(rows):
        assert row["summary"] == f"Sum {i}"
        assert json.loads(row["keywords"]) == [f"k{i}"]

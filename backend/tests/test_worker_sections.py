"""Worker pipeline reorder: derive_sections + chunk_by_sections in _stage_process."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from bibilab.config import BibilabConfig
from bibilab.pipeline.digest import DigestResult, SectionDigest
from bibilab.pipeline.section import Section
from tests.factories import SectionedSourceFactory, SourceFactory

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


async def _run_digest_rerun(worker, source_id: str, list_id: str, title: str) -> str:
    """Create a digest (rerun) job for source_id and run it. Returns the job id."""
    from bibilab.db import create_job

    meta = {"source_id": source_id, "list_id": list_id, "source_title": title}
    job_id = await create_job("digest", meta)
    await worker._run_digest_job({"id": job_id, "type": "digest", "meta": json.dumps(meta)})
    return job_id


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
# Worker ingest uses digest_sections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_one_section_byte_identical(tmp_bibilab_home: Path, mock_call_llm):
    """1-section regression: section_digests[0] mirrors the DigestResult.

    For sources that derive into a single section, the SectionDigest list
    must be byte-identical to the DigestResult's summary/keywords. The
    pre-change `digest()` path produced the same payload; the new
    `digest_sections` path must preserve that contract so downstream
    citation and rerun paths stay aligned.
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
async def test_ingest_n_sections_produces_ordered_section_digests(tmp_bibilab_home: Path, mock_call_llm):
    """N>1 sections: _stage_process runs digest_sections (1 digest + N-1
    refines) and returns section_digests in seq order; the DigestResult
    carries the section-1 facets. (Atomic persistence of these rows is
    covered by test_db's section_digests write tests.)
    """
    from bibilab.db import bootstrap_db, create_list
    from bibilab.pipeline.section import derive_sections
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # 300 long segments → ≥ 2 sections at SECTION_TARGET_TOKENS=12000.
    # Each segment is 100 tokens (100 words * 1 tok/word); 300 × 100 = 30K tokens.
    segs = [_seg(float(i), float(i + 1), ("word " * 100).strip() + ".") for i in range(300)]
    cfg = BibilabConfig()
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)

    n_sections = len(derive_sections(segs))
    assert n_sections >= 2, f"expected >= 2 sections, got {n_sections}"

    # 1 digest + (N-1) refines.
    digest_response = json.dumps({"summary": "Sum 0", "keywords": ["k0"], "series_name": "S0", "sequence_number": 1})
    refine_responses = [json.dumps({"summary": f"Sum {i}", "keywords": [f"k{i}"]}) for i in range(1, n_sections)]
    mock_call_llm.side_effect = [digest_response] + refine_responses

    with patch("bibilab.worker.embed_chunks", side_effect=lambda *a, **kw: None):
        result = await worker._stage_process(
            job={"id": "job-n", "meta": "{}"},
            sentence_segments=segs,
            source_id="src-n",
            video_meta=_video_meta(),
            list_id="list-1",
            cfg=cfg,
            effective_language="en",
        )

    assert result is not None
    extraction, _sections, section_digests = result
    # 1 LLM call per section (1 digest + N-1 refines), digests in seq order.
    assert mock_call_llm.call_count == n_sections
    assert [sd.summary for sd in section_digests] == [f"Sum {i}" for i in range(n_sections)]
    # The DigestResult carries the section-1 digest (facets extracted once).
    assert extraction.summary == "Sum 0"
    assert extraction.keywords == ["k0"]
    assert extraction.series_name == "S0"
    assert extraction.sequence_number == 1


# ---------------------------------------------------------------------------
# Worker rerun is section-level
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_updates_section_rows(tmp_bibilab_home: Path, mock_call_llm):
    """Rerun on a sectioned source: 1 digest + (N-1) refines, then
    update_section_summaries populates all section rows. Sections are the
    sole digest store (the source carries facets only)."""
    from bibilab.db import bootstrap_db, create_list, get_job, get_sections
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-rerun", "Rerun Test", "2025-01-01T00:00:00Z")
    source_id = "src-rerun-1"
    # 30 segs → 3 sections of 10, each with a prior valid summary.
    await SectionedSourceFactory.build(
        "list-rerun",
        source_id=source_id,
        video_id="BVrerun1",
        title="Rerun Test",
        section_digests=[SectionDigest(summary=f"OLD {i}", keywords=[f"old{i}"]) for i in range(3)],
    )

    # 1 digest + 2 refines.
    mock_call_llm.side_effect = [
        json.dumps({"summary": "NEW 0", "keywords": ["new0"]}),
        json.dumps({"summary": "NEW 1", "keywords": ["new1"]}),
        json.dumps({"summary": "NEW 2", "keywords": ["new2"]}),
    ]

    worker = WorkerLoop(home=tmp_bibilab_home, config=BibilabConfig(), adapter=None)
    job_id = await _run_digest_rerun(worker, source_id, "list-rerun", "Rerun Test")

    assert dict(await get_job(job_id))["status"] == "done"
    # All 3 section rows updated, in seq order.
    rows = await get_sections(source_id)
    assert [r["summary"] for r in rows] == ["NEW 0", "NEW 1", "NEW 2"]


@pytest.mark.asyncio
async def test_rerun_refine_failure_preserves_prior_valid_summaries(tmp_bibilab_home: Path, mock_call_llm):
    """Section 2's refine exhausts retries → job fails; section rows retain
    their prior valid summaries (no NULL window)."""
    from bibilab.db import bootstrap_db, create_list, get_job, get_sections
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-rerun-fail", "Rerun Fail", "2025-01-01T00:00:00Z")
    source_id = "src-rerun-fail"
    await SectionedSourceFactory.build(
        "list-rerun-fail",
        source_id=source_id,
        video_id="BVrerunFail",
        title="Rerun Fail",
        section_digests=[SectionDigest(summary=f"PRIOR {i}", keywords=[f"k{i}"]) for i in range(3)],
    )

    # Section 1 digest succeeds; section 2 refine exhausts retries.
    mock_call_llm.side_effect = [
        json.dumps({"summary": "NEW 0", "keywords": ["n0"]}),
        httpx.HTTPError("transient"),
        httpx.HTTPError("transient"),
        httpx.HTTPError("transient"),
    ]

    worker = WorkerLoop(home=tmp_bibilab_home, config=BibilabConfig(), adapter=None)
    job_id = await _run_digest_rerun(worker, source_id, "list-rerun-fail", "Rerun Fail")

    assert dict(await get_job(job_id))["status"] == "failed"
    # Section rows still have prior valid summaries (no NULL window).
    rows = await get_sections(source_id)
    assert [r["summary"] for r in rows] == ["PRIOR 0", "PRIOR 1", "PRIOR 2"]


@pytest.mark.asyncio
async def test_rerun_legacy_source_without_sections_fails_loud(tmp_bibilab_home: Path, mock_call_llm):
    """Source has transcript segments but 0 section rows. Rerun must fail
    loud with a re-ingest pointer."""
    from bibilab.db import bootstrap_db, create_list, get_job, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-legacy", "Legacy", "2025-01-01T00:00:00Z")
    source_id = await SourceFactory.build("list-legacy", video_id="BVlegacy", title="Legacy", whisper_model="base")
    # Has transcript segments but no section rows.
    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=5.0, text="legacy transcript", speaker=None)]
    )

    worker = WorkerLoop(home=tmp_bibilab_home, config=BibilabConfig(), adapter=None)
    job_id = await _run_digest_rerun(worker, source_id, "list-legacy", "Legacy")

    row = dict(await get_job(job_id))
    assert row["status"] == "failed"
    assert "no sections" in row["error"]

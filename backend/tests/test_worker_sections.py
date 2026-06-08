"""Worker pipeline reorder: derive_sections + chunk_by_sections in _stage_process."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bibilab.config import BibilabConfig
from bibilab.pipeline.digest import DigestResult
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


def _empty_digest(*args, **kwargs):
    """Bypass the LLM path entirely — return a valid empty DigestResult."""
    return DigestResult(
        summary="",
        keywords=[],
        series_name=None,
        sequence_number=None,
        season_number=None,
    )


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
        patch("bibilab.worker.digest", side_effect=_empty_digest),
        patch("bibilab.worker.embed_chunks", side_effect=fake_embed_chunks),
    ):
        extraction, sections = await worker._stage_process(
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
        patch("bibilab.worker.digest", side_effect=_empty_digest),
        patch("bibilab.worker.embed_chunks", side_effect=fake_embed_chunks),
    ):
        extraction, sections = await worker._stage_process(
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

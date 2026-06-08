"""One-shot backfill script: derive_sections for every existing source and
write section rows. Idempotent (DELETE+INSERT per source). No re-embed
(all current sources are short → 1 section → existing chunks already nest)."""

from pathlib import Path

import pytest

from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_backfill_writes_sections_for_existing_source(tmp_bibilab_home: Path, capsys):
    from scripts import backfill_sections

    from bibilab.db import bootstrap_db, create_list, get_sections, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1bf")

    # 30 short segments → 1 section per the algorithm
    segs = [WhisperSegment(start=float(i), end=float(i + 1), text=f"sentence {i}.", speaker="S") for i in range(30)]
    await write_transcript_segments(source_id, segs)

    # Pre-condition: no section rows yet
    assert (await get_sections(source_id)) == []

    # Run backfill
    rc = await backfill_sections.run(dry_run=False)
    assert rc == 0

    rows = await get_sections(source_id)
    assert len(rows) == 1
    assert rows[0]["seg_start"] == 0
    assert rows[0]["seg_end"] == 29

    captured = capsys.readouterr()
    assert "1 source(s)" in captured.out or "1 source" in captured.out


@pytest.mark.asyncio
async def test_backfill_dry_run_prints_no_writes(tmp_bibilab_home: Path, capsys):
    from scripts import backfill_sections

    from bibilab.db import bootstrap_db, create_list, get_sections, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests.factories import SourceFactory

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1bfd")
    await write_transcript_segments(
        source_id,
        [
            WhisperSegment(start=0.0, end=1.0, text="hi.", speaker="S"),
        ],
    )

    rc = await backfill_sections.run(dry_run=True)
    assert rc == 0
    assert (await get_sections(source_id)) == []
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out or "dry run" in captured.out.lower()

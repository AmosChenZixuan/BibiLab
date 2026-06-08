"""Tests for the backfill_sections one-shot script.

The script is deleted after the verified run; the test file stays as a
regression guard for the migration logic. The live-DB run is operational,
not a test.
"""

import json

import pytest

from bibilab.db import bootstrap_db, create_list, get_db, get_sections
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


async def test_backfill_populates_section_row_per_source(tmp_bibilab_home):
    from bibilab.scripts.backfill_sections import run_backfill

    await bootstrap_db()
    list_id = "list-backfill-1"
    await create_list(list_id, "Backfill", "2026-01-01T00:00:00")

    # Set up N=3 sources with no section rows, pre-populated summary/keywords.
    source_ids = []
    for i in range(3):
        sid = await SourceFactory.build(
            list_id=list_id,
            video_id=f"BV-bf-{i}",
            title=f"S{i}",
            summary=f"sum {i}",
            keywords=[f"k{i}"],
        )
        source_ids.append(sid)
        # Insert segments so derive_sections has data.
        async with get_db() as db:
            await db.executemany(
                "INSERT INTO transcript_segments (source_id, seq, start_s, end_s, speaker, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(sid, j, float(j), float(j + 1), None, f"s{j}.") for j in range(5)],
            )
            await db.commit()

    # Run backfill.
    await run_backfill(fail_on_mismatch=True)

    # Post-conditions: each source has exactly 1 section with summary/keywords mirrored.
    for sid, i in zip(source_ids, range(3)):
        sec_rows = await get_sections(sid)
        assert len(sec_rows) == 1, f"expected 1 section for {sid}, got {len(sec_rows)}"
        row = sec_rows[0]
        assert row["summary"] == f"sum {i}"
        assert json.loads(row["keywords"]) == [f"k{i}"]


async def test_backfill_skips_source_with_existing_sections(tmp_bibilab_home):
    from bibilab.scripts.backfill_sections import run_backfill

    await bootstrap_db()
    list_id = "list-backfill-2"
    await create_list(list_id, "Backfill skip", "2026-01-01T00:00:00")
    s_id = await SourceFactory.build(list_id=list_id, summary="x", keywords=[])

    # Insert >1 section rows directly (mimics a post-#452 ingest). The script
    # must skip this source — the backfill is for pre-#452 sources only.
    async with get_db() as db:
        await db.executemany(
            "INSERT INTO sections (source_id, seq, seg_start, seg_end, "
            "token_count, timestamp_start, timestamp_end, summary, keywords) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            [
                (s_id, 0, 0, 9, 100, 0.0, 10.0),
                (s_id, 1, 10, 19, 100, 10.0, 20.0),
                (s_id, 2, 20, 29, 100, 20.0, 30.0),
            ],
        )
        await db.commit()

    await run_backfill()

    # Section rows untouched (still NULL summary/keywords from the pre-#452 write).
    rows = await get_sections(s_id)
    assert len(rows) == 3
    for r in rows:
        assert r["summary"] is None
        assert r["keywords"] is None

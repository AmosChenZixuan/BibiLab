"""Sections-table readers: get_section_ranges."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bibilab.db import bootstrap_db, create_list, get_section_ranges, write_source_with_segments
from bibilab.pipeline.section import Section
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        yield tmp_path


@pytest.mark.asyncio
async def test_get_section_ranges_returns_ordered_ranges(tmp_bibilab_home: Path):
    """3 sections written via the factory: get_section_ranges returns them
    in (seq, seg_start, seg_end, token_count, timestamp_start, timestamp_end)
    order with the right values."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1x")
    # SourceFactory.build does not accept sections=...; write sections in a
    # second pass via the production atomic path.
    sections = [
        Section(seg_start=0, seg_end=3, token_count=100, timestamp_start=0.0, timestamp_end=60.0),
        Section(seg_start=4, seg_end=7, token_count=110, timestamp_start=60.0, timestamp_end=120.0),
        Section(seg_start=8, seg_end=9, token_count=80, timestamp_start=120.0, timestamp_end=150.0),
    ]
    await write_source_with_segments(
        segments=[],
        sections=sections,
        source_id=source_id,
        video_id="BV1x",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="",
        keywords=[],
        cover_url=None,
        source_url="https://x",
        duration_seconds=0,
        uploader="u",
        language="en",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        settings_snapshot={},
    )

    rows = await get_section_ranges(source_id)

    assert len(rows) == 3
    assert [
        (r["seq"], r["seg_start"], r["seg_end"], r["token_count"], r["timestamp_start"], r["timestamp_end"])
        for r in rows
    ] == [
        (0, 0, 3, 100, 0.0, 60.0),
        (1, 4, 7, 110, 60.0, 120.0),
        (2, 8, 9, 80, 120.0, 150.0),
    ]


@pytest.mark.asyncio
async def test_get_section_ranges_empty_when_no_sections(tmp_bibilab_home: Path):
    """A source with no section rows returns [] (NOT a hard error at this layer)."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    # SourceFactory.build without sections=... → no section rows written
    source_id = await SourceFactory.build("list-1", video_id="BV1x")

    rows = await get_section_ranges(source_id)

    assert rows == []

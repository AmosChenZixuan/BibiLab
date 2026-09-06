"""Sections-table readers: get_section_ranges."""

from pathlib import Path

import pytest

from bibilab.db import bootstrap_db, count_sections_for_sources, create_list, get_section_ranges
from bibilab.pipeline.section import Section
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_get_section_ranges_returns_ordered_ranges(tmp_bibilab_home: Path):
    """3 sections written via the factory: get_section_ranges returns them
    in (seq, seg_start, seg_end, token_count, timestamp_start, timestamp_end)
    order with the right values."""
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    sections = [
        Section(seg_start=0, seg_end=3, token_count=100, timestamp_start=0.0, timestamp_end=60.0),
        Section(seg_start=4, seg_end=7, token_count=110, timestamp_start=60.0, timestamp_end=120.0),
        Section(seg_start=8, seg_end=9, token_count=80, timestamp_start=120.0, timestamp_end=150.0),
    ]
    source_id = await SourceFactory.build(
        "list-1",
        video_id="BV1x",
        title="T",
        uploader="u",
        language="en",
        source_url="https://x",
        sections=sections,
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
    source_id = await SourceFactory.build("list-1", video_id="BV1x")

    rows = await get_section_ranges(source_id)

    assert rows == []


@pytest.mark.asyncio
async def test_count_sections_for_sources_sums_and_handles_empty(tmp_bibilab_home: Path):
    """Pins the helper directly: 2 sources × 3 sections = 6; subset 1 source = 3;
    unknown source = 0; empty list short-circuits to 0 without raising.
    """
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    three = [
        Section(seg_start=0, seg_end=3, token_count=100, timestamp_start=0.0, timestamp_end=60.0),
        Section(seg_start=4, seg_end=7, token_count=110, timestamp_start=60.0, timestamp_end=120.0),
        Section(seg_start=8, seg_end=9, token_count=80, timestamp_start=120.0, timestamp_end=150.0),
    ]
    src_a = await SourceFactory.build(
        "list-1", video_id="BVa", title="A", uploader="u", language="en", source_url="https://a", sections=three
    )
    src_b = await SourceFactory.build(
        "list-1", video_id="BVb", title="B", uploader="u", language="en", source_url="https://b", sections=three
    )

    assert await count_sections_for_sources([src_a, src_b]) == 6
    assert await count_sections_for_sources([src_a]) == 3
    assert await count_sections_for_sources(["unknown"]) == 0
    assert await count_sections_for_sources([]) == 0


@pytest.mark.asyncio
async def test_count_sections_for_sources_cache_invalidates_on_rewrite(tmp_bibilab_home: Path):
    """A second write_source_with_segments call replaces the section rows;
    the cache must reflect the new count, not the old one. Same source_id
    so the UPSERT updates the existing source row in place.
    """
    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    three = [
        Section(seg_start=0, seg_end=3, token_count=100, timestamp_start=0.0, timestamp_end=60.0),
        Section(seg_start=4, seg_end=7, token_count=110, timestamp_start=60.0, timestamp_end=120.0),
        Section(seg_start=8, seg_end=9, token_count=80, timestamp_start=120.0, timestamp_end=150.0),
    ]
    five = three + [
        Section(seg_start=10, seg_end=12, token_count=90, timestamp_start=150.0, timestamp_end=180.0),
        Section(seg_start=13, seg_end=15, token_count=95, timestamp_start=180.0, timestamp_end=210.0),
    ]
    src = await SourceFactory.build(
        "list-1", video_id="BV1", title="T", uploader="u", language="en", source_url="https://x", sections=three
    )

    assert await count_sections_for_sources([src]) == 3

    await SourceFactory.build(
        "list-1",
        source_id=src,
        video_id="BV1",
        title="T",
        uploader="u",
        language="en",
        source_url="https://x",
        sections=five,
    )

    assert await count_sections_for_sources([src]) == 5

import pytest

from bibilab.db import bootstrap_db, count_sources, create_list, get_db


async def _seed(sources: list[dict]):
    """Insert a list and sources. Each source dict may have id, duration_seconds, language, title."""
    list_id = "list-seed-1"
    await create_list(list_id, "Test List", "2026-01-01T00:00:00")
    async with get_db() as db:
        for src in sources:
            await db.execute(
                """INSERT INTO sources
                (id, video_id, platform, list_id, title, source_url, duration_seconds, uploader, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    src["id"],
                    "vid-" + src["id"],
                    "bilibili",
                    list_id,
                    src.get("title", "Test Video"),
                    "https://example.com/" + src["id"],
                    src.get("duration_seconds", 0),
                    "TestUploader",
                    src.get("language"),
                ),
            )
        await db.commit()


@pytest.mark.asyncio
async def test_count_sources_returns_count(tmp_bibilab_home):
    await bootstrap_db()
    await _seed(
        [
            {"id": "src-a"},
            {"id": "src-b"},
            {"id": "src-c"},
        ]
    )
    result = await count_sources(["src-a", "src-b", "src-c"])
    assert result == 3


@pytest.mark.asyncio
async def test_count_sources_subset(tmp_bibilab_home):
    await bootstrap_db()
    await _seed(
        [
            {"id": "src-a"},
            {"id": "src-b"},
            {"id": "src-c"},
        ]
    )
    result = await count_sources(["src-a", "src-b"])
    assert result == 2


@pytest.mark.asyncio
async def test_count_sources_empty(tmp_bibilab_home):
    await bootstrap_db()
    await _seed(
        [
            {"id": "src-a"},
            {"id": "src-b"},
            {"id": "src-c"},
        ]
    )
    result = await count_sources([])
    assert result == 0


@pytest.mark.asyncio
async def test_longest_source_returns_max(tmp_bibilab_home):
    from bibilab.db import longest_source

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "title": "Short", "duration_seconds": 60},
            {"id": "s2", "title": "Medium", "duration_seconds": 600},
            {"id": "s3", "title": "Long", "duration_seconds": 3600},
        ]
    )

    result = await longest_source(["s1", "s2", "s3"])

    assert result == {"title": "Long", "duration_seconds": 3600}


@pytest.mark.asyncio
async def test_longest_source_subset_excludes_unselected(tmp_bibilab_home):
    from bibilab.db import longest_source

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "title": "Short", "duration_seconds": 60},
            {"id": "s2", "title": "Medium", "duration_seconds": 600},
            {"id": "s3", "title": "Long", "duration_seconds": 3600},
        ]
    )

    result = await longest_source(["s1", "s2"])

    assert result == {"title": "Medium", "duration_seconds": 600}


@pytest.mark.asyncio
async def test_longest_source_empty(tmp_bibilab_home):
    from bibilab.db import longest_source

    await bootstrap_db()
    assert await longest_source([]) is None


@pytest.mark.asyncio
async def test_longest_source_no_matches(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, longest_source

    await bootstrap_db()
    assert await longest_source(["nonexistent"]) is None

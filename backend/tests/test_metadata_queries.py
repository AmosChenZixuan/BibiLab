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


@pytest.mark.asyncio
async def test_longest_source_tiebreaker_lower_id_wins(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, longest_source

    await bootstrap_db()
    await _seed(
        [
            {"id": "s2", "title": "Second", "duration_seconds": 600},
            {"id": "s1", "title": "First", "duration_seconds": 600},
        ]
    )

    result = await longest_source(["s1", "s2"])

    # id ASC tiebreaker: s1 < s2, so s1 wins
    assert result == {"title": "First", "duration_seconds": 600}


@pytest.mark.asyncio
async def test_language_breakdown_groups_by_language(tmp_bibilab_home):
    from bibilab.db import language_breakdown

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "language": "zh"},
            {"id": "s2", "language": "zh"},
            {"id": "s3", "language": "en"},
        ]
    )

    result = await language_breakdown(["s1", "s2", "s3"])

    assert result == {"zh": 2, "en": 1}


@pytest.mark.asyncio
async def test_language_breakdown_null_grouped_as_unknown(tmp_bibilab_home):
    from bibilab.db import language_breakdown

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "language": "zh"},
            {"id": "s2", "language": None},
            {"id": "s3", "language": None},
        ]
    )

    result = await language_breakdown(["s1", "s2", "s3"])

    assert result == {"zh": 1, "unknown": 2}


@pytest.mark.asyncio
async def test_language_breakdown_subset(tmp_bibilab_home):
    from bibilab.db import language_breakdown

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "language": "zh"},
            {"id": "s2", "language": "en"},
            {"id": "s3", "language": "en"},
        ]
    )

    result = await language_breakdown(["s1", "s2"])

    assert result == {"zh": 1, "en": 1}


@pytest.mark.asyncio
async def test_language_breakdown_empty(tmp_bibilab_home):
    from bibilab.db import language_breakdown

    await bootstrap_db()
    assert await language_breakdown([]) == {}


@pytest.mark.asyncio
async def test_get_titles_returns_id_title_pairs(tmp_bibilab_home):
    from bibilab.db import get_titles

    await bootstrap_db()
    await _seed(
        [
            {"id": "s-b", "title": "第八集 美食推荐"},
            {"id": "s-a", "title": "第3道菜 做法"},
        ]
    )

    result = await get_titles(["s-a", "s-b"])

    assert result == [
        {"source_id": "s-a", "title": "第3道菜 做法"},
        {"source_id": "s-b", "title": "第八集 美食推荐"},
    ]


@pytest.mark.asyncio
async def test_get_titles_subset(tmp_bibilab_home):
    from bibilab.db import get_titles

    await bootstrap_db()
    await _seed(
        [
            {"id": "s1", "title": "T1"},
            {"id": "s2", "title": "T2"},
            {"id": "s3", "title": "T3"},
        ]
    )

    result = await get_titles(["s1", "s3"])

    assert result == [
        {"source_id": "s1", "title": "T1"},
        {"source_id": "s3", "title": "T3"},
    ]


@pytest.mark.asyncio
async def test_get_titles_empty(tmp_bibilab_home):
    from bibilab.db import get_titles

    await bootstrap_db()
    assert await get_titles([]) == []

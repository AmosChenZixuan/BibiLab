from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_query_chunks_empty_source_ids(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    result = await query_chunks("test query", [], cfg)

    assert result == []


@pytest.mark.asyncio
async def test_query_chunks_no_video_ids_found(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    with patch(
        "bibilab.pipeline.embed.get_video_ids_for_sources",
        new_callable=AsyncMock,
    ) as mock_map:
        mock_map.return_value = {}

        result = await query_chunks("test query", ["src-uuid-1", "src-uuid-2"], cfg)

        assert result == []
        mock_map.assert_called_once_with(["src-uuid-1", "src-uuid-2"])


@pytest.mark.asyncio
async def test_query_chunks_filters_by_video_id(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [
            [
                {
                    "video_id": "bvid123",
                    "video_title": "Test Video",
                    "timestamp_start": 0.0,
                    "timestamp_end": 10.0,
                }
            ]
        ],
        "distances": [[0.1]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch(
            "bibilab.pipeline.embed._get_collection",
            return_value=mock_collection,
        ) as mock_get_col,
    ):
        mock_map.return_value = {"src-uuid-1": "bvid123"}

        await query_chunks("test query", ["src-uuid-1"], cfg, top_k=5)

        mock_get_col.assert_called_once_with(cfg)
        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["query_texts"] == ["test query"]
        assert call_kwargs["n_results"] == 5
        assert call_kwargs["where"] == {"video_id": {"$in": ["bvid123"]}}


@pytest.mark.asyncio
async def test_query_chunks_applies_relevance_floor(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3", "c4"]],
        "metadatas": [
            [
                {"video_id": "bvid1", "video_title": "V1", "timestamp_start": 0.0, "timestamp_end": 5.0},
                {"video_id": "bvid2", "video_title": "V2", "timestamp_start": 5.0, "timestamp_end": 10.0},
                {"video_id": "bvid3", "video_title": "V3", "timestamp_start": 10.0, "timestamp_end": 15.0},
                {"video_id": "bvid4", "video_title": "V4", "timestamp_start": 15.0, "timestamp_end": 20.0},
            ]
        ],
        "distances": [[0.15, 0.25, 0.28, 0.4]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "bvid1", "s2": "bvid2", "s3": "bvid3", "s4": "bvid4"}

        result = await query_chunks("test query", ["s1", "s2", "s3", "s4"], cfg)

        assert len(result) == 3
        distances = [c.distance for c in result]
        assert distances == [0.15, 0.25, 0.28]


@pytest.mark.asyncio
async def test_query_chunks_returns_chunk_metadata(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["some transcript content here"]],
        "metadatas": [
            [
                {
                    "video_id": "bvid999",
                    "video_title": "My Video Title",
                    "timestamp_start": 12.5,
                    "timestamp_end": 27.3,
                }
            ]
        ],
        "distances": [[0.15]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"src-abc": "bvid999"}

        result = await query_chunks("test query", ["src-abc"], cfg)

        assert len(result) == 1
        chunk = result[0]
        assert chunk.content == "some transcript content here"
        assert chunk.video_title == "My Video Title"
        assert chunk.timestamp_start == 12.5
        assert chunk.timestamp_end == 27.3
        assert chunk.video_id == "bvid999"
        assert chunk.distance == 0.15


@pytest.mark.asyncio
async def test_query_chunks_sorts_by_distance_ascending(tmp_bibilab_home):
    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["c1", "c2", "c3"]],
        "metadatas": [
            [
                {"video_id": "v1", "video_title": "V1", "timestamp_start": 0.0, "timestamp_end": 1.0},
                {"video_id": "v2", "video_title": "V2", "timestamp_start": 0.0, "timestamp_end": 1.0},
                {"video_id": "v3", "video_title": "V3", "timestamp_start": 0.0, "timestamp_end": 1.0},
            ]
        ],
        "distances": [[0.1, 0.2, 0.25]],
    }

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
    ):
        mock_map.return_value = {"s1": "v1", "s2": "v2", "s3": "v3"}

        result = await query_chunks("test query", ["s1", "s2", "s3"], cfg)

        assert len(result) == 3
        assert result[0].distance == 0.1
        assert result[1].distance == 0.2
        assert result[2].distance == 0.25


@pytest.mark.asyncio
async def test_query_chunks_chroma_error_returns_empty(tmp_bibilab_home, caplog):
    import logging

    from bibilab.config import BibilabConfig, RagConfig
    from bibilab.pipeline.embed import query_chunks

    cfg = BibilabConfig(rag=RagConfig(max_distance=0.3))

    mock_collection = MagicMock()
    mock_collection.query.side_effect = Exception("ChromaDB connection error")

    with (
        patch(
            "bibilab.pipeline.embed.get_video_ids_for_sources",
            new_callable=AsyncMock,
        ) as mock_map,
        patch("bibilab.pipeline.embed._get_collection", return_value=mock_collection),
        caplog.at_level(logging.WARNING),
    ):
        mock_map.return_value = {"src-1": "bvid1"}

        result = await query_chunks("test query", ["src-1"], cfg)

        assert result == []
        assert "ChromaDB query failed" in caplog.text


@pytest.mark.asyncio
async def test_get_video_ids_for_sources(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_db, get_video_ids_for_sources

    await bootstrap_db()

    async with get_db() as db:
        await db.execute(
            "INSERT INTO lists (id, name, created_at) VALUES (?, ?, ?)",
            ("list-1", "Test List", "2026-01-01T00:00:00"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-a", "bv123", "bilibili", "list-1", "Test", "https://example.com"),
        )
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title, source_url) VALUES (?, ?, ?, ?, ?, ?)",
            ("src-b", "bv456", "bilibili", "list-1", "Test2", "https://example.com"),
        )
        await db.commit()

    result = await get_video_ids_for_sources(["src-a", "src-b", "src-c"])

    assert result == {"src-a": "bv123", "src-b": "bv456"}


@pytest.mark.asyncio
async def test_get_video_ids_for_sources_empty(tmp_bibilab_home):
    from bibilab.db import get_video_ids_for_sources

    result = await get_video_ids_for_sources([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_video_ids_for_sources_no_matches(tmp_bibilab_home):
    from bibilab.db import bootstrap_db, get_video_ids_for_sources

    await bootstrap_db()

    result = await get_video_ids_for_sources(["nonexistent-src"])

    assert result == {}

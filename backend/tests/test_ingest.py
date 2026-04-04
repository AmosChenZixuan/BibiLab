from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _make_video_info(bvid="BV1abc123", title="Test Video"):
    return {
        "id": bvid,
        "title": title,
        "webpage_url": f"https://www.bilibili.com/video/{bvid}",
        "thumbnail": "https://example.com/cover.jpg",
        "duration": 3600,
        "uploader": "TestUser",
        "ext": "mp4",
    }


@pytest.fixture()
def mock_ydl_single():
    info = _make_video_info()
    mock = MagicMock()
    mock.__enter__ = lambda s: mock
    mock.__exit__ = MagicMock(return_value=False)
    mock.extract_info = MagicMock(return_value=info)
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
        yield info


@pytest.mark.asyncio
async def test_ingest_single_video(client: httpx.AsyncClient, mock_ydl_single):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_dedup(client: httpx.AsyncClient, mock_ydl_single, tmp_bibilab_home: Path):
    """Submitting the same video twice should skip on second attempt."""
    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path=tmp_bibilab_home / "notes" / "BV1abc123.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = await client.post(
        "/ingest/url",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["queued"] == []
    assert "BV1abc123" in data["skipped"]


@pytest.mark.asyncio
async def test_ingest_rerun_bypasses_dedup(
    client: httpx.AsyncClient, mock_ydl_single, tmp_bibilab_home: Path
):
    """?rerun=true should queue even if source already exists."""
    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path=tmp_bibilab_home / "notes" / "BV1abc123.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = await client.post(
        "/ingest/url?rerun=true",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_unknown_list(client: httpx.AsyncClient, mock_ydl_single):
    resp = await client.post(
        "/ingest/url",
        json={"list_id": "nonexistent", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ingest_rerun_endpoint_removed(client: httpx.AsyncClient):
    """The old /ingest/rerun/{video_id} endpoint must no longer exist."""
    resp = await client.post("/ingest/rerun/BV1abc123")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_write_source_stores_relative_paths(tmp_bibilab_home: Path):
    """After write_source, note_path and transcript_path are stored as relative paths."""
    from bibilab.db import bootstrap_db, create_list, get_source, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    # Call write_source (the DB write happens here)
    await write_source(
        video_id="BV1relative",
        platform="bilibili",
        list_id="list-1",
        title="Test",
        summary="Test summary.",
        note_path=tmp_bibilab_home / "notes" / "BV1relative.md",
        transcript_path=tmp_bibilab_home / "transcripts" / "BV1relative.txt",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Read back and verify paths are stored as relative
    row = await get_source("BV1relative")
    assert row["note_path"] == "notes/BV1relative.md"
    assert row["transcript_path"] == "transcripts/BV1relative.txt"

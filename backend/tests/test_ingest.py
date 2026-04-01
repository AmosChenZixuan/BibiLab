from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            with patch("locus.main.locus_home", return_value=tmp_path):
                yield tmp_path


@pytest.fixture()
def client(tmp_locus_home: Path):
    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


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
    with patch("locus.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
        yield info


def test_ingest_single_video(client: TestClient, mock_ydl_single):
    list_id = client.post("/lists", json={"name": "Test"}).json()["id"]
    resp = client.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_dedup(client: TestClient, mock_ydl_single, tmp_locus_home: Path):
    """Submitting the same video twice should skip on second attempt."""
    from locus.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path="/tmp/note.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = client.post(
        "/ingest/url",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["queued"] == []
    assert "BV1abc123" in data["skipped"]


@pytest.mark.asyncio
async def test_ingest_rerun_bypasses_dedup(
    client: TestClient, mock_ydl_single, tmp_locus_home: Path
):
    """?rerun=true should queue even if source already exists."""
    from locus.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await write_source(
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path="/tmp/note.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = client.post(
        "/ingest/url?rerun=true",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


def test_ingest_unknown_list(client: TestClient, mock_ydl_single):
    resp = client.post(
        "/ingest/url",
        json={"list_id": "nonexistent", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 404


def test_ingest_rerun_endpoint_removed(client: TestClient):
    """The old /ingest/rerun/{video_id} endpoint must no longer exist."""
    resp = client.post("/ingest/rerun/BV1abc123")
    assert resp.status_code == 404

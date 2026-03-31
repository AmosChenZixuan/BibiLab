"""Tests for lists CRUD and ingest URL endpoint (yt-dlp mocked)."""

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


@pytest.fixture()
def client_with_vault(tmp_locus_home: Path, tmp_path: Path):
    """Client with vault_path configured so ingest doesn't 400."""
    from locus.config import load_config, save_config

    cfg = load_config()
    cfg.obsidian.vault_path = str(tmp_path / "vault")
    save_config(cfg)

    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Lists CRUD
# ---------------------------------------------------------------------------


def test_create_list(client: TestClient):
    resp = client.post("/lists", json={"name": "ML Course"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "ML Course"
    assert "id" in data


def test_create_list_empty_name(client: TestClient):
    resp = client.post("/lists", json={"name": "  "})
    assert resp.status_code == 422


def test_create_list_duplicate(client: TestClient):
    client.post("/lists", json={"name": "Physics"})
    resp = client.post("/lists", json={"name": "Physics"})
    assert resp.status_code == 409


def test_get_lists_empty(client: TestClient):
    assert client.get("/lists").json() == []


def test_get_lists(client: TestClient):
    client.post("/lists", json={"name": "List A"})
    client.post("/lists", json={"name": "List B"})
    names = [r["name"] for r in client.get("/lists").json()]
    assert names == ["List A", "List B"]


# ---------------------------------------------------------------------------
# Ingest URL
# ---------------------------------------------------------------------------


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
    """Mock yt-dlp to return a single video."""
    info = _make_video_info()
    mock = MagicMock()
    mock.__enter__ = lambda s: mock
    mock.__exit__ = MagicMock(return_value=False)
    mock.extract_info = MagicMock(return_value=info)
    with patch("locus.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
        yield info


def test_ingest_single_video(client_with_vault: TestClient, mock_ydl_single):
    list_id = client_with_vault.post("/lists", json={"name": "Test"}).json()["id"]
    resp = client_with_vault.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_dedup(
    client_with_vault: TestClient, mock_ydl_single, tmp_locus_home: Path
):
    """Submitting the same video twice should skip on second attempt."""
    from locus.db import bootstrap_db, get_db

    # Seed processing_log to simulate already-processed video
    await bootstrap_db()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO processing_log (video_id, platform, processed_at)
               VALUES (?, ?, datetime('now'))""",
            ("BV1abc123", "bilibili"),
        )
        await db.commit()

    list_id = client_with_vault.post("/lists", json={"name": "Test"}).json()["id"]
    resp = client_with_vault.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["queued"] == []
    assert "BV1abc123" in data["skipped"]


def test_ingest_unknown_list(client_with_vault: TestClient, mock_ydl_single):
    resp = client_with_vault.post(
        "/ingest/url",
        json={
            "list_id": "nonexistent",
            "url": "https://www.bilibili.com/video/BV1abc123",
        },
    )
    assert resp.status_code == 404


def test_ingest_no_vault_configured(client: TestClient, mock_ydl_single):
    list_id = client.post("/lists", json={"name": "Test"}).json()["id"]
    resp = client.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 400
    assert "vault_path" in resp.json()["detail"]

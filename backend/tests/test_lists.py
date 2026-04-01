import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            main_module = importlib.import_module("locus.main")
            with patch.object(main_module, "locus_home", return_value=tmp_path):
                yield tmp_path


@pytest.fixture()
def client(tmp_locus_home: Path):
    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_create_list(client: TestClient):
    resp = client.post("/lists", json={"name": "ML Course"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "ML Course"
    assert "id" in data
    assert "created_at" in data


def test_create_list_empty_name(client: TestClient):
    resp = client.post("/lists", json={"name": "  "})
    assert resp.status_code == 422


def test_create_list_allows_duplicate_names(client: TestClient):
    client.post("/lists", json={"name": "Physics"})
    resp = client.post("/lists", json={"name": "Physics"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "Physics"


def test_get_lists_empty(client: TestClient):
    assert client.get("/lists").json() == []


def test_get_lists(client: TestClient):
    client.post("/lists", json={"name": "List A"})
    client.post("/lists", json={"name": "List B"})
    names = [r["name"] for r in client.get("/lists").json()]
    assert names == ["List A", "List B"]


def test_delete_list(client: TestClient):
    list_id = client.post("/lists", json={"name": "ToDelete"}).json()["id"]
    resp = client.delete(f"/lists/{list_id}")
    assert resp.status_code == 204
    assert client.get("/lists").json() == []


def test_delete_list_not_found(client: TestClient):
    assert client.delete("/lists/nonexistent").status_code == 404


@pytest.mark.asyncio
async def test_delete_list_rejects_active_jobs(client: TestClient, tmp_locus_home: Path):
    from locus.db import create_job

    list_id = client.post("/lists", json={"name": "Active"}).json()["id"]
    await create_job(
        type="video",
        source_url="https://www.bilibili.com/video/BV1active",
        platform="bilibili",
        meta={"video_id": "BV1active", "list_id": list_id},
    )
    resp = client.delete(f"/lists/{list_id}")
    assert resp.status_code == 409
    assert "active jobs" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_list_sources(client: TestClient, tmp_locus_home: Path):
    from locus.db import write_source

    list_id = client.post("/lists", json={"name": "ML"}).json()["id"]
    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id=list_id,
        title="Intro",
        summary="A summary.",
        note_path="/tmp/BV1abc.md",
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = client.get(f"/lists/{list_id}/sources")
    assert resp.status_code == 200
    sources = resp.json()
    assert len(sources) == 1
    assert sources[0]["video_id"] == "BV1abc"
    assert sources[0]["title"] == "Intro"


@pytest.mark.asyncio
async def test_delete_source_from_list(client: TestClient, tmp_locus_home: Path, tmp_path: Path):
    from locus.db import get_source, write_source

    list_id = client.post("/lists", json={"name": "ML"}).json()["id"]
    note_file = tmp_path / "BV1abc.md"
    note_file.write_text("# Note", encoding="utf-8")
    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id=list_id,
        title="Intro",
        summary="S",
        note_path=str(note_file),
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    with patch("locus.routers.lists.clear_embeddings_for_video") as mock_clear:
        resp = client.delete(f"/lists/{list_id}/sources/BV1abc")
    assert resp.status_code == 204
    assert not note_file.exists()
    assert await get_source("BV1abc") is None
    mock_clear.assert_called_once()
    assert mock_clear.call_args[0][0] == "BV1abc"


def test_delete_source_not_found(client: TestClient):
    list_id = client.post("/lists", json={"name": "ML"}).json()["id"]
    assert client.delete(f"/lists/{list_id}/sources/nonexistent").status_code == 404

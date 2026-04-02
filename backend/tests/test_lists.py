from pathlib import Path
from unittest.mock import patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_create_list(client: httpx.AsyncClient):
    resp = await client.post("/lists", json={"name": "ML Course"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "ML Course"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_list_empty_name(client: httpx.AsyncClient):
    resp = await client.post("/lists", json={"name": "  "})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_list_allows_duplicate_names(client: httpx.AsyncClient):
    await client.post("/lists", json={"name": "Physics"})
    resp = await client.post("/lists", json={"name": "Physics"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "Physics"


@pytest.mark.asyncio
async def test_get_lists_empty(client: httpx.AsyncClient):
    assert (await client.get("/lists")).json() == []


@pytest.mark.asyncio
async def test_get_lists(client: httpx.AsyncClient):
    await client.post("/lists", json={"name": "List A"})
    await client.post("/lists", json={"name": "List B"})
    names = [r["name"] for r in (await client.get("/lists")).json()]
    assert names == ["List A", "List B"]


@pytest.mark.asyncio
async def test_rename_list(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Before"})).json()["id"]
    resp = await client.patch(f"/lists/{list_id}", json={"name": "After"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "After"
    names = [r["name"] for r in (await client.get("/lists")).json()]
    assert names == ["After"]


@pytest.mark.asyncio
async def test_delete_list(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "ToDelete"})).json()["id"]
    resp = await client.delete(f"/lists/{list_id}")
    assert resp.status_code == 204
    assert (await client.get("/lists")).json() == []


@pytest.mark.asyncio
async def test_delete_list_not_found(client: httpx.AsyncClient):
    assert (await client.delete("/lists/nonexistent")).status_code == 404


@pytest.mark.asyncio
async def test_delete_list_rejects_active_jobs(client: httpx.AsyncClient, tmp_locus_home: Path):
    from locus.db import create_job

    list_id = (await client.post("/lists", json={"name": "Active"})).json()["id"]
    await create_job(
        type="ingest",
        meta={
            "video_id": "BV1active",
            "list_id": list_id,
            "source_url": "https://www.bilibili.com/video/BV1active",
            "platform": "bilibili",
        },
    )
    resp = await client.delete(f"/lists/{list_id}")
    assert resp.status_code == 409
    assert "active jobs" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_list_sources(client: httpx.AsyncClient, tmp_locus_home: Path):
    from locus.db import write_source

    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
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
    resp = await client.get(f"/lists/{list_id}/sources")
    assert resp.status_code == 200
    sources = resp.json()
    assert len(sources) == 1
    assert sources[0]["video_id"] == "BV1abc"
    assert sources[0]["title"] == "Intro"


@pytest.mark.asyncio
async def test_delete_source_from_list(
    client: httpx.AsyncClient, tmp_locus_home: Path, tmp_path: Path
):
    from locus.db import get_source, write_source

    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
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
        resp = await client.delete(f"/lists/{list_id}/sources/BV1abc")
    assert resp.status_code == 204
    assert not note_file.exists()
    assert await get_source("BV1abc") is None
    mock_clear.assert_called_once()
    assert mock_clear.call_args[0][0] == "BV1abc"


@pytest.mark.asyncio
async def test_delete_source_not_found(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
    assert (await client.delete(f"/lists/{list_id}/sources/nonexistent")).status_code == 404

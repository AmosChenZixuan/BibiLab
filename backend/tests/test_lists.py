from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


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
    assert names == ["List B", "List A"]


@pytest.mark.asyncio
async def test_get_lists_returns_thumbnail_fields_and_prefers_cached_cover(
    client: httpx.AsyncClient, tmp_bibilab_home: Path
):

    list_id = (await client.post("/lists", json={"name": "Annotated"})).json()["id"]
    source_id = "BV1cover"
    cover_path = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_bytes(b"fake-image")

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BV1cover",
        title="Episode 1",
        cover_url="https://example.com/remote-cover.jpg",
        source_url="https://www.bilibili.com/video/BV1cover",
    )

    patch_resp = await client.patch(
        f"/lists/{list_id}",
        json={"name": "Annotated", "thumbnail_source_id": source_id},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["thumbnail_source_id"] == source_id

    list_resp = await client.get("/lists")
    assert list_resp.status_code == 200
    assert list_resp.json() == [
        {
            "id": list_id,
            "name": "Annotated",
            "created_at": patch_resp.json()["created_at"],
            "thumbnail_source_id": source_id,
            "thumbnail_url": f"/api/sources/{source_id}/cover",
            "source_count": 1,
            "updated_at": patch_resp.json()["updated_at"],
        }
    ]

    cover_resp = await client.get(f"/sources/{source_id}/cover")
    assert cover_resp.status_code == 200
    assert cover_resp.content == b"fake-image"


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
async def test_delete_list_rejects_active_jobs(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_job

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
async def test_get_list_sources(client: httpx.AsyncClient, tmp_bibilab_home: Path):

    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
    source_id = "src-list-src"
    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BV1abc",
        title="Intro",
        source_url="https://www.bilibili.com/video/BV1abc",
    )
    resp = await client.get(f"/lists/{list_id}/sources")
    assert resp.status_code == 200
    sources = resp.json()
    assert len(sources) == 1
    assert sources[0]["id"] == source_id
    assert sources[0]["video_id"] == "BV1abc"
    assert sources[0]["title"] == "Intro"


@pytest.mark.asyncio
async def test_delete_source_from_list(client: httpx.AsyncClient, tmp_bibilab_home: Path, tmp_path: Path):
    from bibilab.db import get_source

    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
    source_id = "src-delete-src"
    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BV1abc",
        title="Intro",
        source_url="https://www.bilibili.com/video/BV1abc",
    )
    with patch("bibilab.routers.lists.clear_embeddings_for_source") as mock_clear:
        resp = await client.delete(f"/lists/{list_id}/sources/{source_id}")
    assert resp.status_code == 204
    assert await get_source(source_id) is None
    mock_clear.assert_called_once()
    assert mock_clear.call_args[0][0] == source_id


@pytest.mark.asyncio
async def test_delete_source_clears_thumbnail_and_cover(
    client: httpx.AsyncClient, tmp_bibilab_home: Path, tmp_path: Path
):
    from bibilab.db import get_list

    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
    source_id = "src-thumb-src"
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"fake-image")
    assert cover_file.exists()

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BV1abc",
        title="Intro",
        source_url="https://www.bilibili.com/video/BV1abc",
    )

    # Assign the source as the list's thumbnail
    await client.patch(f"/lists/{list_id}", json={"thumbnail_source_id": source_id})

    with patch("bibilab.routers.lists.clear_embeddings_for_source"):
        resp = await client.delete(f"/lists/{list_id}/sources/{source_id}")
    assert resp.status_code == 204

    # Thumbnail reference is cleared
    row = await get_list(list_id)
    assert row["thumbnail_source_id"] is None

    # Cover file is deleted
    assert not cover_file.exists()


@pytest.mark.asyncio
async def test_delete_source_not_found(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "ML"})).json()["id"]
    assert (await client.delete(f"/lists/{list_id}/sources/nonexistent")).status_code == 404


@pytest.mark.asyncio
async def test_first_source_auto_assigned_as_thumbnail(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import get_list_with_display

    # Create an empty list
    list_id = (await client.post("/lists", json={"name": "Empty"})).json()["id"]

    # Write the first source (simulating first ingest completing)
    source_id = "BVfirst"
    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVfirstVid",
        title="First Video",
        source_url="https://www.bilibili.com/video/BVfirstVid",
    )

    # List's thumbnail_source_id should be auto-assigned to the first source
    row = await get_list_with_display(list_id)
    assert row["thumbnail_source_id"] == source_id


@pytest.mark.asyncio
async def test_delete_source_cascades_segments_no_txt(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import _now, bootstrap_db, create_list, get_db, get_transcript_segments, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", _now())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id) VALUES (?, ?, ?, ?)",
            ("src-1", "BV1", "bilibili", "list-1"),
        )
        await db.commit()
    await write_transcript_segments("src-1", [WhisperSegment(start=0.0, end=1.0, text="x。", speaker=None)])

    resp = await client.delete("/lists/list-1/sources/src-1")
    assert resp.status_code == 204
    assert await get_transcript_segments("src-1") == []

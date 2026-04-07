import json
from pathlib import Path

import httpx
import pytest

from bibilab.db import write_source


@pytest.mark.asyncio
async def test_get_source_returns_digest_and_transcript(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list

    list_id = "list-for-source-test"
    await create_list(list_id, "Test List", "2025-01-01T00:00:00Z")

    source_id = "src-abc123"
    video_id = "BVtest456"
    transcript_path = f"transcripts/{video_id}.txt"

    # Write the transcript file
    transcript_file = tmp_bibilab_home / transcript_path
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("hello transcript", encoding="utf-8")

    # Write the cover file
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"fake-cover-image")

    await write_source(
        source_id=source_id,
        video_id=video_id,
        platform="bilibili",
        list_id=list_id,
        title="Test Video",
        summary="A test summary.",
        keywords=["ai", "video"],
        cover_url="https://example.com/cover.jpg",
        transcript_path=transcript_path,
        source_url="https://www.bilibili.com/video/BVtest456",
        duration_seconds=120,
        uploader="TestUploader",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={"language": "en"},
    )

    resp = await client.get(f"/sources/{source_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == source_id
    assert data["summary"] == "A test summary."
    assert data["keywords"] == ["ai", "video"]
    assert data["transcript"] == "hello transcript"
    assert data["cover_url"] == "https://example.com/cover.jpg"
    assert data["video_id"] == video_id


@pytest.mark.asyncio
async def test_get_source_cover(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list

    list_id = "list-for-cover-test"
    await create_list(list_id, "Cover Test List", "2025-01-01T00:00:00Z")

    source_id = "src-cover-789"
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"cover-image-bytes")

    await write_source(
        source_id=source_id,
        video_id="BVcoverVid",
        platform="bilibili",
        list_id=list_id,
        title="Cover Test Video",
        summary="S",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://www.bilibili.com/video/BVcoverVid",
        duration_seconds=60,
        uploader="Uploader",
        language=None,
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    resp = await client.get(f"/sources/{source_id}/cover")
    assert resp.status_code == 200
    assert resp.content == b"cover-image-bytes"


@pytest.mark.asyncio
async def test_get_source_not_found(client: httpx.AsyncClient):
    resp = await client.get("/sources/nonexistent-source-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_source_cover_not_found(client: httpx.AsyncClient):
    resp = await client.get("/sources/nonexistent-source-id/cover")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rerun_source_not_found(client: httpx.AsyncClient):
    """POST /sources/{source_id}/rerun returns 404 when source does not exist."""
    resp = await client.post("/sources/nonexistent-source-id/rerun")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rerun_source_success(client: httpx.AsyncClient, tmp_bibilab_home: Path, monkeypatch):
    """POST /sources/{source_id}/rerun re-runs digest and updates summary/keywords."""
    from bibilab.db import create_list, get_source, write_source

    await create_list("list-rerun-test", "Rerun Test", "2025-01-01T00:00:00Z")

    source_id = "src-rerun-001"
    video_id = "BVrerun001"
    transcript_path = f"transcripts/{video_id}.txt"
    transcript_file = tmp_bibilab_home / transcript_path
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("original transcript text", encoding="utf-8")

    await write_source(
        source_id=source_id,
        video_id="BVrerun001",
        platform="bilibili",
        list_id="list-rerun-test",
        title="Rerun Test Video",
        summary="old summary",
        keywords=["old", "keyword"],
        cover_url=None,
        transcript_path=transcript_path,
        source_url="https://www.bilibili.com/video/BVrerun001",
        duration_seconds=120,
        uploader="TestUser",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Mock the LLM call to return new digest
    new_digest = '{"summary": "new summary from rerun", "keywords": ["new", "rerun", "test"]}'

    def mock_call_llm(prompt, cfg):
        return new_digest

    import bibilab.pipeline.digest as digest_module

    monkeypatch.setattr(digest_module, "_call_llm", mock_call_llm)

    resp = await client.post(f"/sources/{source_id}/rerun")
    assert resp.status_code == 200

    # Verify summary and keywords were updated
    source = await get_source(source_id)
    assert source is not None
    assert source["summary"] == "new summary from rerun"
    assert json.loads(source["keywords"]) == ["new", "rerun", "test"]

    # Verify transcript file was not modified
    assert transcript_file.read_text(encoding="utf-8") == "original transcript text"

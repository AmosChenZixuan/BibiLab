from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_get_note_content(client: httpx.AsyncClient, tmp_bibilab_home: Path, tmp_path: Path):
    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML", "2026-01-01T00:00:00")

    note_file = tmp_bibilab_home / "notes" / "BV1abc.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text("# Intro to ML\n\n## Summary\nGreat video.", encoding="utf-8")

    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="Intro to ML",
        summary="Great video.",
        note_path=str(note_file),
        transcript_path=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    resp = await client.get("/notes/BV1abc/content")
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == "BV1abc"
    assert data["title"] == "Intro to ML"
    assert "Intro to ML" in data["markdown"]


@pytest.mark.asyncio
async def test_get_note_content_not_found(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db

    await bootstrap_db()
    assert (await client.get("/notes/nonexistent/content")).status_code == 404


@pytest.mark.asyncio
async def test_get_transcript(client: httpx.AsyncClient, tmp_bibilab_home: Path, tmp_path: Path):
    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML", "2026-01-01T00:00:00")

    note_file = tmp_bibilab_home / "notes" / "BV1abc.md"
    note_file.parent.mkdir(parents=True, exist_ok=True)
    note_file.write_text("# Note", encoding="utf-8")
    transcript_file = tmp_bibilab_home / "transcripts" / "BV1abc.txt"
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("[00:00:01] Hello world", encoding="utf-8")

    await write_source(
        video_id="BV1abc",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        note_path=str(note_file),
        transcript_path=str(transcript_file),
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    resp = await client.get("/notes/BV1abc/transcript")
    assert resp.status_code == 200
    assert resp.json()["text"] == "[00:00:01] Hello world"


@pytest.mark.asyncio
async def test_patch_note_path_endpoint_removed(client: httpx.AsyncClient):
    resp = await client.patch("/notes/BV1abc/path", json={"path": "/some/path"})
    assert resp.status_code == 404

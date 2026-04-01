from pathlib import Path
from unittest.mock import patch

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


@pytest.mark.asyncio
async def test_get_note_content(client: TestClient, tmp_locus_home: Path, tmp_path: Path):
    from locus.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML", "2026-01-01T00:00:00")

    note_file = tmp_path / "BV1abc.md"
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

    resp = client.get("/notes/BV1abc/content")
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == "BV1abc"
    assert data["title"] == "Intro to ML"
    assert "Intro to ML" in data["markdown"]


@pytest.mark.asyncio
async def test_get_note_content_not_found(client: TestClient, tmp_locus_home: Path):
    from locus.db import bootstrap_db

    await bootstrap_db()
    assert client.get("/notes/nonexistent/content").status_code == 404


@pytest.mark.asyncio
async def test_get_transcript(client: TestClient, tmp_locus_home: Path, tmp_path: Path):
    from locus.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "ML", "2026-01-01T00:00:00")

    note_file = tmp_path / "BV1abc.md"
    note_file.write_text("# Note", encoding="utf-8")
    transcript_file = tmp_path / "BV1abc.txt"
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

    resp = client.get("/notes/BV1abc/transcript")
    assert resp.status_code == 200
    assert resp.json()["text"] == "[00:00:01] Hello world"


def test_patch_note_path_endpoint_removed(client: TestClient):
    resp = client.patch("/notes/BV1abc/path", json={"path": "/some/path"})
    assert resp.status_code == 404

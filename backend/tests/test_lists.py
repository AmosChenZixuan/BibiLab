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


@pytest.fixture()
def created_list_id(client: TestClient) -> str:
    response = client.post("/lists", json={"name": "ML Course"})
    assert response.status_code == 201
    return response.json()["id"]


@pytest.mark.asyncio
async def test_get_list_notes_returns_processing_log_rows(client: TestClient, created_list_id: str):
    from locus.db import get_db

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO processing_log (
                video_id, platform, list_id, note_path, transcript_path,
                whisper_model, ai_model, vision_enabled, processed_at, settings_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BV1abc",
                "bilibili",
                created_list_id,
                "Locus/ML Course/Intro.md",
                "/tmp/transcripts/BV1abc.txt",
                "large-v3",
                "gpt-4o",
                0,
                "2026-03-30T12:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client.get(f"/lists/{created_list_id}/notes")
    assert response.status_code == 200
    assert response.json() == [
        {
            "video_id": "BV1abc",
            "note_path": "Locus/ML Course/Intro.md",
            "processed_at": "2026-03-30T12:00:00+00:00",
            "platform": "bilibili",
        }
    ]


@pytest.mark.asyncio
async def test_delete_list_rejects_active_jobs(client: TestClient, created_list_id: str):
    from locus.db import create_job

    await create_job(
        type="video",
        source_url="https://www.bilibili.com/video/BV1active",
        platform="bilibili",
        meta={"video_id": "BV1active", "list_id": created_list_id},
    )

    response = client.delete(f"/lists/{created_list_id}")
    assert response.status_code == 409
    assert response.json()["detail"] == "Cannot delete a list with active jobs"


def test_delete_list_removes_row_when_inactive(client: TestClient, created_list_id: str):
    response = client.delete(f"/lists/{created_list_id}")
    assert response.status_code == 204
    assert client.get("/lists").json() == []


@pytest.mark.asyncio
async def test_delete_list_cascades_processing_log(client: TestClient, tmp_locus_home: Path):
    import aiosqlite

    response = client.post("/lists", json={"name": "ToDelete"})
    assert response.status_code == 201
    list_id = response.json()["id"]

    db_path = tmp_locus_home / "locus.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO processing_log
              (video_id, platform, list_id, note_path, transcript_path,
               whisper_model, ai_model, vision_enabled, processed_at, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BV1test",
                "bilibili",
                list_id,
                None,
                None,
                "large-v3",
                "gpt-4o",
                0,
                "2026-01-01T00:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client.delete(f"/lists/{list_id}")
    assert response.status_code == 204

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT 1 FROM processing_log WHERE list_id=?", (list_id,)) as cur:
            row = await cur.fetchone()
    assert row is None

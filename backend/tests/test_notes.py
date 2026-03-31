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
async def test_patch_note_path_updates_processing_log(client: TestClient):
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
                "BV1patch",
                "bilibili",
                "list-1",
                "old/path.md",
                "/tmp/transcripts/BV1patch.txt",
                "large-v3",
                "gpt-4o",
                0,
                "2026-03-30T12:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client.patch("/notes/BV1patch/path", json={"path": "new/path.md"})
    assert response.status_code == 204

    async with get_db() as db:
        async with db.execute(
            "SELECT note_path FROM processing_log WHERE video_id=?",
            ("BV1patch",),
        ) as cur:
            row = await cur.fetchone()
    assert row["note_path"] == "new/path.md"


@pytest.mark.asyncio
async def test_patch_note_path_accepts_null(client: TestClient):
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
                "BV1null",
                "bilibili",
                "list-1",
                "note.md",
                "/tmp/transcripts/BV1null.txt",
                "large-v3",
                "gpt-4o",
                0,
                "2026-03-30T12:00:00+00:00",
                "{}",
            ),
        )
        await db.commit()

    response = client.patch("/notes/BV1null/path", json={"path": None})
    assert response.status_code == 204

    async with get_db() as db:
        async with db.execute(
            "SELECT note_path FROM processing_log WHERE video_id=?",
            ("BV1null",),
        ) as cur:
            row = await cur.fetchone()
    assert row["note_path"] is None


def test_patch_note_path_returns_404_for_missing_note(client: TestClient):
    response = client.patch("/notes/BV1missing/path", json={"path": "any.md"})
    assert response.status_code == 404

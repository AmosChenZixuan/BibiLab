"""Tests for job queue CRUD and state machine."""

import logging
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio


@pytest_asyncio.fixture()
async def seeded_job(tmp_bibilab_home: Path):  # noqa: ARG001
    """Bootstrap DB and insert a queued job (tmp_bibilab_home ensures patch is active)."""
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    return await create_job(
        type="ingest",
        meta={
            "video_id": "BV1test",
            "list_id": "list-1",
            "source_url": "https://bilibili.com/video/BV1test",
            "platform": "bilibili",
        },
    )


@pytest.mark.asyncio
async def test_list_jobs_empty(client: httpx.AsyncClient):
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_job_by_id(client: httpx.AsyncClient, seeded_job: str):
    resp = await client.get(f"/jobs/{seeded_job}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == seeded_job
    assert data["status"] == "queued"
    assert data["type"] == "ingest"
    assert "platform" not in data
    assert "source_url" not in data
    assert data["meta"]["platform"] == "bilibili"
    assert data["meta"]["source_url"] == "https://bilibili.com/video/BV1test"


@pytest.mark.asyncio
async def test_get_job_not_found(client: httpx.AsyncClient):
    resp = await client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_queued_job(client: httpx.AsyncClient, seeded_job: str):
    resp = await client.delete(f"/jobs/{seeded_job}")
    assert resp.status_code == 204
    # Verify it's gone
    assert (await client.get(f"/jobs/{seeded_job}")).status_code == 404


@pytest.mark.asyncio
async def test_state_transitions(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, get_job, update_job_status

    await bootstrap_db()
    job_id = await create_job("ingest", {"source_url": "https://b.tv/BV1", "platform": "bilibili"})

    row = await get_job(job_id)
    assert dict(row)["status"] == "queued"

    for status, progress in [
        ("downloading", 10),
        ("transcribing", 40),
        ("extracting", 70),
        ("writing", 90),
        ("done", 100),
    ]:
        await update_job_status(job_id, status, progress)
        row = await get_job(job_id)
        assert dict(row)["status"] == status
        assert dict(row)["progress"] == progress


@pytest.mark.asyncio
async def test_reset_stuck_jobs(tmp_bibilab_home: Path):
    from bibilab.db import (
        bootstrap_db,
        create_job,
        get_job,
        reset_stuck_jobs,
        update_job_status,
    )

    await bootstrap_db()
    job_id = await create_job("ingest", {"source_url": "https://b.tv/BV1", "platform": "bilibili"})
    await update_job_status(job_id, "transcribing", 40)

    await reset_stuck_jobs()

    row = await get_job(job_id)
    assert dict(row)["status"] == "queued"


@pytest.mark.asyncio
async def test_pipeline_fails_fast_when_list_deleted(tmp_bibilab_home: Path):
    """Worker must fail the job if the target list no longer exists."""
    from bibilab.db import bootstrap_db, create_job, get_job
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    transcript_path = tmp_bibilab_home / "transcript.txt"
    transcript_path.write_text("hello world", encoding="utf-8")

    job_id = await create_job(
        type="ingest",
        meta={
            "video_id": "BV1abc123",
            "list_id": "nonexistent-list",
            "title": "Test",
            "cover_url": "",
            "duration_seconds": 0,
            "uploader": "",
            "rerun": False,
            "platform": "bilibili",
            "source_url": "https://www.bilibili.com/video/BV1abc123",
        },
    )

    job_row = dict(await get_job(job_id))
    worker = WorkerLoop()

    cfg = MagicMock()
    cfg.accounts.bilibili.cookie = ""
    cfg.transcription.model_size = "large-v3"
    cfg.ai.model = "gpt-4o"
    cfg.vision.enabled = False
    cfg.model_dump.return_value = {}

    with (
        patch("bibilab.worker.load_config", return_value=cfg),
        patch(
            "bibilab.worker.BilibiliAdapter.download", return_value=tmp_bibilab_home / "video.mp4"
        ),
        patch("bibilab.worker.extract_audio", return_value=tmp_bibilab_home / "audio.wav"),
        patch("bibilab.worker.transcribe", return_value=[]),
        patch("bibilab.worker.write_transcript", return_value=transcript_path),
        patch("bibilab.worker.chunk_segments", return_value=[]),
        patch(
            "bibilab.worker.extract_knowledge",
            return_value=MagicMock(title="Test", summary="Summary"),
        ),
        patch("bibilab.worker.write_video_note", return_value=tmp_bibilab_home / "note.md"),
        patch("bibilab.worker.embed_chunks", return_value=None),
    ):
        await worker._run_job(job_row)

    result = dict(await get_job(job_id))
    assert result["status"] == "failed"
    assert "list" in result["error"].lower()


@pytest.mark.asyncio
async def test_pipeline_logs_when_embedding_model_missing(tmp_bibilab_home: Path, caplog):
    from bibilab.worker import WorkerLoop

    transcript_path = tmp_bibilab_home / "transcript.txt"
    transcript_path.write_text("hello world", encoding="utf-8")

    note_path = tmp_bibilab_home / "notes" / "note.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("", encoding="utf-8")

    cfg = MagicMock()
    cfg.accounts.bilibili.cookie = ""
    cfg.transcription.model_size = "large-v3"
    cfg.ai.model = "gpt-4o"
    cfg.vision.enabled = False
    cfg.model_dump.return_value = {}

    extraction = MagicMock()
    extraction.title = "Test title"
    extraction.summary = "Summary"

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    job = {
        "id": "test-job-id",
        "type": "ingest",
        "meta": {
            "video_id": "BV1abc123",
            "list_id": "list-001",
            "title": "T",
            "cover_url": "",
            "duration_seconds": 0,
            "uploader": "",
            "rerun": False,
            "platform": "bilibili",
            "source_url": "https://www.bilibili.com/video/BV1abc123",
        },
    }

    worker = WorkerLoop()
    with (
        patch("bibilab.worker.load_config", return_value=cfg),
        patch("bibilab.worker.get_list", new=AsyncMock(return_value={"id": "list-001"})),
        patch("bibilab.worker.update_job_status", new=AsyncMock()),
        patch("bibilab.worker.asyncio.to_thread", new=AsyncMock(side_effect=fake_to_thread)),
        patch(
            "bibilab.worker.BilibiliAdapter.download", return_value=tmp_bibilab_home / "video.mp4"
        ),
        patch("bibilab.worker.extract_audio", return_value=tmp_bibilab_home / "audio.wav"),
        patch("bibilab.worker.transcribe", return_value=[]),
        patch("bibilab.worker.write_transcript", return_value=transcript_path),
        patch("bibilab.worker.chunk_segments", return_value=[]),
        patch("bibilab.worker.extract_knowledge", return_value=extraction),
        patch("bibilab.worker.write_video_note", return_value=note_path),
        patch("bibilab.worker.embed_chunks", return_value=None),
        patch("bibilab.worker.write_source", new=AsyncMock()),
        patch("bibilab.worker.is_embedding_model_downloaded", return_value=False),
        caplog.at_level(logging.INFO, logger="bibilab.worker"),
    ):
        await worker._pipeline(job)

    assert any("embedding model not found" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_delete_job_cleans_up_ingest_artifacts(
    client: httpx.AsyncClient, tmp_bibilab_home: Path
):
    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    video_id = "BV1cleanup"
    transcript_path = tmp_bibilab_home / "transcripts" / f"{video_id}.txt"
    note_path = tmp_bibilab_home / "notes" / f"{video_id}.md"
    cover_path = tmp_bibilab_home / "notes" / "attachments" / f"{video_id}_cover.jpg"
    download_path = tmp_bibilab_home / "downloads" / f"{video_id}.mp4"

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    download_path.parent.mkdir(parents=True, exist_ok=True)

    transcript_path.write_text("transcript", encoding="utf-8")
    note_path.write_text("# note", encoding="utf-8")
    cover_path.write_text("cover", encoding="utf-8")
    download_path.write_text("video", encoding="utf-8")

    job_id = await create_job(
        type="ingest",
        meta={
            "video_id": video_id,
            "list_id": "list-1",
            "title": "Cleanup Me",
            "platform": "bilibili",
            "source_url": f"https://www.bilibili.com/video/{video_id}",
        },
    )

    db_path = tmp_bibilab_home / "bibilab.db"
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
        db.commit()

    with patch("bibilab.cleanup.clear_embeddings_for_video") as mock_clear:
        resp = await client.delete(f"/jobs/{job_id}")

    assert resp.status_code == 204
    assert await get_job(job_id) is None
    assert not transcript_path.exists()
    assert not note_path.exists()
    assert not cover_path.exists()
    assert not download_path.exists()
    mock_clear.assert_called_once()
    assert mock_clear.call_args[0][0] == video_id

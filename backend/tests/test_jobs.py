"""Tests for job queue CRUD and state machine."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
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


@pytest_asyncio.fixture()
async def seeded_job(tmp_locus_home: Path):  # noqa: ARG001
    """Bootstrap DB and insert a queued job (tmp_locus_home ensures patch is active)."""
    from locus.db import bootstrap_db, create_job

    await bootstrap_db()
    return await create_job(
        type="video",
        source_url="https://bilibili.com/video/BV1test",
        platform="bilibili",
        meta={"video_id": "BV1test", "list_id": "list-1"},
    )


def test_list_jobs_empty(client: TestClient):
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_job_by_id(client: TestClient, seeded_job: str):
    resp = client.get(f"/jobs/{seeded_job}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == seeded_job
    assert data["status"] == "queued"
    assert data["platform"] == "bilibili"


@pytest.mark.asyncio
async def test_get_job_not_found(client: TestClient):
    resp = client.get("/jobs/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_queued_job(client: TestClient, seeded_job: str):
    resp = client.delete(f"/jobs/{seeded_job}")
    assert resp.status_code == 204
    # Verify it's gone
    assert client.get(f"/jobs/{seeded_job}").status_code == 404


@pytest.mark.asyncio
async def test_state_transitions(tmp_locus_home: Path):
    from locus.db import bootstrap_db, create_job, get_job, update_job_status

    await bootstrap_db()
    job_id = await create_job("video", "https://b.tv/BV1", "bilibili", {})

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
async def test_reset_stuck_jobs(tmp_locus_home: Path):
    from locus.db import (
        bootstrap_db,
        create_job,
        get_job,
        reset_stuck_jobs,
        update_job_status,
    )

    await bootstrap_db()
    job_id = await create_job("video", "https://b.tv/BV1", "bilibili", {})
    await update_job_status(job_id, "transcribing", 40)

    await reset_stuck_jobs()

    row = await get_job(job_id)
    assert dict(row)["status"] == "queued"


@pytest.mark.asyncio
async def test_pipeline_fails_fast_when_list_deleted(tmp_locus_home: Path):
    """Worker must fail the job if the target list no longer exists."""
    from locus.db import bootstrap_db, create_job, get_job
    from locus.worker import WorkerLoop

    await bootstrap_db()
    transcript_path = tmp_locus_home / "transcript.txt"
    transcript_path.write_text("hello world", encoding="utf-8")

    job_id = await create_job(
        type="video",
        source_url="https://www.bilibili.com/video/BV1abc123",
        platform="bilibili",
        meta={
            "video_id": "BV1abc123",
            "list_id": "nonexistent-list",
            "title": "Test",
            "cover_url": "",
            "duration_seconds": 0,
            "uploader": "",
            "rerun": False,
        },
    )

    job_row = dict(await get_job(job_id))
    worker = WorkerLoop()

    cfg = MagicMock()
    cfg.accounts.bilibili.cookie = ""
    cfg.transcription.model_size = "large-v3"
    cfg.ai.model = "gpt-4o"
    cfg.vision.enabled = False
    cfg.obsidian.vault_path = str(tmp_locus_home / "vault")
    cfg.model_dump.return_value = {}

    with (
        patch("locus.worker.load_config", return_value=cfg),
        patch("locus.worker.BilibiliAdapter.download", return_value=tmp_locus_home / "video.mp4"),
        patch("locus.worker.extract_audio", return_value=tmp_locus_home / "audio.wav"),
        patch("locus.worker.transcribe", return_value=[]),
        patch("locus.worker.write_transcript", return_value=transcript_path),
        patch("locus.worker.chunk_segments", return_value=[]),
        patch(
            "locus.worker.extract_knowledge",
            return_value=MagicMock(title="Test", summary="Summary"),
        ),
        patch("locus.worker.write_video_note", return_value=tmp_locus_home / "vault/note.md"),
        patch("locus.worker.embed_chunks", return_value=None),
        patch("locus.worker.write_processing_log", return_value=None),
        patch("locus.worker.get_processing_log_videos", return_value=[]),
        patch("locus.worker.generate_overview", return_value=[]),
        patch("locus.worker.write_overview_note", return_value=None),
    ):
        await worker._run_job(job_row)

    result = dict(await get_job(job_id))
    assert result["status"] == "failed"
    assert "list" in result["error"].lower()


@pytest.mark.asyncio
async def test_pipeline_logs_when_embedding_model_missing(tmp_locus_home: Path, caplog):
    from locus.worker import WorkerLoop

    transcript_path = tmp_locus_home / "transcript.txt"
    transcript_path.write_text("hello world", encoding="utf-8")

    note_path = tmp_locus_home / "vault" / "note.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("", encoding="utf-8")

    cfg = MagicMock()
    cfg.accounts.bilibili.cookie = ""
    cfg.transcription.model_size = "large-v3"
    cfg.ai.model = "gpt-4o"
    cfg.vision.enabled = False
    cfg.obsidian.vault_path = str(tmp_locus_home / "vault")
    cfg.model_dump.return_value = {}

    extraction = MagicMock()
    extraction.title = "Test title"
    extraction.summary = "Summary"

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    job = {
        "id": "test-job-id",
        "type": "video",
        "platform": "bilibili",
        "source_url": "https://www.bilibili.com/video/BV1abc123",
        "meta": {
            "video_id": "BV1abc123",
            "list_id": "list-001",
            "title": "T",
            "cover_url": "",
            "duration_seconds": 0,
            "uploader": "",
            "rerun": False,
        },
    }

    worker = WorkerLoop()
    with (
        patch("locus.worker.load_config", return_value=cfg),
        patch("locus.worker._get_list_name_from_vault", return_value="MyList"),
        patch("locus.worker.update_job_status", new=AsyncMock()),
        patch("locus.worker.asyncio.to_thread", new=AsyncMock(side_effect=fake_to_thread)),
        patch("locus.worker.BilibiliAdapter.download", return_value=tmp_locus_home / "video.mp4"),
        patch("locus.worker.extract_audio", return_value=tmp_locus_home / "audio.wav"),
        patch("locus.worker.transcribe", return_value=[]),
        patch("locus.worker.write_transcript", return_value=transcript_path),
        patch("locus.worker.chunk_segments", return_value=[]),
        patch("locus.worker.extract_knowledge", return_value=extraction),
        patch("locus.worker.write_video_note", return_value=note_path),
        patch("locus.worker.embed_chunks", return_value=None),
        patch("locus.worker.write_processing_log", new=AsyncMock()),
        patch("locus.worker.get_processing_log_videos", new=AsyncMock(return_value=[])),
        patch("locus.worker.generate_overview", return_value=[]),
        patch("locus.worker.write_overview_note", return_value=None),
        patch("locus.worker.is_embedding_model_downloaded", return_value=False),
        caplog.at_level(logging.INFO, logger="locus.worker"),
    ):
        await worker._pipeline(job)

    assert any("embedding model not found" in record.message.lower() for record in caplog.records)

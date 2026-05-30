"""Tests for job queue CRUD and state machine."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

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
async def test_delete_job_cleans_up_ingest_artifacts(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    video_id = "BV1cleanup"
    download_path = tmp_bibilab_home / "downloads" / f"{video_id}.mp4"

    download_path.parent.mkdir(parents=True, exist_ok=True)

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
    assert not download_path.exists()
    mock_clear.assert_called_once()
    assert mock_clear.call_args[0][0] == video_id


@pytest.mark.asyncio
async def test_delete_job_cleans_up_cover_when_source_id_in_meta(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    source_id = "BV1cover-cleanup"
    video_id = "BV1cover-cleanup-vid"
    cover_path = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_bytes(b"fake-cover")

    job_id = await create_job(
        type="ingest",
        meta={
            "video_id": video_id,
            "source_id": source_id,
            "list_id": "list-1",
            "title": "Cover Cleanup",
            "platform": "bilibili",
            "source_url": f"https://www.bilibili.com/video/{video_id}",
        },
    )

    db_path = tmp_bibilab_home / "bibilab.db"
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
        db.commit()

    with patch("bibilab.cleanup.clear_embeddings_for_video"):
        resp = await client.delete(f"/jobs/{job_id}")

    assert resp.status_code == 204
    assert await get_job(job_id) is None
    assert not cover_path.exists()

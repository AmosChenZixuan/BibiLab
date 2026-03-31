"""Tests for job queue CRUD and state machine."""

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
async def seeded_job(tmp_locus_home: Path):
    """Bootstrap DB and insert a queued job."""
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
    from locus.db import bootstrap_db, create_job, get_job, reset_stuck_jobs, update_job_status

    await bootstrap_db()
    job_id = await create_job("video", "https://b.tv/BV1", "bilibili", {})
    await update_job_status(job_id, "transcribing", 40)

    await reset_stuck_jobs()

    row = await get_job(job_id)
    assert dict(row)["status"] == "queued"

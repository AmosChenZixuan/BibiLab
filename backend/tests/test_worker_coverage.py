"""Tests for worker.py uncovered paths: _download_cover, _download_model_job,
_run_job dispatch/exception handling, _run_artifact_job error paths, start/stop."""

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.db import bootstrap_db, create_list
from bibilab.worker import WorkerLoop, _download_cover, _parse_job_meta

# ---------------------------------------------------------------------------
# _download_cover
# ---------------------------------------------------------------------------


class TestDownloadCover:
    def test_success(self, tmp_path: Path):
        dest = tmp_path / "cover.jpg"
        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG"
        mock_resp.raise_for_status = MagicMock()

        with patch("bibilab.worker.httpx.get", return_value=mock_resp):
            assert _download_cover("https://example.com/cover.jpg", dest) is True
        assert dest.read_bytes() == b"\x89PNG"

    def test_http_error(self, tmp_path: Path):
        import httpx

        dest = tmp_path / "cover.jpg"
        with patch("bibilab.worker.httpx.get", side_effect=httpx.HTTPError("timeout")):
            assert _download_cover("https://example.com/cover.jpg", dest) is False
        assert not dest.exists()

    def test_os_error(self, tmp_path: Path):
        dest = tmp_path / "nonexistent" / "cover.jpg"
        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG"
        mock_resp.raise_for_status = MagicMock()

        with patch("bibilab.worker.httpx.get", return_value=mock_resp):
            assert _download_cover("https://example.com/cover.jpg", dest) is False


# ---------------------------------------------------------------------------
# _parse_job_meta
# ---------------------------------------------------------------------------


class TestParseJobMeta:
    def test_dict_meta(self):
        assert _parse_job_meta({"meta": {"key": "val"}}) == {"key": "val"}

    def test_string_meta(self):
        assert _parse_job_meta({"meta": '{"key": "val"}'}) == {"key": "val"}

    def test_empty_string_meta(self):
        assert _parse_job_meta({"meta": ""}) == {}

    def test_missing_meta(self):
        assert _parse_job_meta({}) == {}


# ---------------------------------------------------------------------------
# _download_model_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_model_job_success(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    meta = {"model_family": "whisper", "model_size": "tiny"}
    job_id = await create_job("model_download", meta)

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {"id": job_id, "type": "model_download", "meta": json.dumps(meta)}

    with patch("bibilab.worker.download_whisper_model") as mock_dl:
        await worker._download_model_job(job)
        mock_dl.assert_called_once_with("tiny")


@pytest.mark.asyncio
async def test_download_model_job_unsupported_family(tmp_bibilab_home: Path):
    from bibilab.pipeline.audio import PipelineError

    await bootstrap_db()
    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {"id": "job-1", "type": "model_download", "meta": json.dumps({"model_family": "llama", "model_size": "7b"})}

    with pytest.raises(PipelineError, match="Unsupported model family"):
        await worker._download_model_job(job)


# ---------------------------------------------------------------------------
# _run_job dispatch and exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_job_cancelled_before_start(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker.cancel_job(job_id)

    job = {"id": job_id, "type": "ingest", "meta": "{}"}
    await worker._run_job(job)

    assert job_id not in worker._cancelled
    assert job_id not in worker._in_flight


@pytest.mark.asyncio
async def test_run_job_auth_required_error(tmp_bibilab_home: Path):
    from bibilab.adapters.base import AuthRequiredError
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with patch.object(worker, "_pipeline", side_effect=AuthRequiredError("video")):
        await worker._run_job(job)

    assert job_id not in worker._in_flight


@pytest.mark.asyncio
async def test_run_job_pipeline_error(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job
    from bibilab.pipeline.audio import PipelineError

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with patch.object(worker, "_pipeline", side_effect=PipelineError("something broke")):
        await worker._run_job(job)

    assert job_id not in worker._in_flight


@pytest.mark.asyncio
async def test_run_job_generic_exception(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with patch.object(worker, "_pipeline", side_effect=RuntimeError("unexpected")):
        await worker._run_job(job)

    assert job_id not in worker._in_flight


# ---------------------------------------------------------------------------
# _run_artifact_job error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_job_missing_source(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    artifact_id = str(uuid.uuid4())
    meta = {
        "artifact_id": artifact_id,
        "list_id": "list-1",
        "type": "brief",
        "prompt": "Summarize",
        "source_ids": ["nonexistent-source"],
    }
    job_id = await create_job("artifact", meta)

    worker = WorkerLoop(config=MagicMock(), home=tmp_bibilab_home)
    job = {"id": job_id, "type": "artifact", "meta": json.dumps(meta)}

    await worker._run_artifact_job(job)

    from bibilab.db import get_db

    async with get_db() as db:
        cursor = await db.execute("SELECT status, error FROM artifacts WHERE id=?", (artifact_id,))
        row = await cursor.fetchone()
    assert row is not None
    assert row["status"] == "failed"
    assert "not found" in row["error"]


# ---------------------------------------------------------------------------
# WorkerLoop start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_start_stop(tmp_bibilab_home: Path):
    await bootstrap_db()
    worker = WorkerLoop(home=tmp_bibilab_home)
    await worker.start()
    assert worker._running is True
    assert worker._task is not None
    await worker.stop()
    assert worker._running is False

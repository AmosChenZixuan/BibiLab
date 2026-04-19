"""Tests for pipeline correctness issues: cancellation propagation, error handling, transcript lookup."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.db import bootstrap_db, create_list
from bibilab.worker import WorkerLoop


@pytest.fixture()
def setup_pipeline_test(tmp_path: Path):
    """Common setup for pipeline tests."""
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.main.bibilab_home", return_value=tmp_path):
            with patch("bibilab.cleanup.bibilab_home", return_value=tmp_path):
                with patch("bibilab.routers.lists.bibilab_home", return_value=tmp_path):
                    with patch("bibilab.worker.bibilab_home", return_value=tmp_path):
                        with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
                            with patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path):
                                with patch("bibilab.routers.sources.bibilab_home", return_value=tmp_path):
                                    with patch("pathlib.Path.home", return_value=tmp_path):
                                        yield tmp_path


@pytest.mark.asyncio
async def test_extract_audio_cancellation_stops_pipeline(setup_pipeline_test: Path):
    """
    Verify that when a job is cancelled before the pipeline starts,
    no pipeline stages are executed.
    """

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    job_id = "job-cancel-test"
    source_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "type": "ingest",
        "meta": {
            "source_id": source_id,
            "video_id": "BVcancel123",
            "list_id": "list-1",
            "title": "Cancel Test",
            "platform": "bilibili",
            "source_url": "https://bilibili.com/video/BVcancel123",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 100,
            "uploader": "TestUser",
            "ui_lang": "en",
        },
    }

    # Create required dirs
    (setup_pipeline_test / "downloads").mkdir(parents=True, exist_ok=True)
    (setup_pipeline_test / "covers").mkdir(parents=True, exist_ok=True)
    (setup_pipeline_test / "transcripts").mkdir(parents=True, exist_ok=True)

    # Create fake video and wav files
    tmp_video = setup_pipeline_test / "downloads" / "BVcancel123.mp4"
    tmp_video.write_bytes(b"fake video")
    tmp_wav = setup_pipeline_test / "downloads" / "BVcancel123.wav"
    tmp_wav.write_bytes(b"fake wav")

    # Track which stages were called
    stages_called = []

    # Mock adapter and functions
    mock_adapter = MagicMock()
    mock_adapter.download = MagicMock(return_value=tmp_video)

    worker = WorkerLoop(concurrency=1, adapter=mock_adapter, home=setup_pipeline_test)

    def mock_extract_audio(path):
        stages_called.append("extract_audio")
        return tmp_wav

    def mock_transcribe(*args, **kwargs):
        stages_called.append("transcribe")
        return ([], None)

    mock_digest = MagicMock(return_value='{"summary": "test", "keywords": []}')
    mock_embed = MagicMock()
    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover") or True)

    # Cancel BEFORE extract_audio stage is called
    worker.cancel_job(job_id)

    with (
        patch("bibilab.worker.extract_audio", mock_extract_audio),
        patch("bibilab.worker.transcribe", mock_transcribe),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.pipeline.digest._call_llm", mock_digest),
        patch("bibilab.worker.embed_chunks", mock_embed),
    ):
        await worker._pipeline(job)

    # Cancel was set before any stage ran, so no stage should be called
    assert "extract_audio" not in stages_called, (
        f"extract_audio was called even though job was cancelled before pipeline started. "
        f"Stages called: {stages_called}"
    )
    assert "transcribe" not in stages_called, (
        f"transcribe was called even though job was cancelled before pipeline started. Stages called: {stages_called}"
    )


@pytest.mark.asyncio
async def test_pipeline_stage_process_cleanup_called_on_cancellation(setup_pipeline_test: Path):
    """
    Verify that cleanup_job_artifacts is called when a job is cancelled.
    """

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    job_id = "job-cleanup-test"
    source_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "type": "ingest",
        "meta": {
            "source_id": source_id,
            "video_id": "BVcleanup123",
            "list_id": "list-1",
            "title": "Cleanup Test",
            "platform": "bilibili",
            "source_url": "https://bilibili.com/video/BVcleanup123",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 100,
            "uploader": "TestUser",
            "ui_lang": "en",
        },
    }

    (setup_pipeline_test / "downloads").mkdir(parents=True, exist_ok=True)
    (setup_pipeline_test / "covers").mkdir(parents=True, exist_ok=True)
    (setup_pipeline_test / "transcripts").mkdir(parents=True, exist_ok=True)

    tmp_video = setup_pipeline_test / "downloads" / "BVcleanup123.mp4"
    tmp_video.write_bytes(b"fake video")
    tmp_wav = setup_pipeline_test / "downloads" / "BVcleanup123.wav"
    tmp_wav.write_bytes(b"fake wav")

    mock_adapter = MagicMock()
    mock_adapter.download = MagicMock(return_value=tmp_video)

    worker = WorkerLoop(concurrency=1, adapter=mock_adapter, home=setup_pipeline_test)

    mock_extract_audio = MagicMock(return_value=tmp_wav)
    mock_transcribe = MagicMock(return_value=([], None))
    mock_digest = MagicMock(return_value='{"summary": "test", "keywords": []}')
    mock_embed = MagicMock()
    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover") or True)

    cleanup_calls = []

    def mock_cleanup(artifact_job):
        cleanup_calls.append(artifact_job)

    # Cancel BEFORE any stage runs
    worker.cancel_job(job_id)

    with (
        patch("bibilab.worker.extract_audio", mock_extract_audio),
        patch("bibilab.worker.transcribe", mock_transcribe),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.pipeline.digest._call_llm", mock_digest),
        patch("bibilab.worker.embed_chunks", mock_embed),
        patch("bibilab.worker.cleanup_job_artifacts", mock_cleanup),
    ):
        await worker._pipeline(job)

    # Cleanup should have been called when job was cancelled
    assert len(cleanup_calls) > 0, "cleanup_job_artifacts should have been called when job was cancelled"

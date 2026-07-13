"""Tests for pipeline correctness issues: cancellation propagation, error handling, transcript lookup."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.db import bootstrap_db, create_list
from bibilab.worker import WorkerLoop
from tests.factories import SourceFactory


@pytest.fixture()
def setup_pipeline_test(tmp_path: Path):
    """Common setup for pipeline tests."""
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.main.bibilab_home", return_value=tmp_path):
            with patch("bibilab.cleanup.bibilab_home", return_value=tmp_path):
                with patch("bibilab.routers.lists.bibilab_home", return_value=tmp_path):
                    with patch("bibilab.worker.bibilab_home", return_value=tmp_path):
                        with patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path):
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

    def mock_extract_audio(path, expected_duration=0.0):
        stages_called.append("extract_audio")
        return tmp_wav

    def mock_transcribe(*args, **kwargs):
        stages_called.append("transcribe")
        return ([], None)

    mock_embed = MagicMock()
    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover") or True)

    # Cancel BEFORE extract_audio stage is called
    worker.cancel_job(job_id)

    with (
        patch("bibilab.worker.extract_audio", mock_extract_audio),
        patch("bibilab.worker.transcribe", mock_transcribe),
        patch("bibilab.worker._download_cover", mock_dl_cover),
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


def _make_cancel_job(home: Path, job_id: str, video_id: str) -> tuple[dict, "WorkerLoop", Path]:
    """Ingest job dict + worker wired to a mock adapter, plus fake video/wav files."""
    job = {
        "id": job_id,
        "type": "ingest",
        "meta": {
            "source_id": str(uuid.uuid4()),
            "video_id": video_id,
            "list_id": "list-1",
            "title": "Cancel Test",
            "platform": "bilibili",
            "source_url": f"https://bilibili.com/video/{video_id}",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 100,
            "uploader": "TestUser",
            "ui_lang": "en",
        },
    }
    (home / "downloads").mkdir(parents=True, exist_ok=True)
    (home / "covers").mkdir(parents=True, exist_ok=True)
    tmp_video = home / "downloads" / f"{video_id}.mp4"
    tmp_video.write_bytes(b"fake video")
    tmp_wav = home / "downloads" / f"{video_id}.wav"
    tmp_wav.write_bytes(b"fake wav")
    adapter = MagicMock()
    adapter.download = MagicMock(return_value=tmp_video)
    worker = WorkerLoop(concurrency=1, adapter=adapter, home=home)
    return job, worker, tmp_wav


def _assert_cleanup_received_full_job(cleanup_calls: list, video_id: str) -> None:
    # cleanup_job_artifacts must receive a dict it can act on: a type-less
    # dict makes it early-return and purge nothing.
    assert len(cleanup_calls) > 0, "cleanup_job_artifacts should have been called when job was cancelled"
    for received in cleanup_calls:
        assert received.get("type") == "ingest", f"cleanup received an uncleanable job dict: {received}"
        assert received.get("meta", {}).get("video_id") == video_id


@pytest.mark.asyncio
async def test_pipeline_stage_process_cleanup_called_on_cancellation(setup_pipeline_test: Path):
    """
    Verify that cleanup_job_artifacts is called when a job is cancelled.
    """

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    job, worker, tmp_wav = _make_cancel_job(setup_pipeline_test, "job-cleanup-test", "BVcleanup123")
    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover") or True)
    cleanup_calls: list = []

    # Cancel BEFORE any stage runs
    worker.cancel_job("job-cleanup-test")

    with (
        patch("bibilab.worker.extract_audio", MagicMock(return_value=tmp_wav)),
        patch("bibilab.worker.transcribe", MagicMock(return_value=([], None))),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.worker.embed_chunks", MagicMock()),
        patch("bibilab.worker.cleanup_job_artifacts", cleanup_calls.append),
    ):
        await worker._pipeline(job)

    _assert_cleanup_received_full_job(cleanup_calls, "BVcleanup123")


@pytest.mark.asyncio
async def test_pipeline_mid_stage_cancel_cleanup_receives_full_job(setup_pipeline_test: Path):
    """Cancelling while a stage runs (transcribe gate) must purge with the full job dict."""

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    job, worker, tmp_wav = _make_cancel_job(setup_pipeline_test, "job-midcancel-test", "BVmidcancel1")
    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover") or True)
    cleanup_calls: list = []

    def mock_transcribe(*args, **kwargs):
        # Cancel arrives while transcription is running
        worker.cancel_job("job-midcancel-test")
        return ([], None)

    with (
        patch("bibilab.worker.extract_audio", MagicMock(return_value=tmp_wav)),
        patch("bibilab.worker.transcribe", mock_transcribe),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.worker.embed_chunks", MagicMock()),
        patch("bibilab.worker.cleanup_job_artifacts", cleanup_calls.append),
    ):
        await worker._pipeline(job)

    _assert_cleanup_received_full_job(cleanup_calls, "BVmidcancel1")


async def _seed_source(source_id: str) -> None:

    await create_list("list-cleanup", "Cleanup", "2026-01-01T00:00:00")
    await SourceFactory.build(
        "list-cleanup",
        source_id=source_id,
        video_id="BVlive",
        title="Live Source",
        cover_url="https://example.com/cover.jpg",
        source_url="https://bilibili.com/video/BVlive",
        duration_seconds=10,
        uploader="u",
        language="en",
        whisper_model="m",
        ai_model="m",
    )


@pytest.mark.asyncio
async def test_cleanup_preserves_live_source_artifacts(setup_pipeline_test: Path):
    """A failed/non-done job must NOT purge artifacts of a source that already persisted.

    Reproduces the persist-then-fail window: the source write committed but the job
    raised before reaching DONE. cleanup_job_artifacts must leave the live
    source's cover, embeddings and FTS intact.
    """
    from bibilab.cleanup import cleanup_job_artifacts

    await bootstrap_db()
    source_id = str(uuid.uuid4())
    await _seed_source(source_id)

    covers = setup_pipeline_test / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    cover = covers / f"{source_id}.jpg"
    cover.write_bytes(b"fake cover")

    job = {
        "id": "job-failed-after-persist",
        "type": "ingest",
        "status": "failed",
        "meta": {"source_id": source_id, "video_id": "BVlive", "list_id": "list-cleanup"},
    }

    embed_calls: list = []
    fts_calls: list = []
    with (
        patch("bibilab.cleanup.clear_embeddings_for_source", lambda *a, **k: embed_calls.append(a)),
        patch("bibilab.cleanup.clear_fts_for_source_sync", lambda *a, **k: fts_calls.append(a)),
    ):
        cleanup_job_artifacts(job)

    assert cover.exists(), "live source's cover must survive cleanup of a failed job"
    assert embed_calls == [], "must not clear embeddings for a persisted source"
    assert fts_calls == [], "must not clear FTS for a persisted source"


@pytest.mark.asyncio
async def test_cleanup_removes_orphan_cover_when_no_source(setup_pipeline_test: Path):
    """Pre-persist failure (no source row): the orphan cover is still purged."""
    from bibilab.cleanup import cleanup_job_artifacts

    await bootstrap_db()
    source_id = str(uuid.uuid4())

    covers = setup_pipeline_test / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    cover = covers / f"{source_id}.jpg"
    cover.write_bytes(b"fake cover")

    job = {
        "id": "job-failed-before-persist",
        "type": "ingest",
        "status": "failed",
        "meta": {"source_id": source_id, "video_id": "BVorphan", "list_id": "list-x"},
    }

    with (
        patch("bibilab.cleanup.clear_embeddings_for_source", lambda *a, **k: None),
        patch("bibilab.cleanup.clear_fts_for_source_sync", lambda *a, **k: None),
    ):
        cleanup_job_artifacts(job)

    assert not cover.exists(), "orphan cover (no source row) must be purged"


@pytest.mark.asyncio
async def test_cleanup_clears_embeddings_with_correct_signature(setup_pipeline_test: Path):
    """cleanup_job_artifacts must call clear_embeddings_for_source(source_id) with one arg.

    Regression: a previous version called clear_embeddings_for_source(source_id, load_config()),
    but the production function only accepts source_id. The worker's outer except swallowed the
    TypeError, which left orphan embeddings AND skipped the FTS cleanup on the next line.
    The spec=... mock below enforces the real function's signature so any future extra-arg
    regression fails loud.
    """
    from bibilab.cleanup import cleanup_job_artifacts
    from bibilab.pipeline.embed import clear_embeddings_for_source

    await bootstrap_db()
    source_id = str(uuid.uuid4())

    covers = setup_pipeline_test / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    (covers / f"{source_id}.jpg").write_bytes(b"orphan cover")

    job = {
        "id": "job-partial-ingest",
        "type": "ingest",
        "status": "failed",
        "meta": {"source_id": source_id, "video_id": "BVpartial", "list_id": "list-x"},
    }

    embed_calls: list = []
    fts_calls: list = []

    def _record_embed(source_id_arg):
        embed_calls.append(source_id_arg)

    with (
        patch(
            "bibilab.cleanup.clear_embeddings_for_source",
            MagicMock(spec=clear_embeddings_for_source, side_effect=_record_embed),
        ),
        patch("bibilab.cleanup.clear_fts_for_source_sync", lambda *a, **k: fts_calls.append(a)),
    ):
        cleanup_job_artifacts(job)

    assert embed_calls == [source_id], (
        f"clear_embeddings_for_source must be called with one arg, got calls: {embed_calls!r}"
    )
    assert fts_calls, "clear_fts_for_source_sync must be reached — if it was skipped, an earlier call raised"

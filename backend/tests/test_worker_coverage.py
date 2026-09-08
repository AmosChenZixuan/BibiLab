"""Tests for worker.py uncovered paths: _download_cover, _download_model_job,
_run_job dispatch/exception handling, _run_artifact_job error paths, start/stop."""

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bibilab.db import bootstrap_db, create_list, parse_job_meta
from bibilab.worker import WorkerLoop, _download_cover
from tests.factories import SourceFactory

pytestmark = pytest.mark.integration

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
# parse_job_meta
# ---------------------------------------------------------------------------


class TestParseJobMeta:
    def test_dict_meta(self):
        assert parse_job_meta({"meta": {"key": "val"}}) == {"key": "val"}

    def test_string_meta(self):
        assert parse_job_meta({"meta": '{"key": "val"}'}) == {"key": "val"}

    def test_empty_string_meta(self):
        assert parse_job_meta({"meta": ""}) == {}

    def test_missing_meta(self):
        assert parse_job_meta({}) == {}


# ---------------------------------------------------------------------------
# _download_model_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["large-v3", "cam++"])
async def test_download_model_job_success(model_name: str, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    meta = {"model_name": model_name}
    job_id = await create_job("model_download", meta)

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {"id": job_id, "type": "model_download", "meta": json.dumps(meta)}

    with patch("bibilab.worker.ensure") as mock_ensure:
        await worker._download_model_job(job)
        mock_ensure.assert_called_once_with(model_name)


@pytest.mark.asyncio
async def test_download_model_job_unknown_model(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    meta = {"model_name": "garbage"}
    job_id = await create_job("model_download", meta)

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "model_download", "meta": json.dumps(meta)}

    await worker._run_job(job)

    from bibilab.db import get_db

    async with get_db() as db:
        cursor = await db.execute("SELECT status, error FROM jobs WHERE id=?", (job_id,))
        row = await cursor.fetchone()
    assert row["status"] == "failed"
    assert "Unknown model" in row["error"]


# ---------------------------------------------------------------------------
# _run_job dispatch and exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_frees_slot_without_waiting_for_stage(tmp_bibilab_home: Path):
    """cancel_job frees the slot without waiting for the running stage.

    Deliberately does not register the task in worker._tasks, so cancel_job
    only exercises the _in_flight-discard path here, not the task.cancel()
    path added later — that one has its own coverage in
    test_cancel_job_cancels_running_task_mid_stage."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    started, release = asyncio.Event(), asyncio.Event()

    async def _long_stage(job):
        started.set()
        await release.wait()  # stands in for a long asyncio.to_thread stage

    job = {"id": job_id, "type": "ingest", "meta": "{}"}
    with patch.object(worker, "_pipeline", _long_stage):
        worker._in_flight.add(job_id)
        task = asyncio.create_task(worker._run_job(job))
        await started.wait()

        worker.cancel_job(job_id)
        assert job_id not in worker._in_flight

        release.set()
        await task


@pytest.mark.asyncio
async def test_cancel_job_cancels_running_task_mid_stage(tmp_bibilab_home: Path):
    """cancel_job now cancels the tracked task itself, unwinding an await
    that lands inside a stage — not just at an old between-stage checkpoint,
    which is the behaviour the old polling mechanism could never provide.
    Verifies the collapsed CancelledError handler purges with the full job
    dict, deletes the row, and clears every bookkeeping set."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    never_set = asyncio.Event()
    cleanup_calls = []

    async def _blocked_pipeline(job):
        await never_set.wait()

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with (
        patch.object(worker, "_pipeline", _blocked_pipeline),
        patch("bibilab.worker.cleanup_job_artifacts", side_effect=cleanup_calls.append),
    ):
        task = asyncio.create_task(worker._run_job(job))
        worker._tasks[job_id] = task
        await asyncio.sleep(0)  # let _run_job reach the blocked await

        worker.cancel_job(job_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    assert cleanup_calls == [job]
    assert await get_job(job_id) is None
    assert job_id not in worker._in_flight
    assert job_id not in worker._tasks


@pytest.mark.asyncio
async def test_cancel_handler_reraises_even_if_cleanup_fails(tmp_bibilab_home: Path):
    """A disk or DB hiccup while cleaning up a cancelled job (Windows file
    lock, a busy SQLite connection) must not swallow the CancelledError —
    that would leave `raise` unreachable and the row stuck in a
    non-terminal status until reset_stuck_jobs() requeues it. The task must
    still end up cancelled, and bookkeeping still gets cleared, even though
    cleanup itself failed."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    never_set = asyncio.Event()

    async def _blocked_pipeline(job):
        await never_set.wait()

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with (
        patch.object(worker, "_pipeline", _blocked_pipeline),
        patch("bibilab.worker.cleanup_job_artifacts", side_effect=OSError("mock EBUSY")),
    ):
        task = asyncio.create_task(worker._run_job(job))
        worker._tasks[job_id] = task
        await asyncio.sleep(0)

        worker.cancel_job(job_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    assert job_id not in worker._in_flight
    assert job_id not in worker._tasks
    assert await get_job(job_id) is None


@pytest.mark.asyncio
async def test_cancel_handler_reraises_even_if_delete_fails(tmp_bibilab_home: Path):
    """Mirror of test_cancel_handler_reraises_even_if_cleanup_fails: the
    handler's two cleanup steps are independent try/excepts, so a failure in
    delete_job (the second step) must not swallow the CancelledError either,
    and cleanup_job_artifacts must still have been attempted."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    never_set = asyncio.Event()

    async def _blocked_pipeline(job):
        await never_set.wait()

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    with (
        patch.object(worker, "_pipeline", _blocked_pipeline),
        patch("bibilab.worker.cleanup_job_artifacts") as mock_cleanup,
        patch("bibilab.worker.delete_job", side_effect=OSError("mock db busy")),
    ):
        task = asyncio.create_task(worker._run_job(job))
        worker._tasks[job_id] = task
        await asyncio.sleep(0)

        worker.cancel_job(job_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    mock_cleanup.assert_called_once_with(job)
    assert job_id not in worker._in_flight
    assert job_id not in worker._tasks


@pytest.mark.asyncio
async def test_cancel_job_is_idempotent_and_safe_for_unknown_jobs(tmp_bibilab_home: Path):
    """Cancelling a job with no tracked task (never started, or already
    finished) is a no-op — no KeyError. Covers double-cancel too."""
    worker = WorkerLoop(home=tmp_bibilab_home)

    worker.cancel_job("never-existed")
    worker.cancel_job("never-existed")  # double-cancel

    assert "never-existed" not in worker._tasks


@pytest.mark.asyncio
async def test_cancel_job_on_non_ingest_job_purges_nothing(tmp_bibilab_home: Path):
    """Cancelling a model_download/artifact/digest job goes through the same
    collapsed CancelledError handler; cleanup_job_artifacts no-ops for
    non-ingest types, so nothing gets purged."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job, get_job

    # video_id is deliberately present in meta: if the JobType gate in
    # cleanup_job_artifacts were ever removed, this would give purge_download_files
    # something to act on — proving the gate (not an absent video_id) is what
    # keeps this test's purge_calls empty.
    meta = {"model_name": "x", "video_id": "BVshouldnotpurge"}
    await bootstrap_db()
    job_id = await create_job("model_download", meta)

    never_set = asyncio.Event()
    purge_calls = []

    async def _blocked_download(job):
        await never_set.wait()

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)
    job = {"id": job_id, "type": "model_download", "meta": json.dumps(meta)}

    with (
        patch.object(worker, "_download_model_job", _blocked_download),
        patch("bibilab.cleanup.purge_download_files", side_effect=purge_calls.append),
    ):
        task = asyncio.create_task(worker._run_job(job))
        worker._tasks[job_id] = task
        await asyncio.sleep(0)

        worker.cancel_job(job_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    assert purge_calls == []
    assert await get_job(job_id) is None


@pytest.mark.asyncio
async def test_cancel_job_on_digest_job_leaves_sections_unwritten(tmp_bibilab_home: Path, mock_call_llm):
    """A digest (rerun) job cancelled mid-LLM-call goes through the same
    collapsed CancelledError handler as ingest: the row is deleted, and since
    the cancel lands before either of the two post-LLM writes, the section's
    prior summary/keywords are left exactly as they were — no partial write."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job, create_list, get_job, get_sections, write_source_with_segments
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment
    from tests import thread_signal

    await bootstrap_db()
    await create_list("list-digest-cancel", "Digest Cancel Test", "2025-01-01T00:00:00Z")
    source_id = "src-digest-cancel-001"
    segs = [WhisperSegment(start=0.0, end=5.0, text="test transcript text", speaker=None)]
    secs = [Section(seg_start=0, seg_end=0, token_count=5, timestamp_start=0.0, timestamp_end=5.0)]
    await write_source_with_segments(
        segments=segs,
        sections=secs,
        section_digests=[SectionDigest(summary="old section", keywords=["old"])],
        source_id=source_id,
        video_id="BVdigestcancel001",
        platform="bilibili",
        list_id="list-digest-cancel",
        title="Digest Cancel Test",
        cover_url=None,
        source_url="https://bilibili.com/video/BVdigestcancel001",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        settings_snapshot={},
    )

    job_id = await create_job("digest", {"source_id": source_id, "list_id": "list-digest-cancel", "ui_lang": None})
    job = {"id": job_id, "type": "digest", "meta": json.dumps({"source_id": source_id, "ui_lang": None})}

    started, release, signal_started = thread_signal()

    def _blocking_llm(*a, **k):
        signal_started()
        release.wait()
        return (
            '{"summary": "new summary", "keywords": ["new"], '
            '"series_name": null, "sequence_number": null, "season_number": null}'
        )

    mock_call_llm.side_effect = _blocking_llm

    worker = WorkerLoop(home=tmp_bibilab_home)
    worker._in_flight.add(job_id)

    task = asyncio.create_task(worker._run_job(job))
    worker._tasks[job_id] = task
    await started.wait()

    worker.cancel_job(job_id)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert await get_job(job_id) is None
    rows = await get_sections(source_id)
    assert rows[0]["summary"] == "old section"


@pytest.mark.asyncio
async def test_cancel_job_cancels_running_artifact_job_mid_section_view(tmp_bibilab_home: Path):
    """Real task.cancel() mid-_run_artifact_job (blocked inside
    _build_section_views) unwinds through the same collapsed CancelledError
    handler as ingest/digest/model_download."""
    import asyncio

    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    await create_list("list-artifact-cancel", "Artifact Cancel Test", "2025-01-01T00:00:00Z")
    artifact_id = str(uuid.uuid4())
    meta = {
        "artifact_id": artifact_id,
        "list_id": "list-artifact-cancel",
        "type": "brief",
        "prompt": "Summarize",
        "source_ids": ["some-source"],
    }
    job_id = await create_job("artifact", meta)
    job = {"id": job_id, "type": "artifact", "meta": json.dumps(meta)}

    started = asyncio.Event()
    never_set = asyncio.Event()

    async def _blocked_section_views(source_ids):
        started.set()
        await never_set.wait()

    worker = WorkerLoop(config=MagicMock(), home=tmp_bibilab_home)
    worker._in_flight.add(job_id)

    with (
        patch("bibilab.worker._build_section_views", _blocked_section_views),
        patch("bibilab.worker.cleanup_job_artifacts") as mock_cleanup,
    ):
        task = asyncio.create_task(worker._run_job(job))
        worker._tasks[job_id] = task
        await started.wait()

        worker.cancel_job(job_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    mock_cleanup.assert_called_once_with(job)
    assert await get_job(job_id) is None
    assert job_id not in worker._in_flight
    assert job_id not in worker._tasks


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
async def test_run_job_no_speech_sets_stage_prefixed_error(tmp_bibilab_home: Path):
    """A speech-less video ends FAILED with the user-visible stage-prefixed
    message, exercising the guard through the real _run_job → _pipeline path."""
    from bibilab.db import bootstrap_db, create_job, get_job

    await bootstrap_db()
    await create_list("list-ns", "NoSpeech", "2026-01-01T00:00:00")

    (tmp_bibilab_home / "downloads").mkdir(parents=True, exist_ok=True)
    tmp_video = tmp_bibilab_home / "downloads" / "BVnospeech.mp4"
    tmp_video.write_bytes(b"fake video")
    tmp_wav = tmp_bibilab_home / "downloads" / "BVnospeech.wav"
    tmp_wav.write_bytes(b"fake wav")

    meta = {
        "source_id": str(uuid.uuid4()),
        "video_id": "BVnospeech",
        "list_id": "list-ns",
        "title": "Music Only",
        "platform": "bilibili",
        "source_url": "https://bilibili.com/video/BVnospeech",
        "cover_url": "",
        "duration_seconds": 100,
        "uploader": "u",
        "ui_lang": "en",
    }
    job_id = await create_job("ingest", meta)
    job = {"id": job_id, "type": "ingest", "meta": json.dumps(meta)}

    mock_adapter = MagicMock()
    mock_adapter.download = AsyncMock(return_value=tmp_video)
    worker = WorkerLoop(concurrency=1, adapter=mock_adapter, home=tmp_bibilab_home)
    worker._in_flight.add(job_id)

    with (
        patch("bibilab.worker.extract_audio", MagicMock(return_value=tmp_wav)),
        patch("bibilab.worker.transcribe", MagicMock(return_value=([], None))),
    ):
        await worker._run_job(job)

    row = await get_job(job_id)
    assert row["status"] == "failed"
    assert row["error"] == "[transcribing] no speech detected in audio"


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
        artifact_row = await cursor.fetchone()
        cursor = await db.execute("SELECT status, error FROM jobs WHERE id=?", (job_id,))
        job_row = await cursor.fetchone()
    assert artifact_row is None
    assert job_row["status"] == "failed"
    assert "not found" in job_row["error"]


# ---------------------------------------------------------------------------
# WorkerLoop start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_transcribe_punctuates_and_returns_sentences(tmp_bibilab_home: Path, monkeypatch):
    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    vad = [WhisperSegment(start=0.0, end=5.0, text="天花板明显是地板", speaker="SPK_0")]
    sentences = [
        WhisperSegment(start=0.0, end=5.0, text="天花板。", speaker="SPK_0"),
        WhisperSegment(start=0.0, end=5.0, text="明显是地板。", speaker="SPK_0"),
    ]

    monkeypatch.setattr("bibilab.worker.transcribe", lambda *a, **k: (vad, "zh"))
    called = {}

    def _fake_punctuate(segs, language):
        called["language"] = language
        return sentences

    monkeypatch.setattr("bibilab.worker.punctuate", _fake_punctuate)

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    sentence_segments = await loop._stage_transcribe(
        {"id": job_id, "type": "ingest", "meta": {}}, wav, "src-1", BibilabConfig()
    )

    assert called["language"] == "zh"
    assert sentence_segments == sentences


@pytest.mark.asyncio
async def test_stage_transcribe_trusts_detected_language_over_config(tmp_bibilab_home: Path, monkeypatch):
    """large-v3 always decodes English regardless of cfg.transcription.language
    (see transcribe.py), so transcribe() reports detected_language="en" even when
    the user explicitly configured "zh". _stage_transcribe must not re-derive
    effective_language from cfg — that would route English text through the
    zh-gated ct-punc path. Observable only via what punctuate() receives —
    effective_language isn't part of the return value."""
    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    vad = [WhisperSegment(start=0.0, end=5.0, text="hello world", speaker="SPK_0")]
    monkeypatch.setattr("bibilab.worker.transcribe", lambda *a, **k: (vad, "en"))
    called = {}

    def _fake_punctuate(segs, language):
        called["language"] = language
        return segs

    monkeypatch.setattr("bibilab.worker.punctuate", _fake_punctuate)

    cfg = BibilabConfig()
    cfg.transcription.model = "large-v3"
    cfg.transcription.language = "zh"

    loop = WorkerLoop(config=cfg, home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    await loop._stage_transcribe({"id": job_id, "type": "ingest", "meta": {}}, wav, "src-1", cfg)

    assert called["language"] == "en"


@pytest.mark.asyncio
async def test_stage_transcribe_none_detected_language_degrades_safely(tmp_bibilab_home: Path, monkeypatch):
    """auto mode + failed detection (detected_language=None) must not crash:
    effective_language stays None, punctuate() skips (non-"zh" gate) instead
    of receiving a fabricated "en". Observable via what punctuate() receives."""
    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    vad = [WhisperSegment(start=0.0, end=5.0, text="unrecognized", speaker="SPK_0")]
    monkeypatch.setattr("bibilab.worker.transcribe", lambda *a, **k: (vad, None))
    called = {}

    def _fake_punctuate(segs, language):
        called["language"] = language
        return segs

    monkeypatch.setattr("bibilab.worker.punctuate", _fake_punctuate)

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    await loop._stage_transcribe({"id": job_id, "type": "ingest", "meta": {}}, wav, "src-1", BibilabConfig())

    assert called["language"] is None


@pytest.mark.asyncio
async def test_stage_transcribe_no_speech_raises_pipeline_error(tmp_bibilab_home: Path, monkeypatch):
    """A speech-less video (empty transcription) fails loud with a clear message
    instead of crashing downstream with IndexError in digest."""
    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job
    from bibilab.pipeline.audio import PipelineError

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    monkeypatch.setattr("bibilab.worker.transcribe", lambda *a, **k: ([], None))
    monkeypatch.setattr("bibilab.worker.punctuate", lambda segs, language: [])

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")

    with pytest.raises(PipelineError, match="no speech detected"):
        await loop._stage_transcribe({"id": job_id, "type": "ingest", "meta": {}}, wav, "src-1", BibilabConfig())


@pytest.mark.asyncio
async def test_stage_transcribe_cancel_wins_over_no_speech(tmp_bibilab_home: Path, monkeypatch):
    """A job cancelled while decoding ends as a clean cancel (job deleted), not
    a FAILED no-speech error — even though the segments would come back empty.
    The old between-stage checkpoint that made this true is gone; the
    property now holds because task.cancel() unwinds the await before the
    no-speech raise ever runs."""
    import asyncio

    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job, get_job
    from tests import thread_signal

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    started, release, signal_started = thread_signal()

    def _blocking_transcribe(*a, **k):
        signal_started()
        release.wait()
        return [], None

    monkeypatch.setattr("bibilab.worker.transcribe", _blocking_transcribe)

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    async def _pipeline_stub(_job):
        await loop._stage_transcribe(_job, wav, "src-1", BibilabConfig())

    with patch.object(loop, "_pipeline", _pipeline_stub):
        task = asyncio.create_task(loop._run_job(job))
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert await get_job(job_id) is None


@pytest.mark.asyncio
async def test_stage_transcribe_cancel_wins_while_punctuate_runs(tmp_bibilab_home: Path, monkeypatch):
    """Same guarantee as the test above, extended past transcribe(): a cancel
    while punctuate() is still in flight also wins over the no-speech raise.

    This does not by itself prove the narrower gap the awaiting `asyncio.sleep(0)`
    in _stage_transcribe exists to close — a cancel requested strictly after
    punctuate() has already returned, in the zero-width synchronous stretch
    before the no-speech check, unwinds at that sleep(0) rather than here.
    That specific window has no yield point to hang a deterministic test off
    of without faking asyncio's own scheduling; its correctness rests on
    asyncio's documented guarantee that a pending cancellation is delivered
    at the very next checkpoint, which the sleep(0) provides."""
    import asyncio

    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job, get_job
    from tests import thread_signal

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    started, release, signal_started = thread_signal()

    monkeypatch.setattr("bibilab.worker.transcribe", lambda *a, **k: ([object()], None))

    def _blocking_punctuate(*a, **k):
        signal_started()
        release.wait()
        return []

    monkeypatch.setattr("bibilab.worker.punctuate", _blocking_punctuate)

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    async def _pipeline_stub(_job):
        await loop._stage_transcribe(_job, wav, "src-1", BibilabConfig())

    with patch.object(loop, "_pipeline", _pipeline_stub):
        task = asyncio.create_task(loop._run_job(job))
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert await get_job(job_id) is None


@pytest.mark.asyncio
async def test_stage_transcribe_sets_cancel_flag_on_task_cancel(tmp_bibilab_home: Path, monkeypatch):
    """_stage_transcribe must set the cancel Event it passes to transcribe()
    when the awaiting task is cancelled, so the orphaned decode thread's span
    loop can stop promptly instead of running to the end of the audio."""
    import asyncio

    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db, create_job
    from tests import thread_signal

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    started, release, signal_started = thread_signal()
    recorded = {}

    def _blocking_transcribe(wav_path, cfg, cancel=None):
        recorded["cancel"] = cancel
        signal_started()
        release.wait()
        return [], None

    monkeypatch.setattr("bibilab.worker.transcribe", _blocking_transcribe)

    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    wav = tmp_bibilab_home / "a.wav"
    wav.write_bytes(b"")
    job = {"id": job_id, "type": "ingest", "meta": "{}"}

    async def _pipeline_stub(_job):
        await loop._stage_transcribe(_job, wav, "src-1", BibilabConfig())

    with patch.object(loop, "_pipeline", _pipeline_stub):
        task = asyncio.create_task(loop._run_job(job))
        await started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert recorded["cancel"] is not None
    assert recorded["cancel"].is_set()


@pytest.mark.asyncio
async def test_stage_process_chunks_sentence_segments(tmp_bibilab_home: Path, monkeypatch):
    """_stage_process chunks sentence_segments, not vad_segments."""
    from bibilab.config import BibilabConfig
    from bibilab.db import bootstrap_db
    from bibilab.pipeline.transcribe import WhisperSegment
    from bibilab.worker import WorkerLoop

    await bootstrap_db()

    sentences = [WhisperSegment(start=0.0, end=2.0, text="第一句。", speaker="SPK_0")]
    captured: dict = {}

    def _fake_chunk(segs, *args, **kw):
        captured["segs"] = segs
        return []

    monkeypatch.setattr("bibilab.worker.chunk_by_sections", _fake_chunk)
    monkeypatch.setattr("bibilab.worker.embed_chunks", lambda *a, **k: None)
    from bibilab.pipeline.digest import DigestResult, SectionDigest

    monkeypatch.setattr(
        "bibilab.worker.digest_sections",
        lambda *a, **k: (
            DigestResult(summary="s", keywords=[], series_name=None, sequence_number=None, season_number=None),
            [SectionDigest(summary="s", keywords=[])],
        ),
    )
    loop = WorkerLoop(config=BibilabConfig(), home=tmp_bibilab_home)
    result = await loop._stage_process(
        job={"id": "j", "meta": {}},
        sentence_segments=sentences,
        source_id="src-1",
        video_meta=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        list_id="l",
        cfg=BibilabConfig(),
    )
    assert captured["segs"] is sentences
    assert result is not None and len(result) == 3  # (extraction, sections, section_digests) tuple


@pytest.mark.asyncio
async def test_stage_persist_atomic_no_orphan_on_segment_write_failure(tmp_bibilab_home: Path):
    """Source + segments persist in one transaction. A segment-write failure rolls
    the source upsert back too — no orphaned source row (atomicity, not compensation)."""
    import bibilab.db as db
    from bibilab.db import bootstrap_db, create_job, create_list, get_source
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "Test List", "2026-01-01T00:00:00")
    job_id = await create_job("ingest", {})

    sentences = [WhisperSegment(start=0.0, end=1.0, text="test。", speaker=None)]

    loop = WorkerLoop(home=tmp_bibilab_home)
    video_meta = MagicMock(
        platform="bilibili", title="T", cover_url=None, source_url="url", duration_seconds=1, uploader="U"
    )

    async def _boom(*args, **kwargs):
        raise Exception("disk full")

    with patch.object(db, "_exec_write_transcript_segments", _boom):
        with pytest.raises(Exception, match="disk full"):
            await loop._stage_persist(
                job_id=job_id,
                source_id="orphan-src",
                video_id="BVorphan",
                video_meta=video_meta,
                list_id="list-1",
                extraction=MagicMock(
                    summary="s", keywords=[], series_name=None, sequence_number=None, season_number=None
                ),
                sections=[],
                section_digests=[],
                cfg=MagicMock(
                    transcription=MagicMock(model="base"),
                    ai=MagicMock(model="gpt"),
                    model_dump=lambda: {},
                ),
                sentence_segments=sentences,
            )

    # The source upsert rolled back with the failed segment write — no orphan
    assert await get_source("orphan-src") is None


@pytest.mark.asyncio
async def test_worker_start_stop(tmp_bibilab_home: Path):
    await bootstrap_db()
    worker = WorkerLoop(home=tmp_bibilab_home)
    await worker.start()
    assert worker._running is True
    assert worker._task is not None
    await worker.stop()
    assert worker._running is False


# ---------------------------------------------------------------------------
# _dispatch_pending — the queued-job dispatch race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_pending_claims_and_runs_a_real_queued_job(tmp_bibilab_home: Path):
    """The ordinary path: a genuinely still-queued job is claimed and gets a
    tracked task."""
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("ingest", {})

    worker = WorkerLoop(home=tmp_bibilab_home)
    ran = []

    async def _fake_run_job(job):
        ran.append(job["id"])

    with patch.object(worker, "_run_job", _fake_run_job):
        await worker._dispatch_pending()
        assert job_id in worker._tasks
        await worker._tasks[job_id]

    assert ran == [job_id]


@pytest.mark.asyncio
async def test_dispatch_pending_skips_a_job_deleted_between_read_and_claim(tmp_bibilab_home: Path):
    """A job the router already deleted (cancelled while still queued)
    between _dispatch_pending's read and its claim step must not be
    dispatched — this is the race claim_queued_job exists to close.
    Simulated directly: the pending snapshot names a job id that was never
    actually written to the DB, standing in for one deleted out from under
    the loop after the read but before the claim."""
    await bootstrap_db()
    worker = WorkerLoop(home=tmp_bibilab_home)
    ghost_job = {"id": "job-ghost", "status": "queued", "type": "ingest", "meta": "{}"}

    with patch("bibilab.worker.get_pending_jobs", return_value=[ghost_job]):
        await worker._dispatch_pending()

    assert "job-ghost" not in worker._in_flight
    assert "job-ghost" not in worker._tasks


# ---------------------------------------------------------------------------
# _run_digest_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_digest_job_success(tmp_bibilab_home: Path, mock_call_llm):
    from bibilab.db import bootstrap_db, create_job, create_list, write_source_with_segments
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-digest", "Digest Test", "2025-01-01T00:00:00Z")
    source_id = "src-digest-001"
    segs = [WhisperSegment(start=0.0, end=5.0, text="test transcript text", speaker=None)]
    secs = [Section(seg_start=0, seg_end=0, token_count=5, timestamp_start=0.0, timestamp_end=5.0)]
    await write_source_with_segments(
        segments=segs,
        sections=secs,
        section_digests=[SectionDigest(summary="old section", keywords=["old"])],
        source_id=source_id,
        video_id="BVdigest001",
        platform="bilibili",
        list_id="list-digest",
        title="Digest Test",
        cover_url=None,
        source_url="https://bilibili.com/video/BVdigest001",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        settings_snapshot={},
    )

    job_id = await create_job("digest", {"source_id": source_id, "list_id": "list-digest", "ui_lang": None})

    mock_call_llm.return_value = (
        '{"summary": "new summary", "keywords": ["new"], '
        '"series_name": null, "sequence_number": null, "season_number": null}'
    )

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {
        "id": job_id,
        "type": "digest",
        "meta": json.dumps({"source_id": source_id, "list_id": "list-digest"}),
    }
    await worker._run_digest_job(job)

    from bibilab.db import get_sections

    sections = await get_sections(source_id)
    assert sections[0]["summary"] == "new summary"

    from bibilab.db import get_job

    row = await get_job(job_id)
    assert dict(row)["status"] == "done"


@pytest.mark.asyncio
async def test_run_digest_job_source_not_found(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job

    await bootstrap_db()
    job_id = await create_job("digest", {"source_id": "nonexistent", "list_id": "list-digest"})

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {"id": job_id, "type": "digest", "meta": json.dumps({"source_id": "nonexistent", "list_id": "list-digest"})}
    await worker._run_digest_job(job)

    from bibilab.db import get_job

    row = await get_job(job_id)
    assert dict(row)["status"] == "failed"
    assert "not found" in dict(row)["error"]


@pytest.mark.asyncio
async def test_run_digest_job_no_transcript(tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, create_list

    await bootstrap_db()
    await create_list("list-no-transcript", "No Transcript", "2025-01-01T00:00:00Z")
    source_id = "src-no-transcript"
    await SourceFactory.build(
        "list-no-transcript",
        source_id=source_id,
        video_id="BVnoTrans",
        title="No Transcript",
        source_url="https://bilibili.com/video/BVnoTrans",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
    )
    # Note: no write_transcript_segments call — source has no transcript

    job_id = await create_job("digest", {"source_id": source_id, "list_id": "list-no-transcript"})

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {
        "id": job_id,
        "type": "digest",
        "meta": json.dumps({"source_id": source_id, "list_id": "list-no-transcript"}),
    }
    await worker._run_digest_job(job)

    from bibilab.db import get_job

    row = await get_job(job_id)
    assert dict(row)["status"] == "failed"
    assert "no transcript" in dict(row)["error"]


@pytest.mark.asyncio
async def test_run_digest_job_llm_failure(tmp_bibilab_home: Path, mock_call_llm):
    from bibilab.db import bootstrap_db, create_job, create_list, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-llm-fail", "LLM Fail", "2025-01-01T00:00:00Z")
    source_id = "src-llm-fail"
    await SourceFactory.build(
        "list-llm-fail",
        source_id=source_id,
        video_id="BVllmFail",
        title="LLM Fail",
        source_url="https://bilibili.com/video/BVllmFail",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
    )
    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=5.0, text="test transcript", speaker=None)]
    )

    job_id = await create_job("digest", {"source_id": source_id, "list_id": "list-llm-fail"})

    mock_call_llm.side_effect = ValueError("LLM error")

    worker = WorkerLoop(home=tmp_bibilab_home)
    job = {"id": job_id, "type": "digest", "meta": json.dumps({"source_id": source_id, "list_id": "list-llm-fail"})}
    await worker._run_digest_job(job)

    from bibilab.db import get_job

    row = await get_job(job_id)
    assert dict(row)["status"] == "failed"


# ---------------------------------------------------------------------------
# Download stage: .part hygiene + concurrency cap
# ---------------------------------------------------------------------------


def _video_meta(video_id: str):
    from bibilab.adapters.base import VideoMeta

    return VideoMeta(
        video_id=video_id,
        title="t",
        platform="bilibili",
        source_url=f"https://www.bilibili.com/video/{video_id}",
        cover_url="https://example.com/c.jpg",
        duration_seconds=100,
        uploader="u",
    )


class TestDownloadHygieneAndCap:
    @pytest.mark.asyncio
    async def test_purges_stale_part_before_download(self, tmp_bibilab_home: Path, downloads_dir: Path):
        """A stale .part from a prior attempt must be gone before the new download
        runs, so yt-dlp never resumes onto corrupt bytes."""
        await bootstrap_db()
        stale = downloads_dir / "BVstale.m4a.part"
        stale.write_bytes(b"old partial")

        seen = {}
        final = downloads_dir / "BVstale.m4a"

        async def fake_download(video_id: str, source_url: str, connections: int):
            seen["stale_existed_at_download"] = stale.exists()
            final.write_bytes(b"new audio")
            return final

        adapter = MagicMock()
        adapter.download = fake_download
        worker = WorkerLoop(adapter=adapter, home=tmp_bibilab_home)

        with patch("bibilab.worker._download_cover", MagicMock(return_value=True)):
            result = await worker._stage_download({"id": "job-x"}, _video_meta("BVstale"), "src-1")

        assert seen["stale_existed_at_download"] is False
        assert result == final

    @pytest.mark.asyncio
    async def test_no_download_only_cap(self, tmp_bibilab_home: Path, downloads_dir: Path):
        """The download-only semaphore is gone — multiple concurrent calls to
        _stage_download are bounded only by the outer job-concurrency gate
        (which this test bypasses by calling _stage_download directly)."""
        import asyncio

        await bootstrap_db()

        active = 0
        peak = 0

        async def fake_download(video_id: str, source_url: str, connections: int):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            p = downloads_dir / f"{video_id}.m4a"
            p.write_bytes(b"a")
            return p

        adapter = MagicMock()
        adapter.download = fake_download
        worker = WorkerLoop(adapter=adapter, home=tmp_bibilab_home, concurrency=1)

        with patch("bibilab.worker._download_cover", MagicMock(return_value=True)):
            await asyncio.gather(
                *(worker._stage_download({"id": f"job-{i}"}, _video_meta(f"BV{i}"), f"src-{i}") for i in range(3))
            )

        assert peak == 3


# ---------------------------------------------------------------------------
# _reraise_gathered_failures — accurate logging for the digest∥embed gather
# ---------------------------------------------------------------------------


class TestReraiseGatheredFailures:
    def test_no_failures_is_noop(self):
        from bibilab.worker import _reraise_gathered_failures

        _reraise_gathered_failures(("digest", []), None)

    def test_embed_only_reraises_without_secondary_log(self, caplog):
        from bibilab.worker import _reraise_gathered_failures

        err = RuntimeError("embed boom")
        with caplog.at_level("ERROR"):
            with pytest.raises(RuntimeError, match="embed boom"):
                _reraise_gathered_failures(("digest", []), err)
        # Embed is the primary (only) error — no misleading "secondary" log.
        assert "secondary" not in caplog.text

    def test_both_failed_logs_secondary_and_raises_digest(self, caplog):
        from bibilab.worker import _reraise_gathered_failures

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError, match="digest boom"):
                _reraise_gathered_failures(ValueError("digest boom"), RuntimeError("embed boom"))
        assert "secondary" in caplog.text

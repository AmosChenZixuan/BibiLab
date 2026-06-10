"""Tests for worker.py uncovered paths: _download_cover, _download_model_job,
_run_job dispatch/exception handling, _run_artifact_job error paths, start/stop."""

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bibilab.db import bootstrap_db, create_list, parse_job_meta
from bibilab.worker import WorkerLoop, _download_cover
from tests.factories import SourceFactory

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
    result = await loop._stage_transcribe(job_id, wav, "src-1", BibilabConfig())

    detected_language, effective_language, sentence_segments = result
    assert called["language"] == "zh"
    assert sentence_segments == sentences


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
        job_id="j",
        job={"meta": {}},
        sentence_segments=sentences,
        source_id="src-1",
        video_meta=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        list_id="l",
        cfg=BibilabConfig(),
        effective_language="zh",
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
                detected_language="en",
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

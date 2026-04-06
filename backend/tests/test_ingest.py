from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _make_video_info(bvid="BV1abc123", title="Test Video"):
    return {
        "id": bvid,
        "title": title,
        "webpage_url": f"https://www.bilibili.com/video/{bvid}",
        "thumbnail": "https://example.com/cover.jpg",
        "duration": 3600,
        "uploader": "TestUser",
        "ext": "mp4",
    }


@pytest.fixture()
def mock_ydl_single():
    info = _make_video_info()
    mock = MagicMock()
    mock.__enter__ = lambda s: mock
    mock.__exit__ = MagicMock(return_value=False)
    mock.extract_info = MagicMock(return_value=info)
    with patch("bibilab.adapters.bilibili.yt_dlp.YoutubeDL", return_value=mock):
        yield info


@pytest.mark.asyncio
async def test_ingest_single_video(client: httpx.AsyncClient, mock_ydl_single):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/url",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_dedup(client: httpx.AsyncClient, mock_ydl_single, tmp_bibilab_home: Path):
    """Submitting the same video twice should skip on second attempt."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await write_source(
        source_id=str(uuid.uuid4()),
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://www.bilibili.com/video/BV1abc123",
        duration_seconds=0,
        uploader="",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = await client.post(
        "/ingest/url",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["queued"] == []
    assert "BV1abc123" in data["skipped"]


@pytest.mark.asyncio
async def test_ingest_rerun_bypasses_dedup(client: httpx.AsyncClient, mock_ydl_single, tmp_bibilab_home: Path):
    """?rerun=true should queue even if source already exists."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())
    await write_source(
        source_id=source_id,
        video_id="BV1abc123",
        platform="bilibili",
        list_id="list-1",
        title="T",
        summary="S",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://www.bilibili.com/video/BV1abc123",
        duration_seconds=0,
        uploader="",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    resp = await client.post(
        "/ingest/url?rerun=true",
        json={
            "list_id": "list-1",
            "url": "https://www.bilibili.com/video/BV1abc123",
            "source_id": source_id,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_unknown_list(client: httpx.AsyncClient, mock_ydl_single):
    resp = await client.post(
        "/ingest/url",
        json={"list_id": "nonexistent", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ingest_rerun_endpoint_removed(client: httpx.AsyncClient):
    """The old /ingest/rerun/{video_id} endpoint must no longer exist."""
    resp = await client.post("/ingest/rerun/BV1abc123")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rerun_digest_only(tmp_bibilab_home: Path):
    """stages=["digest"] with rerun=True re-runs LLM only; skips download/transcribe."""
    import json

    from bibilab.db import bootstrap_db, create_list, get_pending_jobs, get_source, write_source
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    source_id = "digest-rerun-src-001"
    video_id = "BV1abc123"

    # Seed source row with existing transcript file
    (tmp_bibilab_home / "transcripts").mkdir(parents=True, exist_ok=True)
    transcript_file = tmp_bibilab_home / "transcripts" / f"{source_id}.txt"
    transcript_file.write_text("old transcript text", encoding="utf-8")

    await write_source(
        source_id=source_id,
        video_id=video_id,
        platform="bilibili",
        list_id="list-1",
        title="Test Video",
        summary="old summary",
        keywords=["old", "keyword"],
        cover_url="https://example.com/cover.jpg",
        transcript_path=f"transcripts/{source_id}.txt",
        source_url=f"https://bilibili.com/video/{video_id}",
        duration_seconds=3600,
        uploader="TestUser",
        language="en",
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Create a job as if it was queued by the ingest endpoint
    from bibilab.db import create_job

    await create_job(
        type="ingest",
        meta={
            "video_id": video_id,
            "list_id": "list-1",
            "title": "Test Video",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 3600,
            "uploader": "TestUser",
            "rerun": True,
            "source_url": f"https://bilibili.com/video/{video_id}",
            "platform": "bilibili",
            "ui_lang": "en",
            "source_id": source_id,
            "stages": ["digest"],
        },
    )

    new_digest_json = json.dumps(
        {
            "summary": "new summary from LLM",
            "keywords": ["new", "keywords"],
        }
    )

    worker = WorkerLoop(concurrency=1)

    with patch("bibilab.pipeline.digest._call_llm", return_value=new_digest_json):
        jobs = [dict(j) for j in await get_pending_jobs()]
        assert len(jobs) == 1
        await worker._run_job(jobs[0])

    # Verify summary and keywords were updated
    source_row = await get_source(source_id)
    assert source_row is not None
    assert source_row["summary"] == "new summary from LLM"
    assert json.loads(source_row["keywords"]) == ["new", "keywords"]

    # Verify transcript file content unchanged
    assert transcript_file.read_text(encoding="utf-8") == "old transcript text"


@pytest.mark.asyncio
async def test_write_source_stores_relative_paths(tmp_bibilab_home: Path):
    """After write_source, note_path and transcript_path are stored as relative paths."""
    from bibilab.db import bootstrap_db, create_list, get_source, write_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    # Call write_source (the DB write happens here)
    await write_source(
        source_id="test-uuid-1234",
        video_id="BV1relative",
        platform="bilibili",
        list_id="list-1",
        title="Test",
        summary="Test summary.",
        keywords=["test"],
        cover_url="https://example.com/cover.jpg",
        transcript_path="transcripts/BV1relative.txt",
        source_url="https://bilibili.com/video/BV1relative",
        duration_seconds=3600,
        uploader="TestUser",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Read back and verify paths are stored as relative
    row = await get_source("test-uuid-1234")
    assert row["id"] == "test-uuid-1234"
    assert row["transcript_path"] == "transcripts/BV1relative.txt"


@pytest.mark.asyncio
async def test_pipeline_creates_covers_and_transcripts(tmp_bibilab_home: Path):
    """Full pipeline with mocks: covers + transcripts created."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source
    from bibilab.pipeline.chunk import WhisperSegment
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    list_id = "list-pipeline-test"
    await create_list(list_id, "Pipeline Test", "2026-01-01T00:00:00")

    # Create required subdirectories
    (tmp_bibilab_home / "downloads").mkdir(parents=True, exist_ok=True)
    (tmp_bibilab_home / "covers").mkdir(parents=True, exist_ok=True)
    (tmp_bibilab_home / "transcripts").mkdir(parents=True, exist_ok=True)

    # Create a temp video file for adapter.download to return
    tmp_video = tmp_bibilab_home / "downloads" / "BVtest123.mp4"
    tmp_video.write_bytes(b"fake video content")

    # Create the corresponding wav file that extract_audio will create
    tmp_wav = Path(str(tmp_video).replace(".mp4", ".wav"))
    tmp_wav.write_bytes(b"fake wav content")

    mock_segments = [WhisperSegment(start=0.0, end=5.0, text="Hello world test segment.")]

    mock_digest_json = '{"summary": "Test summary.", "keywords": ["test", "keyword"]}'

    job_id = "job-pipeline-test"
    source_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "type": "ingest",
        "meta": {
            "source_id": source_id,
            "video_id": "BVtest123",
            "list_id": list_id,
            "title": "Test Video",
            "platform": "bilibili",
            "source_url": "https://bilibili.com/video/BVtest123",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 3600,
            "uploader": "TestUser",
            "ui_lang": "en",
        },
    }

    worker = WorkerLoop(concurrency=1)

    # Set up mocks before the patch block
    mock_adapter = MagicMock()
    mock_adapter.download = MagicMock(return_value=tmp_video)

    mock_extract_audio = MagicMock(return_value=tmp_wav)

    mock_transcribe_fn = MagicMock(return_value=(mock_segments, "en"))

    mock_call_llm = MagicMock(return_value=mock_digest_json)

    mock_embed = MagicMock()

    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover data") or True)

    with (
        patch("bibilab.worker.BilibiliAdapter", return_value=mock_adapter),
        patch("bibilab.worker.extract_audio", mock_extract_audio),
        patch("bibilab.worker.transcribe", mock_transcribe_fn),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.pipeline.digest._call_llm", mock_call_llm),
        patch("bibilab.worker.embed_chunks", mock_embed),
    ):
        await worker._pipeline(job)

    # Verify cover was downloaded
    assert (tmp_bibilab_home / "covers" / f"{source_id}.jpg").exists(), (
        "cover should be saved as covers/{source_id}.jpg"
    )

    # Verify transcript was written
    assert (tmp_bibilab_home / "transcripts" / f"{source_id}.txt").exists(), (
        "transcript should be saved as transcripts/{source_id}.txt"
    )

    # Verify source row in DB has UUID id
    source_row = await get_source(source_id)
    assert source_row is not None, "source should exist in DB"
    assert source_row["id"] == source_id, "source id should be the generated UUID"
    assert source_row["video_id"] == "BVtest123"
    assert source_row["list_id"] == list_id

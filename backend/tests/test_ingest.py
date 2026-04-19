from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _video_payload(bvid="BV1abc123", title="Test Video", duration=3600):
    return {
        "video_id": bvid,
        "title": title,
        "cover_url": "https://example.com/cover.jpg",
        "duration_seconds": duration,
        "uploader": "TestUser",
        "platform": "bilibili",
        "source_url": f"https://www.bilibili.com/video/{bvid}",
    }


@pytest.mark.asyncio
async def test_ingest_single_video(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/url",
        json={"list_id": list_id, "videos": [_video_payload()]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["queued"]) == 1
    assert data["skipped"] == []


@pytest.mark.asyncio
async def test_ingest_dedup(client: httpx.AsyncClient, tmp_bibilab_home: Path):
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
        json={"list_id": "list-1", "videos": [_video_payload()]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["skipped"] == ["BV1abc123"]
    assert data["queued"] == []


@pytest.mark.asyncio
async def test_ingest_unknown_list(client: httpx.AsyncClient):
    resp = await client.post(
        "/ingest/url",
        json={"list_id": "nonexistent", "videos": [_video_payload()]},
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

    # Set up mocks before the patch block
    mock_adapter = MagicMock()
    mock_adapter.download = MagicMock(return_value=tmp_video)

    worker = WorkerLoop(concurrency=1, adapter=mock_adapter, home=tmp_bibilab_home)

    mock_extract_audio = MagicMock(return_value=tmp_wav)

    mock_transcribe_fn = MagicMock(return_value=(mock_segments, "en"))

    mock_call_llm = MagicMock(return_value=mock_digest_json)

    mock_embed = MagicMock()

    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover data") or True)

    with (
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


@pytest.fixture()
def mock_resolve_single():
    from bibilab.adapters.base import PlaylistMeta, VideoMeta
    from bibilab.adapters.bilibili import BilibiliAdapter

    def _make(video_id="BV1abc123", title="Test Video", part_label=None):
        return VideoMeta(
            video_id=video_id,
            title=title,
            platform="bilibili",
            source_url=f"https://www.bilibili.com/video/{video_id}",
            cover_url="https://example.com/cover.jpg",
            duration_seconds=3600,
            uploader="TestUser",
            part_label=part_label,
        )

    with patch.object(
        BilibiliAdapter,
        "resolve_flat",
        return_value=PlaylistMeta(
            playlist_id="BV1abc123",
            title="Test Video",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BV1abc123",
            videos=[_make()],
        ),
    ) as mock:
        yield mock


@pytest.fixture()
def mock_resolve_multi_part():
    from bibilab.adapters.base import PlaylistMeta, VideoMeta
    from bibilab.adapters.bilibili import BilibiliAdapter

    videos = [
        VideoMeta(
            video_id="BV1abc123_p1",
            title="Part 1",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BV1abc123?p=1",
            cover_url="https://example.com/cover.jpg",
            duration_seconds=600,
            uploader="TestUser",
            part_label="P1",
        ),
        VideoMeta(
            video_id="BV1abc123_p2",
            title="Part 2",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BV1abc123?p=2",
            cover_url="https://example.com/cover.jpg",
            duration_seconds=900,
            uploader="TestUser",
            part_label="P2",
        ),
    ]
    with patch.object(
        BilibiliAdapter,
        "resolve_flat",
        return_value=PlaylistMeta(
            playlist_id="BV1abc123",
            title="Multi-part Video",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BV1abc123",
            videos=videos,
        ),
    ) as mock:
        yield mock


@pytest.fixture()
def mock_resolve_medialist():
    from bibilab.adapters.base import PlaylistMeta, VideoMeta
    from bibilab.adapters.bilibili import BilibiliAdapter

    videos = [
        VideoMeta(
            video_id="BVxyz789",
            title="Medialist Video 1",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BVxyz789",
            cover_url="https://example.com/cover1.jpg",
            duration_seconds=1800,
            uploader="ChannelUser",
            part_label=None,
        ),
        VideoMeta(
            video_id="BVabc123",
            title="Medialist Video 2",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BVabc123",
            cover_url="https://example.com/cover2.jpg",
            duration_seconds=2400,
            uploader="ChannelUser",
            part_label=None,
        ),
    ]
    with patch.object(
        BilibiliAdapter,
        "resolve_flat",
        return_value=PlaylistMeta(
            playlist_id="ml123",
            title="My Medialist",
            platform="bilibili",
            source_url="https://bilibili.com/medialist/ml123",
            videos=videos,
        ),
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_preview_single_video(client: httpx.AsyncClient, mock_resolve_single):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["part_label"] is None
    assert data["videos"][0]["status"] == "new"


@pytest.mark.asyncio
async def test_preview_multi_part_video(client: httpx.AsyncClient, mock_resolve_multi_part):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 2
    assert data["videos"][0]["video_id"] == "BV1abc123_p1"
    assert data["videos"][0]["part_label"] == "P1"
    assert data["videos"][1]["video_id"] == "BV1abc123_p2"
    assert data["videos"][1]["part_label"] == "P2"


@pytest.mark.asyncio
async def test_preview_medialist(client: httpx.AsyncClient, mock_resolve_medialist):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://bilibili.com/medialist/ml123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 2
    for v in data["videos"]:
        assert v["part_label"] is None


@pytest.mark.asyncio
async def test_preview_processed_status(client: httpx.AsyncClient, mock_resolve_single, tmp_bibilab_home: Path):
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
        "/ingest/preview",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["status"] == "processed"


@pytest.mark.asyncio
async def test_preview_in_progress_status(client: httpx.AsyncClient, mock_resolve_single, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, create_list

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await create_job(
        type="ingest",
        meta={
            "video_id": "BV1abc123",
            "list_id": "list-1",
            "title": "Test",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 3600,
            "uploader": "TestUser",
            "source_url": "https://www.bilibili.com/video/BV1abc123",
            "platform": "bilibili",
            "ui_lang": "en",
        },
    )
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_preview_needs_auth_status(client: httpx.AsyncClient, mock_resolve_single, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db, create_job, create_list, update_job_status
    from bibilab.models.jobs import JobStatus

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    job_id = await create_job(
        type="ingest",
        meta={
            "video_id": "BV1abc123",
            "list_id": "list-1",
            "title": "Test",
            "cover_url": "https://example.com/cover.jpg",
            "duration_seconds": 3600,
            "uploader": "TestUser",
            "source_url": "https://www.bilibili.com/video/BV1abc123",
            "platform": "bilibili",
            "ui_lang": "en",
        },
    )
    await update_job_status(job_id, JobStatus.NEEDS_AUTH)
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": "list-1", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["status"] == "needs_auth"


@pytest.mark.asyncio
async def test_preview_unknown_list(client: httpx.AsyncClient, mock_resolve_single):
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": "nonexistent", "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_preview_auth_required(client: httpx.AsyncClient):
    from bibilab.adapters.base import AuthRequiredError
    from bibilab.adapters.bilibili import BilibiliAdapter

    with patch.object(BilibiliAdapter, "resolve_flat", side_effect=AuthRequiredError("course")):
        list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
        resp = await client.post(
            "/ingest/preview",
            json={"list_id": list_id, "url": "https://www.bilibili.com/cheese/course123"},
        )
    assert resp.status_code == 401
    data = resp.json()["detail"]
    assert data["resource_type"] == "course"


@pytest.mark.asyncio
async def test_preview_private_playlist_returns_401(client: httpx.AsyncClient):
    """Private Bilibili favorites list without cookie returns 401 with resource_type=playlist."""
    from bibilab.adapters.base import AuthRequiredError
    from bibilab.adapters.bilibili import BilibiliAdapter

    with patch.object(BilibiliAdapter, "resolve_flat", side_effect=AuthRequiredError("playlist")):
        list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
        resp = await client.post(
            "/ingest/preview",
            json={"list_id": list_id, "url": "https://space.bilibili.com/123/favlist?fid=456"},
        )
    assert resp.status_code == 401
    data = resp.json()["detail"]
    assert data["resource_type"] == "playlist"


@pytest.mark.asyncio
async def test_preview_download_error(client: httpx.AsyncClient):
    from bibilab.adapters.base import DownloadError
    from bibilab.adapters.bilibili import BilibiliAdapter

    with patch.object(BilibiliAdapter, "resolve_flat", side_effect=DownloadError("Video unavailable")):
        list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
        resp = await client.post(
            "/ingest/preview",
            json={"list_id": list_id, "url": "https://www.bilibili.com/video/BVdeadbeef"},
        )
    assert resp.status_code == 400
    assert "Video unavailable" in resp.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_preview_no_sources_or_jobs_created(
    client: httpx.AsyncClient,
    mock_resolve_single,
    tmp_bibilab_home: Path,
):
    from bibilab.db import bootstrap_db, get_pending_jobs

    await bootstrap_db()
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://www.bilibili.com/video/BV1abc123"},
    )
    assert resp.status_code == 200
    jobs = await get_pending_jobs()
    assert len(jobs) == 0


@pytest.fixture()
def mock_resolve_flat_medialist():
    from bibilab.adapters.base import PlaylistMeta, VideoMeta
    from bibilab.adapters.bilibili import BilibiliAdapter

    videos = [
        VideoMeta(
            video_id="BVxyz789",
            title="Medialist Video 1",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BVxyz789",
            cover_url="",  # empty — flat resolve may not have thumbnails
            duration_seconds=0,  # zero — flat resolve may not have durations
            uploader="",  # empty — flat resolve may not have uploader
            part_label=None,
        ),
        VideoMeta(
            video_id="BVabc123",
            title="Medialist Video 2",
            platform="bilibili",
            source_url="https://www.bilibili.com/video/BVabc123",
            cover_url="",
            duration_seconds=0,
            uploader="",
            part_label=None,
        ),
    ]
    with patch.object(
        BilibiliAdapter,
        "resolve_flat",
        return_value=PlaylistMeta(
            playlist_id="ml123",
            title="My Medialist",
            platform="bilibili",
            source_url="https://bilibili.com/medialist/ml123",
            videos=videos,
        ),
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_preview_flat_returns_video_list(client: httpx.AsyncClient, mock_resolve_flat_medialist):
    """POST /ingest/preview returns video list from resolve_flat without downloading full metadata."""
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://bilibili.com/medialist/ml123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["videos"]) == 2
    video_ids = {v["video_id"] for v in data["videos"]}
    assert video_ids == {"BVxyz789", "BVabc123"}
    titles = {v["title"] for v in data["videos"]}
    assert titles == {"Medialist Video 1", "Medialist Video 2"}
    # Flat resolve does not provide full metadata
    for v in data["videos"]:
        assert v["cover_url"] == ""
        assert v["duration_seconds"] == 0
        assert v["uploader"] == ""
        assert v["status"] == "new"


@pytest.mark.asyncio
async def test_preview_flat_unknown_list(client: httpx.AsyncClient, mock_resolve_flat_medialist):
    """POST /ingest/preview returns 404 for unknown list."""
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": "nonexistent", "url": "https://bilibili.com/medialist/ml123"},
    )
    assert resp.status_code == 404

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


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

    from bibilab.db import bootstrap_db, create_list

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await SourceFactory.build(
        "list-1",
        source_id=str(uuid.uuid4()),
        video_id="BV1abc123",
        title="T",
        source_url="https://www.bilibili.com/video/BV1abc123",
        uploader="",
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
async def test_preview_unsupported_domain_400(client: httpx.AsyncClient):
    list_id = (await client.post("/lists", json={"name": "Test"})).json()["id"]
    resp = await client.post(
        "/ingest/preview",
        json={"list_id": list_id, "url": "https://example.com/watch"},
    )
    assert resp.status_code == 400
    assert "example.com" in resp.json()["detail"]["message"]


@pytest.mark.asyncio
async def test_preview_metadata_unsupported_platform_400(client: httpx.AsyncClient):
    resp = await client.post(
        "/ingest/preview/metadata",
        json={"video_ids": ["BV1abc123"], "platform": "nosuch"},
    )
    assert resp.status_code == 400
    assert "nosuch" in resp.json()["detail"]["message"]


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
async def test_write_source_stores_relative_paths(tmp_bibilab_home: Path):
    """After a source write, the source row is persisted correctly."""
    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")

    # Write the source (the DB write happens here)
    await SourceFactory.build(
        "list-1",
        source_id="test-uuid-1234",
        video_id="BV1relative",
        title="Test",
        cover_url="https://example.com/cover.jpg",
        source_url="https://bilibili.com/video/BV1relative",
        duration_seconds=3600,
        uploader="TestUser",
    )

    # Read back and verify source was written
    row = await get_source("test-uuid-1234")
    assert row["id"] == "test-uuid-1234"


@pytest.mark.asyncio
async def test_pipeline_creates_covers_and_segments(tmp_bibilab_home: Path, mock_call_llm):
    """Full pipeline with mocks: covers + transcript segments persisted."""
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

    mock_call_llm.return_value = mock_digest_json

    mock_embed = MagicMock()

    mock_dl_cover = MagicMock(side_effect=lambda url, dest: dest.write_bytes(b"fake cover data") or True)

    with (
        patch("bibilab.worker.extract_audio", mock_extract_audio),
        patch("bibilab.worker.transcribe", mock_transcribe_fn),
        patch("bibilab.worker._download_cover", mock_dl_cover),
        patch("bibilab.worker.embed_chunks", mock_embed),
    ):
        await worker._pipeline(job)

    # Verify cover was downloaded
    assert (tmp_bibilab_home / "covers" / f"{source_id}.jpg").exists(), (
        "cover should be saved as covers/{source_id}.jpg"
    )

    # Verify transcript segments were written
    from bibilab.db import get_transcript_segments

    segs = await get_transcript_segments(source_id)
    assert len(segs) > 0, "transcript segments should be persisted"

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

    from bibilab.db import bootstrap_db, create_list

    await bootstrap_db()
    await create_list("list-1", "Test", "2026-01-01T00:00:00")
    await SourceFactory.build(
        "list-1",
        source_id=str(uuid.uuid4()),
        video_id="BV1abc123",
        title="T",
        source_url="https://www.bilibili.com/video/BV1abc123",
        uploader="",
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


async def test_write_source_persists_facet_columns(tmp_bibilab_home: Path):
    """A source write persists series_name, sequence_number, season_number."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    list_id = "list-facet-ws"
    await create_list(list_id, "Facet WS", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVfacet001",
        title="罗翔说刑法 EP08",
        source_url="https://example.com/BVfacet001",
        duration_seconds=600,
        uploader="UploaderX",
        language="zh",
        series_name="罗翔说刑法",
        sequence_number=8,
    )

    source = await get_source(source_id)
    assert source is not None
    assert source["series_name"] == "罗翔说刑法"
    assert source["sequence_number"] == 8
    assert source["season_number"] is None


async def test_apply_digest_facets_persists_facets(tmp_bibilab_home: Path):
    """apply_digest_facets writes all 3 facet columns."""
    import uuid

    from bibilab.db import (
        apply_digest_facets,
        bootstrap_db,
        create_list,
        get_source,
    )

    await bootstrap_db()
    list_id = "list-facet-upd"
    await create_list(list_id, "Facet Update", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVfacet002",
        title="Test",
        source_url="https://example.com/BVfacet002",
        duration_seconds=300,
        uploader="U",
    )

    await apply_digest_facets(
        source_id,
        series_name="Test Series",
        sequence_number=42,
        season_number=2,
    )

    source = await get_source(source_id)
    assert source is not None
    assert source["series_name"] == "Test Series"
    assert source["sequence_number"] == 42
    assert source["season_number"] == 2


async def test_apply_digest_facets_rerun_preserves_processed_at(tmp_bibilab_home: Path):
    """bump_processed_at=False (rerun) leaves processed_at untouched so list ordering is stable."""
    import uuid

    from bibilab.db import (
        apply_digest_facets,
        bootstrap_db,
        create_list,
        get_source,
    )

    await bootstrap_db()
    list_id = "list-rerun-pa"
    await create_list(list_id, "Rerun Preserve", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())
    original_pa = "2025-06-01T12:00:00+00:00"

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVrerunPA01",
        title="Test",
        source_url="https://example.com/BVrerunPA01",
        duration_seconds=300,
        uploader="U",
    )
    # Pin processed_at to a known past timestamp.
    from bibilab.db import get_db

    async with get_db() as db:
        await db.execute(
            "UPDATE sources SET processed_at=? WHERE id=?",
            (original_pa, source_id),
        )
        await db.commit()

    await apply_digest_facets(
        source_id,
        bump_processed_at=False,
    )

    source = await get_source(source_id)
    assert source is not None
    assert source["processed_at"] == original_pa


async def test_apply_digest_facets_default_bumps_processed_at(tmp_bibilab_home: Path):
    """Default bump_processed_at=True preserves the old invariant: every digest write moves processed_at to now."""
    import uuid

    from bibilab.db import (
        apply_digest_facets,
        bootstrap_db,
        create_list,
        get_source,
    )

    await bootstrap_db()
    list_id = "list-bump-pa"
    await create_list(list_id, "Bump", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())
    original_pa = "2025-06-01T12:00:00+00:00"

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVbumpPA01",
        title="Test",
        source_url="https://example.com/BVbumpPA01",
        duration_seconds=300,
        uploader="U",
    )
    from bibilab.db import get_db

    async with get_db() as db:
        await db.execute(
            "UPDATE sources SET processed_at=? WHERE id=?",
            (original_pa, source_id),
        )
        await db.commit()

    await apply_digest_facets(source_id)

    source = await get_source(source_id)
    assert source is not None
    assert source["processed_at"] != original_pa


async def test_write_source_reingest_coalesces_facets(tmp_bibilab_home: Path):
    """Re-ingest with null facets preserves prior values; non-null overwrites."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    list_id = "list-facet-coalesce"
    await create_list(list_id, "Facet Coalesce", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())

    base = dict(
        source_id=source_id,
        video_id="BVcoalesce01",
        platform="bilibili",
        list_id=list_id,
        title="T",
        cover_url=None,
        source_url="https://example.com/BVcoalesce01",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        settings_snapshot={},
    )

    await SourceFactory.build(
        **base,
        series_name="罗翔说刑法",
        sequence_number=8,
        season_number=1,
    )

    # Re-ingest where the digest found no facets — prior values must survive.
    await SourceFactory.build(**base)
    source = await get_source(source_id)
    assert source is not None
    assert source["series_name"] == "罗翔说刑法"
    assert source["sequence_number"] == 8
    assert source["season_number"] == 1

    # Re-ingest with a fresh non-null value — that field is overwritten.
    await SourceFactory.build(
        **base,
        sequence_number=9,
    )
    source = await get_source(source_id)
    assert source is not None
    assert source["series_name"] == "罗翔说刑法"  # still preserved (null this run)
    assert source["sequence_number"] == 9  # overwritten
    assert source["season_number"] == 1  # still preserved


async def test_apply_digest_facets_coalesces_facets(tmp_bibilab_home: Path):
    """apply_digest_facets with null facets preserves prior values."""
    import uuid

    from bibilab.db import (
        apply_digest_facets,
        bootstrap_db,
        create_list,
        get_source,
    )

    await bootstrap_db()
    list_id = "list-facet-upd-coalesce"
    await create_list(list_id, "Facet Update Coalesce", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVcoalesce02",
        title="T",
        source_url="https://example.com/BVcoalesce02",
        duration_seconds=10,
        uploader="U",
        series_name="Keep Me",
        sequence_number=3,
        season_number=2,
    )

    # Rerun whose digest produced no facets — prior facets survive.
    await apply_digest_facets(source_id)
    source = await get_source(source_id)
    assert source is not None
    assert source["series_name"] == "Keep Me"
    assert source["sequence_number"] == 3
    assert source["season_number"] == 2


async def test_update_source_facets_replace_semantics(tmp_bibilab_home: Path):
    """update_source_facets REPLACES (explicit None clears), unlike the digest COALESCE path."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source, update_source_facets

    await bootstrap_db()
    list_id = "list-usf"
    await create_list(list_id, "USF", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())
    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVusf01",
        title="T",
        source_url="https://example.com/BVusf01",
        duration_seconds=10,
        uploader="U",
        series_name="罗翔说刑法",
        sequence_number=8,
        season_number=1,
    )
    before = await get_source(source_id)

    await update_source_facets(source_id, series_name="新系列", sequence_number=9, season_number=None)
    after = await get_source(source_id)
    assert after["series_name"] == "新系列"
    assert after["sequence_number"] == 9
    assert after["season_number"] is None
    assert after["processed_at"] == before["processed_at"]


async def test_update_source_facets_partial_and_noop(tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source, update_source_facets

    await bootstrap_db()
    list_id = "list-usf2"
    await create_list(list_id, "USF2", "2026-01-01T00:00:00")
    source_id = str(uuid.uuid4())
    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVusf02",
        title="T",
        source_url="https://example.com/BVusf02",
        duration_seconds=10,
        uploader="U",
        series_name="S",
        sequence_number=1,
        season_number=2,
    )
    await update_source_facets(source_id, sequence_number=5)
    await update_source_facets(source_id)
    after = await get_source(source_id)
    assert after["sequence_number"] == 5
    assert after["series_name"] == "S"
    assert after["season_number"] == 2


async def test_update_source_facets_missing_row_raises_lookuperror(tmp_bibilab_home: Path):
    """An UPDATE that matches no row raises LookupError (no silent no-op commit)."""
    import uuid

    import pytest

    from bibilab.db import bootstrap_db, update_source_facets

    await bootstrap_db()
    with pytest.raises(LookupError):
        await update_source_facets(str(uuid.uuid4()), series_name="X")


@pytest.mark.asyncio
async def test_long_source_chunks_nest_in_sections(tmp_bibilab_home: Path):
    """Acceptance: after ingest of a long source, every chunk's seg range
    is fully contained in exactly one section's seg range."""
    from bibilab.db import bootstrap_db, create_list, get_sections, write_transcript_segments
    from bibilab.pipeline.section import chunk_by_sections, derive_sections
    from bibilab.pipeline.transcribe import WhisperSegment
    from bibilab.worker import WorkerLoop

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1long")

    # 180 long segments → ≥2 sections
    segs = [
        WhisperSegment(start=float(i), end=float(i + 1), text=("word " * 100).strip() + ".", speaker="S")
        for i in range(180)
    ]

    # Pre-stage segments via transcript_segments table so the worker sees them
    await write_transcript_segments(source_id, segs)

    # Manually run the per-stage pipeline: derive + chunk + persist
    sections = derive_sections(segs)
    chunks = chunk_by_sections(segs, sections, language="en")

    # Write sections + segments atomically (extraction mocked)
    extraction = SimpleNamespace(
        summary="",
        keywords=[],
        series_name=None,
        sequence_number=None,
        season_number=None,
    )
    from bibilab.pipeline.digest import SectionDigest

    section_digests = [SectionDigest(summary="", keywords=[]) for _ in sections]
    # SimpleNamespace lacks model_dump(); add a lambda so cfg.model_dump() works.
    cfg = SimpleNamespace(
        transcription=SimpleNamespace(model="large-v3", language="en"),
        ai=SimpleNamespace(model="gpt-4o"),
        model_dump=lambda: {},
    )
    video_meta = SimpleNamespace(
        video_id="BV1long",
        platform="bilibili",
        title="T",
        cover_url=None,
        source_url="https://x",
        duration_seconds=0,
        uploader="u",
    )
    worker = WorkerLoop(home=tmp_bibilab_home, config=cfg, adapter=None)
    await worker._stage_persist(
        job_id="job-x",
        source_id=source_id,
        video_id="BV1long",
        video_meta=video_meta,
        list_id="list-1",
        extraction=extraction,
        sections=sections,
        section_digests=section_digests,
        detected_language="en",
        cfg=cfg,
        sentence_segments=segs,
    )

    rows = await get_sections(source_id)
    assert len(rows) >= 2

    for c in chunks:
        containing = [r for r in rows if r["seg_start"] <= c.seg_start and c.seg_end <= r["seg_end"]]
        assert len(containing) == 1, (
            f"chunk [{c.seg_start}..{c.seg_end}] contained in {len(containing)} sections, expected 1"
        )


@pytest.mark.asyncio
async def test_short_source_one_section_byte_identical_chunks(tmp_bibilab_home: Path):
    """Regression guard: a short video (1 section) produces the same chunk
    set as the pre-change single `chunk_segments` call."""
    from bibilab.db import bootstrap_db, create_list
    from bibilab.pipeline.chunk import chunk_segments
    from bibilab.pipeline.section import chunk_by_sections, derive_sections
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    await SourceFactory.build("list-1", video_id="BV1short")

    segs = [
        WhisperSegment(start=float(i), end=float(i + 1), text=f"short sentence number {i}.", speaker="S")
        for i in range(40)
    ]

    pre = chunk_segments(segs, language="en")
    sections = derive_sections(segs)
    assert len(sections) == 1  # pre-condition for byte-identical
    post = chunk_by_sections(segs, sections, language="en")

    # Compare the observable chunk fields
    assert len(pre) == len(post)
    for p, q in zip(pre, post):
        assert p.sequence_index == q.sequence_index
        assert p.seg_start == q.seg_start
        assert p.seg_end == q.seg_end
        assert p.text == q.text
        assert p.timestamp_start == q.timestamp_start
        assert p.timestamp_end == q.timestamp_end


@pytest.mark.asyncio
async def test_re_ingest_replaces_section_rows(tmp_bibilab_home: Path):
    """Re-ingest of a source replaces (does not duplicate) its section rows."""
    from bibilab.db import _exec_write_sections, bootstrap_db, create_list, get_db, get_sections
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section

    await bootstrap_db()
    await create_list("list-1", "L", "2026-01-01T00:00:00")
    source_id = await SourceFactory.build("list-1", video_id="BV1re")

    # First ingest: 2 sections
    async with get_db() as db:
        await _exec_write_sections(
            db,
            source_id,
            [
                Section(seg_start=0, seg_end=5, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
                Section(seg_start=6, seg_end=12, token_count=200, timestamp_start=11.0, timestamp_end=22.0),
            ],
            [SectionDigest(summary="s0", keywords=[]), SectionDigest(summary="s1", keywords=[])],
        )
        await db.commit()
    assert len(await get_sections(source_id)) == 2

    # Re-ingest: 1 section (different shape — re-ingest replaces, not merges)
    async with get_db() as db:
        await _exec_write_sections(
            db,
            source_id,
            [
                Section(seg_start=0, seg_end=20, token_count=500, timestamp_start=0.0, timestamp_end=40.0),
            ],
            [SectionDigest(summary="s0", keywords=[])],
        )
        await db.commit()
    rows = await get_sections(source_id)
    assert len(rows) == 1
    assert rows[0]["seg_end"] == 20

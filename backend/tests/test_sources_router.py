import json
from pathlib import Path

import httpx
import pytest

from tests.factories import SourceFactory

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_get_source_returns_digest_and_transcript(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    list_id = "list-for-source-test"
    await create_list(list_id, "Test List", "2025-01-01T00:00:00Z")

    source_id = "src-abc123"
    video_id = "BVtest456"

    # Write the cover file
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"fake-cover-image")

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id=video_id,
        cover_url="https://example.com/cover.jpg",
        source_url="https://www.bilibili.com/video/BVtest456",
        duration_seconds=120,
        language="en",
        whisper_model="base",
        settings_snapshot={"language": "en"},
    )

    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=2.0, text="hello transcript", speaker=None)]
    )

    resp = await client.get(f"/sources/{source_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == source_id
    assert data["transcript"] == "[SPK? @0:00] hello transcript"
    assert data["cover_url"] == "https://example.com/cover.jpg"
    assert data["video_id"] == video_id


@pytest.mark.asyncio
async def test_get_source_cover(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list

    list_id = "list-for-cover-test"
    await create_list(list_id, "Cover Test List", "2025-01-01T00:00:00Z")

    source_id = "src-cover-789"
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"cover-image-bytes")

    await SourceFactory.build(
        list_id,
        source_id=source_id,
        video_id="BVcoverVid",
        title="Cover Test Video",
        source_url="https://www.bilibili.com/video/BVcoverVid",
        duration_seconds=60,
        uploader="Uploader",
        whisper_model="base",
    )

    resp = await client.get(f"/sources/{source_id}/cover")
    assert resp.status_code == 200
    assert resp.content == b"cover-image-bytes"


@pytest.mark.asyncio
async def test_get_source_not_found(client: httpx.AsyncClient):
    resp = await client.get("/sources/nonexistent-source-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_source_cover_not_found(client: httpx.AsyncClient):
    resp = await client.get("/sources/nonexistent-source-id/cover")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rerun_source_not_found(client: httpx.AsyncClient):
    """POST /sources/{source_id}/rerun returns 404 when source does not exist."""
    resp = await client.post("/sources/nonexistent-source-id/rerun")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rerun_source_success(client: httpx.AsyncClient, tmp_bibilab_home: Path, mock_call_llm):
    """POST /sources/{source_id}/rerun re-runs digest and updates summary/keywords."""
    from bibilab.db import create_list, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await create_list("list-rerun-test", "Rerun Test", "2025-01-01T00:00:00Z")

    source_id = "src-rerun-001"

    await SourceFactory.build(
        "list-rerun-test",
        source_id=source_id,
        video_id="BVrerun001",
        title="Rerun Test Video",
        source_url="https://www.bilibili.com/video/BVrerun001",
        duration_seconds=120,
        uploader="TestUser",
        language="en",
        whisper_model="base",
    )

    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=5.0, text="original transcript text", speaker=None)]
    )

    # Mock the LLM call to return new digest with facets
    mock_call_llm.return_value = (
        '{"summary": "new summary from rerun", "keywords": ["new", "rerun", "test"], '
        '"series_name": "Rerun Series", "sequence_number": 5, "season_number": 1}'
    )

    resp = await client.post(f"/sources/{source_id}/rerun")
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data

    # Verify job was created in DB
    from bibilab.db import get_job

    job = await get_job(data["job_id"])
    assert job is not None
    assert dict(job)["type"] == "digest"
    meta = json.loads(dict(job)["meta"])
    assert meta["source_id"] == source_id
    assert meta["list_id"] == "list-rerun-test"
    assert meta["source_title"] == "Rerun Test Video"


@pytest.mark.asyncio
async def test_rerun_source_dedup_conflict(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    """POST /sources/{source_id}/rerun returns 409 when a digest job for that source is already pending."""
    from bibilab.db import create_job, create_list, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await create_list("list-dedup", "Dedup Test", "2025-01-01T00:00:00Z")

    source_id = "src-dedup-001"

    await SourceFactory.build(
        "list-dedup",
        source_id=source_id,
        video_id="BVdedup001",
        title="Dedup Test Video",
        source_url="https://www.bilibili.com/video/BVdedup001",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
    )

    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=2.0, text="test transcript", speaker=None)]
    )

    # Create a pending digest job for this source
    await create_job(type="digest", meta={"source_id": source_id, "list_id": "list-dedup"})

    # Second rerun should get 409 Conflict
    resp = await client.post(f"/sources/{source_id}/rerun")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Digest already in progress"


@pytest.mark.asyncio
async def test_rerun_source_respects_ui_lang_header(client: httpx.AsyncClient, tmp_bibilab_home: Path, monkeypatch):
    """POST /sources/{source_id}/rerun with X-UI-Lang:zh stores ui_lang in job meta."""
    from bibilab.db import create_list, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await create_list("list-lang-test", "Lang Test", "2025-01-01T00:00:00Z")

    source_id = "src-lang-001"

    await SourceFactory.build(
        "list-lang-test",
        source_id=source_id,
        video_id="BVlang001",
        title="Lang Test Video",
        source_url="https://www.bilibili.com/video/BVlang001",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
    )

    await write_transcript_segments(
        source_id, [WhisperSegment(start=0.0, end=2.0, text="test transcript", speaker=None)]
    )

    resp = await client.post(f"/sources/{source_id}/rerun", headers={"X-UI-Lang": "zh"})
    assert resp.status_code == 202
    data = resp.json()

    from bibilab.db import get_job

    job = await get_job(data["job_id"])
    meta = json.loads(dict(job)["meta"])
    assert meta.get("ui_lang") == "zh"


async def test_patch_facets_replace(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    await create_list("L", "L", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "L",
        source_id=sid,
        video_id="BVpf01",
        title="T",
        source_url="https://example.com/BVpf01",
        duration_seconds=10,
        uploader="U",
        series_name="老系列",
        sequence_number=3,
        season_number=5,
    )
    r = await client.patch(
        f"/sources/{sid}/facets",
        json={"series_name": "新系列", "sequence_number": 7, "season_number": None},
    )
    assert r.status_code == 204
    src = await get_source(sid)
    assert src["series_name"] == "新系列"
    assert src["sequence_number"] == 7
    assert src["season_number"] is None


async def test_patch_facets_kindless_number_persists(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    await create_list("L2", "L2", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "L2",
        source_id=sid,
        video_id="BVpf02",
        title="T",
        source_url="https://example.com/BVpf02",
        duration_seconds=10,
        uploader="U",
    )
    r = await client.patch(f"/sources/{sid}/facets", json={"sequence_number": 8})
    assert r.status_code == 204
    src = await get_source(sid)
    assert src["sequence_number"] == 8


async def test_patch_facets_invalid_int_422(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list

    await bootstrap_db()
    await create_list("L3", "L3", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "L3",
        source_id=sid,
        video_id="BVpf03",
        title="T",
        source_url="https://example.com/BVpf03",
        duration_seconds=10,
        uploader="U",
    )
    for bad in [{"sequence_number": 0}, {"sequence_number": "abc"}, {"season_number": -1}]:
        r = await client.patch(f"/sources/{sid}/facets", json=bad)
        assert r.status_code == 422


async def test_patch_facets_404(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import bootstrap_db

    await bootstrap_db()
    r = await client.patch("/sources/does-not-exist/facets", json={"series_name": "X"})
    assert r.status_code == 404


async def test_patch_facets_empty_body_noop(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    """Empty {} patch on a live source is an idempotent 204 no-op; row unchanged."""
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source

    await bootstrap_db()
    await create_list("Le", "Le", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "Le",
        source_id=sid,
        video_id="BVpf04",
        title="T",
        source_url="https://example.com/BVpf04",
        duration_seconds=10,
        uploader="U",
        series_name="keep",
        sequence_number=3,
        season_number=1,
    )
    r = await client.patch(f"/sources/{sid}/facets", json={})
    assert r.status_code == 204
    src = await get_source(sid)
    assert src["series_name"] == "keep"
    assert src["sequence_number"] == 3
    assert src["season_number"] == 1


async def test_patch_facets_toctou_lookuperror_maps_to_404(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    """If the row vanishes between the existence check and the write,
    update_source_facets raises LookupError and the router returns 404
    (not a silent 204)."""
    import uuid

    from bibilab.db import bootstrap_db, create_list

    await bootstrap_db()
    await create_list("Lt", "Lt", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "Lt",
        source_id=sid,
        video_id="BVpf05",
        title="T",
        source_url="https://example.com/BVpf05",
        duration_seconds=10,
        uploader="U",
    )

    import bibilab.routers.sources as sources_router

    real_update = sources_router.update_source_facets

    async def raise_lookup(source_id, **fields):
        raise LookupError(source_id)

    sources_router.update_source_facets = raise_lookup
    try:
        r = await client.patch(f"/sources/{sid}/facets", json={"series_name": "X"})
    finally:
        sources_router.update_source_facets = real_update
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_source_content_reads_transcript_from_segments_table(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import _now, bootstrap_db, create_list, get_db, write_transcript_segments
    from bibilab.pipeline.transcribe import WhisperSegment

    await bootstrap_db()
    await create_list("list-1", "L", _now())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sources (id, video_id, platform, list_id, title) VALUES (?, ?, ?, ?, ?)",
            ("src-1", "BV1", "bilibili", "list-1", "T"),
        )
        await db.commit()
    await write_transcript_segments("src-1", [WhisperSegment(start=0.0, end=2.0, text="你好。", speaker="SPK_0")])

    resp = await client.get("/sources/src-1")
    assert resp.status_code == 200
    assert resp.json()["transcript"] == "[SPK_0 @0:00] 你好。"


@pytest.mark.asyncio
async def test_get_source_sections_returns_projected_list(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list, write_source_with_segments
    from bibilab.pipeline.digest import SectionDigest
    from bibilab.pipeline.section import Section
    from bibilab.pipeline.transcribe import WhisperSegment

    await create_list("list-sections", "Sections Test", "2025-01-01T00:00:00Z")

    source_id = "src-sections-001"
    segments = [WhisperSegment(start=float(i), end=float(i + 1), text=f"s{i}.", speaker=None) for i in range(20)]
    sections = [
        Section(seg_start=0, seg_end=9, token_count=100, timestamp_start=0.0, timestamp_end=10.0),
        Section(seg_start=10, seg_end=19, token_count=100, timestamp_start=10.0, timestamp_end=20.0),
    ]
    await write_source_with_segments(
        segments=segments,
        sections=sections,
        section_digests=[
            SectionDigest(summary="Sum 0", keywords=["k0"]),
            SectionDigest(summary="Sum 1", keywords=["k1"]),
        ],
        source_id=source_id,
        video_id="BVsections",
        platform="bilibili",
        list_id="list-sections",
        title="Sections Test Video",
        cover_url=None,
        source_url="https://www.bilibili.com/video/BVsections",
        duration_seconds=20,
        uploader="TestUploader",
        language="en",
        whisper_model="x",
        ai_model="y",
        settings_snapshot={},
    )

    resp = await client.get(f"/sources/{source_id}/sections")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0] == {
        "section_id": "1",
        "seq": 0,
        "summary": "Sum 0",
        "keywords": ["k0"],
        "timestamp_start": 0.0,
        "timestamp_end": 10.0,
    }
    assert data[1]["seq"] == 1
    # Response must NOT include internal columns.
    for row in data:
        assert "id" not in row
        assert "source_id" not in row
        assert "seg_start" not in row
        assert "seg_end" not in row
        assert "token_count" not in row


@pytest.mark.asyncio
async def test_get_source_sections_404_for_missing_source(client: httpx.AsyncClient):
    resp = await client.get("/sources/does-not-exist/sections")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_source_sections_empty_list_is_200(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    """Source exists but has no section rows — API must be honest about it."""
    from bibilab.db import bootstrap_db, create_list

    await bootstrap_db()
    await create_list("list-no-sections", "No Sections Test", "2025-01-01T00:00:00Z")

    # Use SourceFactory to create a source without sections (no segments/sections written)
    import uuid

    sid = str(uuid.uuid4())
    await SourceFactory.build(
        "list-no-sections",
        source_id=sid,
        video_id="BVnoSections",
        title="No Sections Video",
        source_url="https://www.bilibili.com/video/BVnoSections",
        duration_seconds=10,
        uploader="U",
    )

    resp = await client.get(f"/sources/{sid}/sections")
    assert resp.status_code == 200
    assert resp.json() == []

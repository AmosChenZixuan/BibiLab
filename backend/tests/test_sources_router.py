import json
from pathlib import Path

import httpx
import pytest

from bibilab.db import write_source


@pytest.mark.asyncio
async def test_get_source_returns_digest_and_transcript(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    from bibilab.db import create_list

    list_id = "list-for-source-test"
    await create_list(list_id, "Test List", "2025-01-01T00:00:00Z")

    source_id = "src-abc123"
    video_id = "BVtest456"
    transcript_path = f"transcripts/{source_id}.txt"

    # Write the transcript file
    transcript_file = tmp_bibilab_home / transcript_path
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("hello transcript", encoding="utf-8")

    # Write the cover file
    cover_file = tmp_bibilab_home / "covers" / f"{source_id}.jpg"
    cover_file.parent.mkdir(parents=True, exist_ok=True)
    cover_file.write_bytes(b"fake-cover-image")

    await write_source(
        source_id=source_id,
        video_id=video_id,
        platform="bilibili",
        list_id=list_id,
        title="Test Video",
        summary="A test summary.",
        keywords=["ai", "video"],
        cover_url="https://example.com/cover.jpg",
        transcript_path=transcript_path,
        source_url="https://www.bilibili.com/video/BVtest456",
        duration_seconds=120,
        uploader="TestUploader",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={"language": "en"},
    )

    resp = await client.get(f"/sources/{source_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == source_id
    assert data["summary"] == "A test summary."
    assert data["keywords"] == ["ai", "video"]
    assert data["transcript"] == "hello transcript"
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

    await write_source(
        source_id=source_id,
        video_id="BVcoverVid",
        platform="bilibili",
        list_id=list_id,
        title="Cover Test Video",
        summary="S",
        keywords=[],
        cover_url=None,
        transcript_path=None,
        source_url="https://www.bilibili.com/video/BVcoverVid",
        duration_seconds=60,
        uploader="Uploader",
        language=None,
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
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
async def test_rerun_source_success(client: httpx.AsyncClient, tmp_bibilab_home: Path, monkeypatch):
    """POST /sources/{source_id}/rerun re-runs digest and updates summary/keywords."""
    from bibilab.db import create_list, get_source, write_source

    await create_list("list-rerun-test", "Rerun Test", "2025-01-01T00:00:00Z")

    source_id = "src-rerun-001"
    transcript_path = f"transcripts/{source_id}.txt"
    transcript_file = tmp_bibilab_home / transcript_path
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("original transcript text", encoding="utf-8")

    await write_source(
        source_id=source_id,
        video_id="BVrerun001",
        platform="bilibili",
        list_id="list-rerun-test",
        title="Rerun Test Video",
        summary="old summary",
        keywords=["old", "keyword"],
        cover_url=None,
        transcript_path=transcript_path,
        source_url="https://www.bilibili.com/video/BVrerun001",
        duration_seconds=120,
        uploader="TestUser",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    # Mock the LLM call to return new digest with facets
    new_digest = (
        '{"summary": "new summary from rerun", "keywords": ["new", "rerun", "test"], '
        '"series_name": "Rerun Series", "sequence_number": 5, "season_number": 1}'
    )

    def mock_call_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        return new_digest

    import bibilab.pipeline.digest as digest_module

    monkeypatch.setattr(digest_module, "_call_llm", mock_call_llm)

    resp = await client.post(f"/sources/{source_id}/rerun")
    assert resp.status_code == 200

    # Verify summary, keywords, and facets were updated
    source = await get_source(source_id)
    assert source is not None
    assert source["summary"] == "new summary from rerun"
    assert json.loads(source["keywords"]) == ["new", "rerun", "test"]
    assert source["series_name"] == "Rerun Series"
    assert source["sequence_number"] == 5
    assert source["season_number"] == 1

    # Verify transcript file was not modified
    assert transcript_file.read_text(encoding="utf-8") == "original transcript text"


@pytest.mark.asyncio
async def test_rerun_source_respects_ui_lang_header(client: httpx.AsyncClient, tmp_bibilab_home: Path, monkeypatch):
    """POST /sources/{source_id}/rerun with X-UI-Lang:zh passes Chinese lang instruction to LLM."""
    from bibilab.db import create_list, write_source

    await create_list("list-lang-test", "Lang Test", "2025-01-01T00:00:00Z")

    source_id = "src-lang-001"
    transcript_path = f"transcripts/{source_id}.txt"
    transcript_file = tmp_bibilab_home / transcript_path
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.write_text("test transcript", encoding="utf-8")

    await write_source(
        source_id=source_id,
        video_id="BVlang001",
        platform="bilibili",
        list_id="list-lang-test",
        title="Lang Test Video",
        summary="old",
        keywords=[],
        cover_url=None,
        transcript_path=transcript_path,
        source_url="https://www.bilibili.com/video/BVlang001",
        duration_seconds=60,
        uploader="TestUser",
        language="en",
        whisper_model="base",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )

    captured_prompt = None

    def mock_call_llm(prompt, cfg, llm_timeout=120, llm_max_tokens=2048):
        nonlocal captured_prompt
        captured_prompt = prompt
        return '{"summary": "new summary", "keywords": []}'

    import bibilab.pipeline.digest as digest_module

    monkeypatch.setattr(digest_module, "_call_llm", mock_call_llm)

    resp = await client.post(f"/sources/{source_id}/rerun", headers={"X-UI-Lang": "zh"})
    assert resp.status_code == 200
    assert captured_prompt is not None
    assert "请用中文回答" in captured_prompt


async def test_patch_facets_replace(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list, get_source, write_source

    await bootstrap_db()
    await create_list("L", "L", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await write_source(
        source_id=sid,
        video_id="BVpf01",
        platform="bilibili",
        list_id="L",
        title="T",
        summary="s",
        keywords=["k"],
        cover_url=None,
        transcript_path=None,
        source_url="https://example.com/BVpf01",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
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

    from bibilab.db import bootstrap_db, create_list, get_source, write_source

    await bootstrap_db()
    await create_list("L2", "L2", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await write_source(
        source_id=sid,
        video_id="BVpf02",
        platform="bilibili",
        list_id="L2",
        title="T",
        summary="s",
        keywords=["k"],
        cover_url=None,
        transcript_path=None,
        source_url="https://example.com/BVpf02",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
    )
    r = await client.patch(f"/sources/{sid}/facets", json={"sequence_number": 8})
    assert r.status_code == 204
    src = await get_source(sid)
    assert src["sequence_number"] == 8


async def test_patch_facets_invalid_int_422(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    import uuid

    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("L3", "L3", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await write_source(
        source_id=sid,
        video_id="BVpf03",
        platform="bilibili",
        list_id="L3",
        title="T",
        summary="s",
        keywords=["k"],
        cover_url=None,
        transcript_path=None,
        source_url="https://example.com/BVpf03",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
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

    from bibilab.db import bootstrap_db, create_list, get_source, write_source

    await bootstrap_db()
    await create_list("Le", "Le", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await write_source(
        source_id=sid,
        video_id="BVpf04",
        platform="bilibili",
        list_id="Le",
        title="T",
        summary="s",
        keywords=["k"],
        cover_url=None,
        transcript_path=None,
        source_url="https://example.com/BVpf04",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
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

    from bibilab.db import bootstrap_db, create_list, write_source

    await bootstrap_db()
    await create_list("Lt", "Lt", "2026-01-01T00:00:00")
    sid = str(uuid.uuid4())
    await write_source(
        source_id=sid,
        video_id="BVpf05",
        platform="bilibili",
        list_id="Lt",
        title="T",
        summary="s",
        keywords=["k"],
        cover_url=None,
        transcript_path=None,
        source_url="https://example.com/BVpf05",
        duration_seconds=10,
        uploader="U",
        language=None,
        whisper_model="large-v3",
        ai_model="gpt-4o",
        vision_enabled=False,
        settings_snapshot={},
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

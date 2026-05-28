"""Tests for /models/asr router (replaces deleted test_whisper_models.py)."""

from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_list_asr_models_returns_all_registry_entries(
    client: httpx.AsyncClient,
    tmp_bibilab_home: Path,  # noqa: ARG001
):
    resp = await client.get("/models/asr")
    assert resp.status_code == 200
    data = resp.json()
    engines = {(m["engine"], m["name"]) for m in data}
    assert ("whisper", "medium") in engines
    assert ("whisper", "large-v3") in engines
    assert ("sensevoice", "small") in engines
    assert ("diarization", "cam++") in engines
    # Nothing downloaded in a fresh tmp home
    assert all(m["installed"] is False for m in data)


@pytest.mark.asyncio
async def test_list_asr_models_marks_selected(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    await client.put(
        "/config",
        json={"transcription": {"engine": "whisper", "model_size": "medium"}},
    )
    resp = await client.get("/models/asr")
    selected = [(m["engine"], m["name"]) for m in resp.json() if m["selected"]]
    assert selected == [("whisper", "medium")]


@pytest.mark.asyncio
async def test_download_asr_model_queues_job(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    resp = await client.post(
        "/models/asr/download",
        json={"engine": "whisper", "model_size": "medium"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["engine"] == "whisper"
    assert body["model_size"] == "medium"
    assert body["job_id"]


@pytest.mark.asyncio
async def test_download_asr_model_rejects_unknown_engine(client: httpx.AsyncClient):
    resp = await client.post(
        "/models/asr/download",
        json={"engine": "garbage", "model_size": "medium"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_download_asr_model_rejects_unknown_model(client: httpx.AsyncClient):
    resp = await client.post(
        "/models/asr/download",
        json={"engine": "whisper", "model_size": "tiny"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_download_asr_model_accepts_diarization(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    resp = await client.post(
        "/models/asr/download",
        json={"engine": "diarization", "model_size": "cam++"},
    )
    assert resp.status_code == 202
    assert resp.json()["engine"] == "diarization"


@pytest.mark.asyncio
async def test_list_asr_models_reports_installed_when_downloaded(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    target = tmp_bibilab_home / "models" / "whisper" / "medium"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.bin").write_bytes(b"")

    resp = await client.get("/models/asr")

    entry = next(m for m in resp.json() if m["engine"] == "whisper" and m["name"] == "medium")
    assert entry["installed"] is True
    assert entry["path"] == str(target)

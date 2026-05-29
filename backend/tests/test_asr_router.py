"""Tests for /models/asr router."""

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
    names = {m["name"] for m in data}
    assert {"large-v3", "sensevoice-small", "cam++"}.issubset(names)
    assert all(m["installed"] is False for m in data)


@pytest.mark.asyncio
async def test_list_asr_models_reports_size_mb(
    client: httpx.AsyncClient,
    tmp_bibilab_home: Path,  # noqa: ARG001
):
    resp = await client.get("/models/asr")
    sizes = {m["name"]: m["size_mb"] for m in resp.json()}
    assert sizes["large-v3"] == 3000
    assert sizes["sensevoice-small"] == 936
    assert sizes["cam++"] == 28


@pytest.mark.asyncio
async def test_list_asr_models_reports_kind(
    client: httpx.AsyncClient,
    tmp_bibilab_home: Path,  # noqa: ARG001
):
    resp = await client.get("/models/asr")
    kinds = {m["name"]: m["kind"] for m in resp.json()}
    assert kinds["large-v3"] == "transcription"
    assert kinds["sensevoice-small"] == "transcription"
    assert kinds["cam++"] == "diarization"


@pytest.mark.asyncio
async def test_list_asr_models_marks_selected(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    await client.put("/config", json={"transcription": {"model": "large-v3"}})
    resp = await client.get("/models/asr")
    selected = [m["name"] for m in resp.json() if m["selected"]]
    assert selected == ["large-v3"]


@pytest.mark.asyncio
async def test_download_asr_model_queues_job(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    resp = await client.post("/models/asr/download", json={"model_name": "sensevoice-small"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["model_name"] == "sensevoice-small"
    assert body["job_id"]


@pytest.mark.asyncio
async def test_download_asr_model_rejects_unknown_model(client: httpx.AsyncClient):
    resp = await client.post("/models/asr/download", json={"model_name": "tiny"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_download_asr_model_accepts_diarization(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    resp = await client.post("/models/asr/download", json={"model_name": "cam++"})
    assert resp.status_code == 202
    assert resp.json()["model_name"] == "cam++"


@pytest.mark.asyncio
async def test_list_asr_models_reports_whisper_installed(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    # WhisperWarp uses openai-whisper's cache (~/.cache/whisper/), not our models_dir.
    target = Path.home() / ".cache" / "whisper"
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "large-v3.pt"
    checkpoint.write_bytes(b"")
    try:
        resp = await client.get("/models/asr")
        entry = next(m for m in resp.json() if m["name"] == "large-v3")
        assert entry["installed"] is True
        assert entry["path"] == str(target)
    finally:
        checkpoint.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_list_asr_models_reports_sensevoice_installed(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    target = tmp_bibilab_home / "models" / "asr" / "sensevoice-small"
    target.mkdir(parents=True)
    (target / "configuration.json").write_text("{}")

    resp = await client.get("/models/asr")

    entry = next(m for m in resp.json() if m["name"] == "sensevoice-small")
    assert entry["installed"] is True
    assert entry["path"] == str(target)

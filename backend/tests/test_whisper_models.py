from pathlib import Path
from unittest.mock import patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_list_whisper_models_marks_local_install(client: httpx.AsyncClient, tmp_path: Path):
    model_dir = tmp_path / "models" / "whisper" / "whisper" / "whisper-medium"
    whisper_root = tmp_path / "models" / "whisper"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"bin")

    with patch("locus.whisper_models.whisper_model_dir", return_value=whisper_root):
        response = await client.get("/models/whisper")

    assert response.status_code == 200
    medium = next(item for item in response.json() if item["name"] == "medium")
    assert medium["installed"] is True
    assert medium["path"].endswith("whisper/whisper-medium")


@pytest.mark.asyncio
async def test_download_whisper_model_calls_downloader(client: httpx.AsyncClient, tmp_path: Path):
    response = await client.post("/models/whisper/download", json={"model_size": "small"})

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["model_family"] == "whisper"
    assert response.json()["model_size"] == "small"
    assert "job_id" in response.json()


@pytest.mark.asyncio
async def test_download_whisper_model_rejects_unknown_size(client: httpx.AsyncClient):
    response = await client.post("/models/whisper/download", json={"model_size": "giant"})
    assert response.status_code == 400


def test_download_whisper_model_uses_model_subdirectory(tmp_path: Path):
    whisper_root = tmp_path / "models" / "whisper"
    cache_root = tmp_path / ".cache" / "huggingface"

    def fake_download_model(model_size: str, output_dir: str, cache_dir: str):
        assert model_size == "medium"
        assert Path(output_dir) == whisper_root / "medium"
        assert Path(cache_dir) == cache_root
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "config.json").write_text("{}", encoding="utf-8")
        (Path(output_dir) / "model.bin").write_bytes(b"bin")
        (Path(output_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)

    with (
        patch("locus.whisper_models.locus_home", return_value=tmp_path),
        patch("faster_whisper.utils.download_model", side_effect=fake_download_model),
    ):
        from locus.whisper_models import download_whisper_model

        path = download_whisper_model("medium")

    assert path == whisper_root / "medium"
    assert not (whisper_root / "medium" / ".cache").exists()

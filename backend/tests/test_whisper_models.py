from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            with patch("locus.main.locus_home", return_value=tmp_path):
                yield tmp_path


@pytest.fixture()
def client(tmp_locus_home: Path):
    from locus.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_list_whisper_models_marks_local_install(client: TestClient, tmp_path: Path):
    model_dir = tmp_path / "models" / "whisper" / "whisper" / "whisper-medium"
    whisper_root = tmp_path / "models" / "whisper"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"bin")

    with patch("locus.whisper_models.whisper_model_dir", return_value=whisper_root):
        response = client.get("/models/whisper")

    assert response.status_code == 200
    medium = next(item for item in response.json() if item["name"] == "medium")
    assert medium["installed"] is True
    assert medium["path"].endswith("whisper/whisper-medium")


def test_download_whisper_model_calls_downloader(client: TestClient, tmp_path: Path):
    downloaded = tmp_path / "models" / "whisper" / "whisper" / "whisper-small"
    whisper_root = tmp_path / "models" / "whisper"
    downloaded.mkdir(parents=True)
    (downloaded / "config.json").write_text("{}", encoding="utf-8")
    (downloaded / "model.bin").write_bytes(b"bin")

    with (
        patch("locus.whisper_models.whisper_model_dir", return_value=whisper_root),
        patch("locus.routers.whisper.asyncio.to_thread") as mock_to_thread,
    ):

        async def run_inline(func, *args):
            return func(*args)

        mock_to_thread.side_effect = run_inline
        response = client.post("/models/whisper/download", json={"model_size": "small"})

    assert response.status_code == 201
    assert response.json()["name"] == "small"
    assert response.json()["path"].endswith("whisper/whisper-small")


def test_download_whisper_model_rejects_unknown_size(client: TestClient):
    response = client.post("/models/whisper/download", json={"model_size": "giant"})
    assert response.status_code == 400

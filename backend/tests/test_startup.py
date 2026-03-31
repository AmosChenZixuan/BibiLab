from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    """Redirect ~/.locus/ to a temp directory for tests."""
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


def test_health_returns_200(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall"] in ("ok", "error", "degraded")
    deps = data["dependencies"]
    assert "backend" in deps
    assert deps["backend"]["status"] == "ok"
    assert all(k in deps for k in ("llm", "whisper_model", "ffmpeg", "cuda", "bilibili_session"))


def test_config_defaults(client: TestClient):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"]["port"] == 8765
    assert data["ai"]["provider"] == "openai"
    assert data["transcription"]["model_size"] == "large-v3"


def test_config_deep_merge(client: TestClient):
    resp = client.put("/config", json={"ai": {"model": "gpt-4-turbo"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ai"]["model"] == "gpt-4-turbo"
    assert data["ai"]["provider"] == "openai"  # sibling preserved
    assert data["backend"]["port"] == 8765  # unrelated section preserved


def test_config_masks_sensitive_fields(client: TestClient):
    client.put("/config", json={
        "ai": {"api_key": "sk-secret"},
        "accounts": {"bilibili": {"cookie": "my-cookie"}},
    })
    data = client.get("/config").json()
    assert data["ai"]["api_key"] == "***"
    assert data["accounts"]["bilibili"]["cookie"] == "***"


def test_locus_dirs_bootstrapped(client: TestClient, tmp_locus_home: Path):
    for subdir in ("transcripts", "downloads", "chroma"):
        assert (tmp_locus_home / subdir).is_dir(), f"Missing {subdir}/"
    assert (tmp_locus_home / "locus.db").exists()

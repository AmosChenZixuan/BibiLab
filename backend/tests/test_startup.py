from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from locus.config import AIConfig
from locus.routers.health import _check_llm


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
    assert all(k in deps for k in ("llm", "whisper_model", "ffmpeg", "cuda", "embedding_model"))
    assert "bilibili_session" not in deps


def test_health_includes_embedding_model(client: TestClient, tmp_locus_home: Path):  # noqa: ARG001
    resp = client.get("/health")
    assert resp.status_code == 200
    deps = resp.json()["dependencies"]
    assert "embedding_model" in deps
    assert deps["embedding_model"]["status"] in ("ok", "error")


def test_health_reports_ffmpeg_install_path(client: TestClient):
    with patch("locus.routers.health.shutil.which", return_value="/usr/bin/ffmpeg"):
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["dependencies"]["ffmpeg"] == {
        "status": "ok",
        "message": "/usr/bin/ffmpeg",
    }


def test_health_reports_embedding_model_install_path(tmp_locus_home: Path, client: TestClient):  # noqa: ARG001
    model_file = tmp_locus_home / "models" / "embedding" / "onnx" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fake")

    with (
        patch(
            "locus.routers.health._embedding_model_dir",
            return_value=tmp_locus_home / "models" / "embedding",
        ),
        patch("locus.routers.health.is_embedding_model_downloaded", return_value=True),
    ):
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["dependencies"]["embedding_model"] == {
        "status": "ok",
        "message": str(model_file),
    }


@pytest.mark.asyncio
async def test_llm_health_requires_base_url():
    cfg = type(
        "Cfg",
        (),
        {"ai": AIConfig(provider="openai", model="gpt-4o", api_key="sk-test", base_url="")},
    )()

    result = await _check_llm(cfg)

    assert result == {"status": "error", "message": "base_url not configured"}


@pytest.mark.asyncio
async def test_llm_health_validates_openai_compatible_response_shape():
    class DummyResponse:
        status_code = 200

        def json(self):
            return {"unexpected": []}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            return DummyResponse()

    cfg = type(
        "Cfg",
        (),
        {
            "ai": AIConfig(
                provider="custom",
                model="gpt-4o",
                api_key="sk-test",
                base_url="http://localhost:8000/v1",
            )
        },
    )()

    with patch("locus.routers.health.httpx.AsyncClient", return_value=DummyClient()):
        result = await _check_llm(cfg)

    assert result == {"status": "error", "message": "Invalid models response"}


def test_is_embedding_model_downloaded_false_when_absent(tmp_path: Path):
    from locus.pipeline.embed import is_embedding_model_downloaded

    with patch("locus.pipeline.embed._embedding_model_dir", return_value=tmp_path):
        assert is_embedding_model_downloaded() is False


def test_is_embedding_model_downloaded_true_when_present(tmp_path: Path):
    from locus.pipeline.embed import is_embedding_model_downloaded

    model_file = tmp_path / "onnx" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fake")

    with patch("locus.pipeline.embed._embedding_model_dir", return_value=tmp_path):
        assert is_embedding_model_downloaded() is True


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
    client.put(
        "/config",
        json={
            "ai": {"api_key": "sk-secret"},
            "accounts": {"bilibili": {"cookie": "my-cookie"}},
        },
    )
    data = client.get("/config").json()
    assert data["ai"]["api_key"] == "***"
    assert data["accounts"]["bilibili"]["cookie"] == "***"


def test_serves_built_spa_without_shadowing_api_routes(client: TestClient, tmp_path: Path):
    spa_dist = tmp_path / "web-dist"
    (spa_dist / "assets").mkdir(parents=True)
    (spa_dist / "index.html").write_text("<!doctype html><html><body>locus web</body></html>")
    (spa_dist / "assets" / "app.js").write_text("console.log('locus');")

    with patch("locus.main.WEB_DIST", spa_dist):
        from locus.main import create_app

        app = create_app()

    with TestClient(app, raise_server_exceptions=True) as spa_client:
        root = spa_client.get("/")
        assert root.status_code == 200
        assert "text/html" in root.headers["content-type"]
        assert "locus web" in root.text

        asset = spa_client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "console.log('locus');" in asset.text

        health = spa_client.get("/health")
        assert health.status_code == 200
        assert health.json()["dependencies"]["backend"]["status"] == "ok"


def test_locus_dirs_bootstrapped(client: TestClient, tmp_locus_home: Path):
    for subdir in ("notes", "transcripts", "downloads", "chroma"):
        assert (tmp_locus_home / subdir).is_dir(), f"Missing {subdir}/"
    assert (tmp_locus_home / "locus.db").exists()


@pytest.mark.asyncio
async def test_bootstrap_db_creates_lists_table(tmp_path: Path):
    import aiosqlite

    from locus.db import bootstrap_db

    with patch("locus.db.locus_home", return_value=tmp_path):
        await bootstrap_db()

    async with aiosqlite.connect(tmp_path / "locus.db") as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lists'"
        ) as cur:
            row = await cur.fetchone()

    assert row is not None

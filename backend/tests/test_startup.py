from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from bibilab.config import AIConfig
from bibilab.routers.health import _check_llm


@pytest.mark.asyncio
async def test_health_returns_200(client: httpx.AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall"] in ("ok", "error", "degraded")
    deps = data["dependencies"]
    assert "backend" in deps
    assert deps["backend"]["status"] == "ok"
    assert all(k in deps for k in ("llm", "asr_model", "ffmpeg", "cuda", "embedding_model"))
    assert "bilibili_session" not in deps


@pytest.mark.asyncio
async def test_health_includes_embedding_model(client: httpx.AsyncClient, tmp_bibilab_home: Path):  # noqa: ARG001
    resp = await client.get("/health")
    assert resp.status_code == 200
    deps = resp.json()["dependencies"]
    assert "embedding_model" in deps
    assert deps["embedding_model"]["status"] in ("ok", "error")


@pytest.mark.asyncio
async def test_health_reports_ffmpeg_install_path(client: httpx.AsyncClient):
    with patch("bibilab.routers.health.shutil.which", return_value="/usr/bin/ffmpeg"):
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["dependencies"]["ffmpeg"] == {
        "status": "ok",
        "message": "/usr/bin/ffmpeg",
    }


@pytest.mark.asyncio
async def test_health_reports_embedding_model_install_path(tmp_bibilab_home: Path, client: httpx.AsyncClient):  # noqa: ARG001
    model_file = tmp_bibilab_home / "models" / "embedding" / "onnx" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fake")

    with (
        patch(
            "bibilab.routers.health._embedding_model_dir",
            return_value=tmp_bibilab_home / "models" / "embedding",
        ),
        patch("bibilab.routers.health.is_embedding_model_downloaded", return_value=True),
    ):
        resp = await client.get("/health")

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
        {"ai": AIConfig(protocol="openai", model="gpt-4o", api_key="sk-test", base_url="")},
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

        async def get(self, url, headers=None, follow_redirects=True):
            return DummyResponse()

    cfg = type(
        "Cfg",
        (),
        {
            "ai": AIConfig(
                protocol="openai",
                model="gpt-4o",
                api_key="sk-test",
                base_url="http://localhost:8000/v1",
            )
        },
    )()

    with patch("bibilab.routers.health.httpx.AsyncClient", return_value=DummyClient()):
        result = await _check_llm(cfg)

    assert result == {"status": "ok", "message": "http://localhost:8000/v1"}


def test_is_embedding_model_downloaded_false_when_absent(tmp_path: Path):
    from bibilab.pipeline.embed import is_embedding_model_downloaded

    with patch("bibilab.pipeline.embed._embedding_model_dir", return_value=tmp_path):
        assert is_embedding_model_downloaded() is False


def test_is_embedding_model_downloaded_true_when_present(tmp_path: Path):
    from bibilab.pipeline.embed import is_embedding_model_downloaded

    model_file = tmp_path / "onnx" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"fake")

    with patch("bibilab.pipeline.embed._embedding_model_dir", return_value=tmp_path):
        assert is_embedding_model_downloaded() is True


@pytest.mark.asyncio
async def test_config_defaults(client: httpx.AsyncClient):
    resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"]["port"] == 8765
    assert data["ai"]["protocol"] == "openai"
    assert data["transcription"]["model_size"] == "large-v3"


@pytest.mark.asyncio
async def test_config_deep_merge(client: httpx.AsyncClient):
    resp = await client.put("/config", json={"ai": {"model": "gpt-4-turbo"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ai"]["model"] == "gpt-4-turbo"
    assert data["ai"]["protocol"] == "openai"  # sibling preserved
    assert data["backend"]["port"] == 8765  # unrelated section preserved


@pytest.mark.asyncio
async def test_config_masks_sensitive_fields(client: httpx.AsyncClient):
    await client.put(
        "/config",
        json={
            "ai": {"api_key": "sk-secret"},
            "accounts": {"bilibili": {"cookie": "my-cookie"}},
        },
    )
    data = (await client.get("/config")).json()
    assert data["ai"]["api_key"] == "***"
    assert data["accounts"]["bilibili"]["cookie"] == "***"


@pytest.mark.asyncio
async def test_put_config_with_masked_values_preserves_real_secrets(client: httpx.AsyncClient):
    await client.put(
        "/config",
        json={
            "ai": {"api_key": "sk-real-key"},
            "accounts": {"bilibili": {"cookie": "real-cookie"}},
        },
    )
    await client.put("/config", json={"ai": {"api_key": "***"}, "accounts": {"bilibili": {"cookie": "***"}}})
    data = (await client.get("/config")).json()
    assert data["ai"]["api_key"] == "***"
    assert data["accounts"]["bilibili"]["cookie"] == "***"

    from bibilab.config import get_config

    real_cfg = get_config()
    assert real_cfg.ai.api_key == "sk-real-key"
    assert real_cfg.accounts.bilibili.cookie == "real-cookie"


@pytest.mark.asyncio
async def test_serves_built_spa_without_shadowing_api_routes(tmp_bibilab_home: Path, tmp_path: Path):  # noqa: ARG001
    spa_dist = tmp_path / "web-dist"
    (spa_dist / "assets").mkdir(parents=True)
    (spa_dist / "index.html").write_text("<!doctype html><html><body>bibilab web</body></html>")
    (spa_dist / "assets" / "app.js").write_text("console.log('bibilab');")

    with patch("bibilab.main.WEB_DIST", spa_dist):
        from bibilab.main import create_app

        app = create_app(start_worker=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as spa_client:
            root = await spa_client.get("/")
            assert root.status_code == 200
            assert "text/html" in root.headers["content-type"]
            assert "bibilab web" in root.text

            asset = await spa_client.get("/assets/app.js")
            assert asset.status_code == 200
            assert "console.log('bibilab');" in asset.text

            health = await spa_client.get("/health")
            assert health.status_code == 200
            assert health.json()["dependencies"]["backend"]["status"] == "ok"


@pytest.mark.asyncio
async def test_bibilab_dirs_bootstrapped(client: httpx.AsyncClient, tmp_bibilab_home: Path):
    for subdir in ("covers", "transcripts", "downloads", "chroma"):
        assert (tmp_bibilab_home / subdir).is_dir(), f"Missing {subdir}/"
    assert (tmp_bibilab_home / "bibilab.db").exists()


@pytest.mark.asyncio
async def test_bootstrap_db_creates_lists_table(tmp_path: Path):
    import sqlite3

    from bibilab.db import bootstrap_db

    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        await bootstrap_db()

    with sqlite3.connect(tmp_path / "bibilab.db") as db:
        row = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lists'").fetchone()

    assert row is not None

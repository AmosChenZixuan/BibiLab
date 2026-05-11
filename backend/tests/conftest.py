from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _clear_llm_client_caches():
    """Prevent MagicMock accumulation in module-level client caches across tests."""
    yield
    from bibilab.pipeline._shared import _async_client_cache, _client_cache

    _async_client_cache.clear()
    _client_cache.clear()


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    from bibilab.config import _reset_cache

    _reset_cache()
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.main.bibilab_home", return_value=tmp_path):
            with patch("bibilab.cleanup.bibilab_home", return_value=tmp_path):
                with patch("bibilab.routers.lists.bibilab_home", return_value=tmp_path):
                    with patch("bibilab.routers.artifacts.bibilab_home", return_value=tmp_path):
                        with patch("bibilab.worker.bibilab_home", return_value=tmp_path):
                            with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
                                with patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path):
                                    with patch("bibilab.routers.sources.bibilab_home", return_value=tmp_path):
                                        with patch("bibilab.adapters.bilibili.bibilab_home", return_value=tmp_path):
                                            with patch("pathlib.Path.home", return_value=tmp_path):
                                                yield tmp_path


@pytest.fixture(autouse=True)
async def _add_tool_blocks_column(tmp_bibilab_home: Path, monkeypatch):
    """Ensure tool_blocks column exists in test DBs (added via ALTER after bootstrap_db)."""
    from bibilab.db import bootstrap_db, get_db

    await bootstrap_db()
    async with get_db() as db:
        try:
            await db.execute("ALTER TABLE messages ADD COLUMN tool_blocks TEXT")
            await db.commit()
        except Exception:
            pass  # column already exists (or SQLite version doesn't support ALTER ADD COLUMN)


@pytest_asyncio.fixture()
async def client(tmp_bibilab_home: Path):  # noqa: ARG001
    from bibilab.main import create_app

    with patch("bibilab.main.WEB_DIST", tmp_bibilab_home / "web-dist-disabled"):
        app = create_app(start_worker=False)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                yield async_client
                from bibilab.pipeline.chat_runs import ChatRunRegistry, get_chat_run_registry

                old = get_chat_run_registry()
                for _, task in old.all_tasks():
                    if not task.done():
                        task.cancel()
                for task in old.all_background_tasks():
                    if not task.done():
                        task.cancel()
                # Replace with fresh registry for next test
                import bibilab.pipeline.chat_runs as cr

                cr._registry = ChatRunRegistry()

                from bibilab.pipeline._shared import _async_client_cache, _client_cache

                _async_client_cache.clear()
                _client_cache.clear()

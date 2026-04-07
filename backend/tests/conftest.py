from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio


@pytest.fixture()
def tmp_bibilab_home(tmp_path: Path):
    with patch("bibilab.config.bibilab_home", return_value=tmp_path):
        with patch("bibilab.main.bibilab_home", return_value=tmp_path):
            with patch("bibilab.cleanup.bibilab_home", return_value=tmp_path):
                with patch("bibilab.routers.lists.bibilab_home", return_value=tmp_path):
                    with patch("bibilab.worker.bibilab_home", return_value=tmp_path):
                        with patch("bibilab.pipeline.transcribe.bibilab_home", return_value=tmp_path):
                            with patch("bibilab.pipeline.embed.bibilab_home", return_value=tmp_path):
                                with patch("pathlib.Path.home", return_value=tmp_path):
                                    yield tmp_path


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

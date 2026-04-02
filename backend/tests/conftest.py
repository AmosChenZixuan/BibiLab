from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio


@pytest.fixture()
def tmp_locus_home(tmp_path: Path):
    with patch("locus.config.locus_home", return_value=tmp_path):
        with patch("locus.db.locus_home", return_value=tmp_path):
            with patch("locus.main.locus_home", return_value=tmp_path):
                with patch("locus.cleanup.locus_home", return_value=tmp_path):
                    yield tmp_path


@pytest_asyncio.fixture()
async def client(tmp_locus_home: Path):  # noqa: ARG001
    from locus.main import create_app

    with patch("locus.main.WEB_DIST", tmp_locus_home / "web-dist-disabled"):
        app = create_app(start_worker=False)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                yield async_client

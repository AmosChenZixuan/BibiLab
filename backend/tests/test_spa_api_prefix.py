"""Single-port production serves the SPA and the API on one origin. The SPA calls
/api/* (api client prefixes origin + "/api"); in dev Vite's proxy strips /api, but
in production the backend must strip it itself or every API call 404/405s. This
guards that strip — it only activates on the SPA-serving path (web/dist present)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture()
async def spa_client(tmp_bibilab_home: Path):  # noqa: ARG001
    from bibilab.main import create_app

    # A web/dist with index.html flips create_app into single-port SPA mode, which
    # registers the /api-strip middleware. assets/ stays absent — not needed here.
    web_dist = tmp_bibilab_home / "web-dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<!doctype html><title>spa</title>")

    with patch("bibilab.main.WEB_DIST", web_dist):
        app = create_app(start_worker=False)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                yield c


async def test_api_prefix_is_stripped_to_root_router(spa_client: httpx.AsyncClient):
    # The SPA's real call shape. Without the strip this 404s (catch-all) — health
    # router mounts at /health, not /api/health.
    prefixed = await spa_client.get("/api/health")
    assert prefixed.status_code == 200
    assert prefixed.json()["overall"] in {"ok", "error"}


async def test_root_path_still_served_without_prefix(spa_client: httpx.AsyncClient):
    # install.sh probes /health directly (no /api) — the strip must not break that.
    assert (await spa_client.get("/health")).status_code == 200


async def test_non_api_path_falls_through_to_spa(spa_client: httpx.AsyncClient):
    # A client-side route returns index.html, not a 404 — strip only touches /api.
    res = await spa_client.get("/lists/abc")
    assert res.status_code == 200
    assert "<title>spa</title>" in res.text

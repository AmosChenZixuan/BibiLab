"""Single-port production serves the SPA and the API on one origin. The SPA calls
/api/* (api client prefixes origin + "/api"); in dev Vite's proxy strips /api. In
production the backend mounts the API as a sub-application at /api (create_app's
SPA-serving path), so /api/* reaches the same routers. These guard that mount —
it only exists on the SPA-serving path (web/dist present)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
    # mounts the API sub-app at /api. assets/ stays absent — not needed here.
    web_dist = tmp_bibilab_home / "web-dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text("<!doctype html><title>spa</title>")

    with patch("bibilab.main.WEB_DIST", web_dist):
        app = create_app(start_worker=False)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                yield c


async def test_api_prefix_routes_to_mounted_api(spa_client: httpx.AsyncClient):
    # The SPA's real call shape. Routed through the /api mount to the same routers;
    # health mounts at /health under the sub-app, i.e. /api/health from outside.
    prefixed = await spa_client.get("/api/health")
    assert prefixed.status_code == 200
    assert prefixed.json()["overall"] in {"ok", "error"}


async def test_root_path_still_served_without_prefix(spa_client: httpx.AsyncClient):
    # install.sh probes /health directly (no /api) — root include must not break that.
    assert (await spa_client.get("/health")).status_code == 200


async def test_non_api_path_falls_through_to_spa(spa_client: httpx.AsyncClient):
    # A client-side route returns index.html, not a 404 — the mount only owns /api.
    res = await spa_client.get("/lists/abc")
    assert res.status_code == 200
    assert "<title>spa</title>" in res.text


async def test_cover_url_is_api_prefixed(spa_client: httpx.AsyncClient):
    # The served list response carries an /api-prefixed cover URL (built from the
    # API_PREFIX literal, not url_for — url_for can't emit /api when routers are
    # registered at both root and the mount), so the SPA's /api/* client reaches it.
    from bibilab.config import cover_path
    from bibilab.db import create_list
    from tests.factories import SourceFactory

    list_id = str(uuid.uuid4())
    await create_list(list_id, "L", datetime.now(timezone.utc).isoformat())
    source_id = await SourceFactory.build(list_id)  # first source auto-assigned as thumbnail
    cover_path(source_id).parent.mkdir(parents=True, exist_ok=True)
    cover_path(source_id).write_bytes(b"jpg")

    res = await spa_client.get("/api/lists")
    assert res.status_code == 200
    (lst,) = res.json()
    assert f"/api/sources/{source_id}/cover" in lst["thumbnail_url"]


async def test_jobs_cancel_via_mount_reaches_worker(spa_client: httpx.AsyncClient):
    # Regression guard: jobs.py reads request.app.state.worker. Under the mount,
    # request.app is the sub-app — it must share the parent's state or this 500s
    # with AttributeError on a non-terminal job. start_worker=False ⇒ worker is None,
    # so a reachable state attribute yields 204; an absent one (broken mount) raises.
    from bibilab.db import create_job, update_job_status

    job_id = await create_job("ingest", {"list_id": "x"})
    await update_job_status(job_id, "downloading", progress=10)

    res = await spa_client.delete(f"/api/jobs/{job_id}")
    assert res.status_code == 204

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bibilab.config import bibilab_home, load_config
from bibilab.db import bootstrap_db, get_db
from bibilab.pipeline.chat_runs import get_chat_run_registry
from bibilab.routers.artifacts import router as artifacts_router
from bibilab.routers.auth import router as auth_router
from bibilab.routers.chat import debug_router
from bibilab.routers.chat import router as chat_router
from bibilab.routers.config_router import router as config_router
from bibilab.routers.eval import router as eval_router
from bibilab.routers.health import router as health_router
from bibilab.routers.ingest import router as ingest_router
from bibilab.routers.jobs import router as jobs_router
from bibilab.routers.lists import router as lists_router
from bibilab.routers.models import router as models_router
from bibilab.routers.proxy import router as proxy_router
from bibilab.routers.sources import router as sources_router
from bibilab.worker import WorkerLoop

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

logger = logging.getLogger(__name__)

SHUTDOWN_DRAIN_TIMEOUT = 5.0


async def sweep_orphaned_streams() -> None:
    """Mark messages stuck mid-turn as failed.

    Called at app startup before accepting requests. Registry is empty at
    startup, so any row whose status is in IN_FLIGHT_MESSAGE_STATUSES
    ('streaming' for the assistant, 'pending' for the user awaiting it)
    was abandoned by the previous process — both rows of the orphaned
    turn are flipped in one UPDATE.
    """
    from bibilab.db import IN_FLIGHT_MESSAGE_STATUSES, _in_placeholders

    in_flight_placeholders = _in_placeholders(IN_FLIGHT_MESSAGE_STATUSES)
    async with get_db() as db:
        await db.execute(
            f"UPDATE messages SET status='failed', error='Server restarted during generation' "
            f"WHERE status IN ({in_flight_placeholders})",
            IN_FLIGHT_MESSAGE_STATUSES,
        )
        await db.execute(
            "UPDATE conversations SET active_stream_message_id=NULL WHERE active_stream_message_id IS NOT NULL"
        )
        await db.commit()


async def _await_all_registered(registry) -> None:
    """Wait for all registered tasks to finish (after cancellation)."""
    for msg_id, task in registry.all_tasks():
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("Task %s raised during shutdown drain", msg_id, exc_info=True)

    for task in registry.all_background_tasks():
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("Background task raised during shutdown drain", exc_info=True)


def make_lifespan(*, start_worker: bool) -> Callable[[], AsyncGenerator[None, None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        home = bibilab_home()
        for subdir in (
            "covers",
            "downloads",
            "chroma",
            "tmp",
            "models/asr",
            "models/embedding",
            "artifacts",
        ):
            (home / subdir).mkdir(parents=True, exist_ok=True)

        await bootstrap_db()
        await sweep_orphaned_streams()

        worker = None
        if start_worker:
            cfg = load_config()
            worker = WorkerLoop(
                concurrency=cfg.backend.max_concurrent_jobs,
            )
            await worker.start()
        app.state.worker = worker

        yield

        if worker is not None:
            await worker.stop()

        registry = get_chat_run_registry()
        for msg_id in registry.all_message_ids():
            registry.cancel(msg_id)
        try:
            await asyncio.wait_for(_await_all_registered(registry), timeout=SHUTDOWN_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("shutdown drain timed out — orphans will be cleaned by startup sweep")

    return lifespan


def _include_api_routers(target: FastAPI) -> None:
    target.include_router(health_router)
    target.include_router(auth_router)
    target.include_router(config_router)
    target.include_router(jobs_router)
    target.include_router(lists_router)
    target.include_router(ingest_router)
    target.include_router(artifacts_router)
    target.include_router(chat_router)
    target.include_router(debug_router)
    target.include_router(proxy_router)
    target.include_router(sources_router)
    target.include_router(models_router)
    target.include_router(eval_router)


def create_app(*, start_worker: bool = True) -> FastAPI:
    cfg = load_config()
    app = FastAPI(title="Bibilab Backend", lifespan=make_lifespan(start_worker=start_worker))
    web_dist = WEB_DIST

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.backend.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root mount: dev (Vite proxies /api → here, prefix already stripped) + install.sh /health.
    _include_api_routers(app)

    if web_dist.joinpath("index.html").exists():
        # Single-port production: the SPA is served from the same origin as the API and
        # calls /api/* (the api client prefixes window.location.origin + "/api"). In dev,
        # Vite's proxy strips /api before forwarding. In production the same routers are
        # mounted again as a sub-application under /api, so /api/* reaches them with no
        # request-scope mutation. The sub-app shares the parent's state so handlers that
        # read app.state (jobs.py → worker) work under the mount. Mounted before the
        # catch-all below so /api/* isn't swallowed by the SPA fallback.
        api = FastAPI()
        api.state = app.state
        _include_api_routers(api)
        app.mount("/api", api)

        assets_dir = web_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_spa_root() -> FileResponse:
            return FileResponse(web_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            return FileResponse(web_dist / "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    import os

    import uvicorn

    cfg = load_config()
    # The container pins the bind port via BIBILAB_PORT and maps a fixed host port to
    # it. config.json is bind-mounted from the host and shared with a native install;
    # a custom backend.port there would otherwise desync the in-container bind from the
    # fixed port-mapping. Native runs leave BIBILAB_PORT unset and honor config.json.
    port = int(os.environ.get("BIBILAB_PORT", cfg.backend.port))
    uvicorn.run(app, host="0.0.0.0", port=port)

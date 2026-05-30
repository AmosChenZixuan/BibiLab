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
from bibilab.routers.chat import router as chat_router
from bibilab.routers.config_router import router as config_router
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
    """Mark messages stuck at status='streaming' as failed.

    Called at app startup before accepting requests. Registry is empty at
    startup, so any 'streaming' row was abandoned by the previous process.
    """
    async with get_db() as db:
        await db.execute(
            "UPDATE messages SET status='failed', error='Server restarted during generation' WHERE status='streaming'"
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
            "transcripts",
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
            worker = WorkerLoop(concurrency=cfg.backend.max_concurrent_jobs)
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

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(config_router)
    app.include_router(jobs_router)
    app.include_router(lists_router)
    app.include_router(ingest_router)
    app.include_router(artifacts_router)
    app.include_router(chat_router)
    app.include_router(proxy_router)
    app.include_router(sources_router)
    app.include_router(models_router)

    if web_dist.joinpath("index.html").exists():
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
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host="0.0.0.0", port=cfg.backend.port)

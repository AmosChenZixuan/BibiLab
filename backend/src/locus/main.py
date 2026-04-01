from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from locus.config import load_config, locus_home
from locus.db import bootstrap_db
from locus.routers.config_router import router as config_router
from locus.routers.health import router as health_router
from locus.routers.ingest import router as ingest_router
from locus.routers.jobs import router as jobs_router
from locus.routers.lists import router as lists_router
from locus.routers.notes import router as notes_router
from locus.routers.whisper import router as whisper_router
from locus.worker import WorkerLoop

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    home = locus_home()
    for subdir in (
        "notes",
        "transcripts",
        "downloads",
        "chroma",
        "tmp",
        "models/whisper",
        "models/embedding",
    ):
        (home / subdir).mkdir(parents=True, exist_ok=True)

    await bootstrap_db()

    cfg = load_config()
    worker = WorkerLoop(concurrency=cfg.backend.worker_concurrency)
    app.state.worker = worker
    await worker.start()

    yield

    await worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Locus Backend", lifespan=lifespan)
    web_dist = WEB_DIST

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:5173",
            "http://127.0.0.1",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(config_router)
    app.include_router(jobs_router)
    app.include_router(lists_router)
    app.include_router(ingest_router)
    app.include_router(notes_router)
    app.include_router(whisper_router)

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

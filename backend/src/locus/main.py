from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    return app


app = create_app()

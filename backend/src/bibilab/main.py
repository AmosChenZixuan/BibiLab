from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bibilab.config import bibilab_home, load_config
from bibilab.db import bootstrap_db
from bibilab.routers.config_router import router as config_router
from bibilab.routers.health import router as health_router
from bibilab.routers.ingest import router as ingest_router
from bibilab.routers.jobs import router as jobs_router
from bibilab.routers.lists import router as lists_router
from bibilab.routers.notes import router as notes_router
from bibilab.routers.whisper import router as whisper_router
from bibilab.worker import WorkerLoop

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def make_lifespan(*, start_worker: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        home = bibilab_home()
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

        worker = None
        if start_worker:
            cfg = load_config()
            worker = WorkerLoop(concurrency=cfg.backend.worker_concurrency)
            await worker.start()
        app.state.worker = worker

        yield

        if worker is not None:
            await worker.stop()

    return lifespan


def create_app(*, start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="Bibilab Backend", lifespan=make_lifespan(start_worker=start_worker))
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)

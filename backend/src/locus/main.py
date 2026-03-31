from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from locus.config import locus_home
from locus.db import bootstrap_db
from locus.routers.config_router import router as config_router
from locus.routers.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Bootstrap ~/.locus/ directory layout
    home = locus_home()
    for subdir in ("transcripts", "downloads", "chroma"):
        (home / subdir).mkdir(parents=True, exist_ok=True)

    await bootstrap_db()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Locus Backend", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["app://obsidian.md", "http://localhost", "http://127.0.0.1"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(config_router)

    return app


app = create_app()

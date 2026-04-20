"""FastAPI entry-point for SynapseMem."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import chat as chat_routes
from .api import ingest as ingest_routes
from .api import query as query_routes
from .config import settings
from .database import init_db

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "public"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="SynapseMem",
    version=__version__,
    description=(
        "Cognitive-inspired memory for coding agents. Ingests source files, "
        "extracts structured facts (symbols, imports, promises), and answers "
        "queries about the project graph."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": __version__,
        "embeddings_model": settings.embeddings_model,
        "extractor_model": settings.extractor_model,
    }


app.include_router(ingest_routes.router)
app.include_router(query_routes.router)
app.include_router(chat_routes.router)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

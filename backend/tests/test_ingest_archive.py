"""Tests for the /ingest/archive endpoint (zip upload)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client(isolated_data_dir: Path):
    # Build a fresh app instance tied to the isolated DB (same env tricks as
    # the `session` fixture in conftest.py).
    from app import config as config_mod
    from app import database as db_mod
    from app import models  # noqa: F401 — register tables

    config_mod.settings = config_mod.Settings()
    db_mod.engine = db_mod.create_async_engine(
        config_mod.settings.resolved_database_url(),
        echo=False,
        future=True,
    )
    db_mod.SessionLocal = db_mod.async_sessionmaker(
        db_mod.engine,
        class_=db_mod.AsyncSession,
        expire_on_commit=False,
    )

    from app.database import Base

    async with db_mod.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c

    await db_mod.engine.dispose()


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_ingest_archive_happy_path(client) -> None:
    body = _make_zip(
        {
            "proj/utils.py": "def helper():\n    return 1\n",
            "proj/app.py": "from utils import helper\n\ndef go():\n    return helper()\n",
            "proj/.venv/ignored.py": "# should be skipped\n",
            "proj/README.md": "not python\n",
        }
    )
    files = {"file": ("proj.zip", body, "application/zip")}
    data = {"project_id": "zip-demo", "language": "python"}
    r = await client.post("/ingest/archive", data=data, files=files)
    assert r.status_code == 200, r.text
    payload = r.json()
    paths = {item["path"] for item in payload["ingested"]}
    assert "utils.py" in paths
    assert "app.py" in paths
    # .venv and non-python files must be filtered out.
    assert not any(".venv" in p for p in paths)
    assert not any(p.endswith(".md") for p in paths)


@pytest.mark.asyncio
async def test_ingest_archive_rejects_non_zip(client) -> None:
    files = {"file": ("not-a-zip.txt", b"hello", "text/plain")}
    data = {"project_id": "x", "language": "python"}
    r = await client.post("/ingest/archive", data=data, files=files)
    assert r.status_code == 400

"""Shared pytest fixtures — isolated SQLite per test."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SYNAPSEMEM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(
        "SYNAPSEMEM_DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
    )
    return tmp_path


@pytest_asyncio.fixture
async def session(isolated_data_dir: Path) -> AsyncIterator:
    # Import inside the fixture so config + engine pick up env vars.
    from app import config as config_mod
    from app import database as db_mod
    from app import models  # noqa: F401 — register tables

    # Rebuild settings + engine fresh for this test's DB URL.
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

    # Use SQLAlchemy's metadata from the current Base binding.
    from app.database import Base  # re-import after engine swap

    async with db_mod.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with db_mod.session_scope() as s:
        yield s

    await db_mod.engine.dispose()

    # Clean env so next test doesn't inherit.
    for key in ("SYNAPSEMEM_DATA_DIR", "SYNAPSEMEM_DATABASE_URL"):
        os.environ.pop(key, None)

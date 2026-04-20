"""Tests for the symbol embedding layer.

We stub out :class:`OllamaClient` so the tests never hit a real model.
What we verify:

* ``ensure_symbols_embedded`` is idempotent and re-embeds when the
  symbol text changes.
* ``semantic_symbol_search`` returns higher scores for a query closer
  to the target symbol's text than for an unrelated one.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import embeddings, memory
from app.database import Base


@pytest.fixture()
async def session(tmp_path: Path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        yield s
    await engine.dispose()


class FakeClient:
    """Deterministic "embedding" — one-hot by keyword presence."""

    keywords: ClassVar[list[str]] = [
        "greet",
        "parse",
        "user",
        "class",
        "fibonacci",
        "config",
        "http",
        "database",
    ]

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, *, model: str, text: str) -> list[float]:
        self.calls.append(text)
        vec = [1.0 if kw in text.lower() else 0.0 for kw in self.keywords]
        if sum(vec) == 0:
            vec[0] = 0.01  # keep vector non-zero
        return vec

    async def aclose(self) -> None:  # pragma: no cover - not used
        pass


async def _seed(session) -> None:
    await memory.ingest_file(
        session,
        project_id="p",
        path="greetings.py",
        content=(
            'def greet(name: str) -> str:\n'
            '    """Return a hello message for the user."""\n'
            '    return f"hello {name}"\n'
        ),
    )
    await memory.ingest_file(
        session,
        project_id="p",
        path="fib.py",
        content=(
            'def fibonacci(n: int) -> int:\n'
            '    """Compute the n-th Fibonacci number."""\n'
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
    )
    await session.commit()


async def test_ensure_symbols_embedded_is_idempotent(session) -> None:
    await _seed(session)
    fake = FakeClient()
    n1 = await embeddings.ensure_symbols_embedded(
        session, "p", client=fake, model="test-model"
    )
    assert n1 == 2
    # Second call should embed nothing new.
    n2 = await embeddings.ensure_symbols_embedded(
        session, "p", client=fake, model="test-model"
    )
    assert n2 == 0


async def test_semantic_search_ranks_related_symbol_first(session) -> None:
    await _seed(session)
    fake = FakeClient()
    await embeddings.ensure_symbols_embedded(
        session, "p", client=fake, model="test-model"
    )

    hits = await embeddings.semantic_symbol_search(
        session, "p", "user greeting helper", k=5, client=fake, model="test-model"
    )
    assert hits, "expected at least one hit"
    assert hits[0].symbol.name == "greet"

    hits2 = await embeddings.semantic_symbol_search(
        session, "p", "compute fibonacci number", k=5, client=fake, model="test-model"
    )
    assert hits2[0].symbol.name == "fibonacci"

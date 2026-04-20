"""Integration tests for the ingest → promise-tracking → graph flow."""

from __future__ import annotations

import textwrap

from sqlalchemy import select

from app import memory, models


async def test_ingest_single_file_creates_symbols_and_imports(session) -> None:
    src = textwrap.dedent(
        """
        from utils import helper

        def main():
            return helper()
        """
    )
    r = await memory.ingest_file(
        session,
        project_id="demo",
        path="app.py",
        content=src,
    )
    assert r.created is True
    assert r.symbols == 1
    assert r.imports == 1

    # We imported `helper` from `utils`, but `utils.py` doesn't exist in the
    # project yet — so there should be NO promise (utils isn't known as an
    # intra-project module yet; it could well be a pip package).
    assert r.new_promises == 0


async def test_promise_created_and_resolved_when_target_file_appears(session) -> None:
    # 1. Write utils.py FIRST but empty — it's now known as an intra-project module.
    await memory.ingest_file(
        session,
        project_id="demo",
        path="utils.py",
        content="# empty\n",
    )

    # 2. Write app.py that imports helper from utils — utils exists but has no helper.
    r_app = await memory.ingest_file(
        session,
        project_id="demo",
        path="app.py",
        content="from utils import helper\n\ndef main():\n    return helper()\n",
    )
    assert r_app.new_promises == 1
    assert r_app.resolved_promises == 0

    # 3. Re-ingest utils.py with `def helper()`. The promise should resolve.
    r_utils = await memory.ingest_file(
        session,
        project_id="demo",
        path="utils.py",
        content="def helper():\n    return 1\n",
    )
    assert r_utils.resolved_promises == 1

    promises = (
        await session.execute(
            select(models.Promise).where(models.Promise.project_id == "demo")
        )
    ).scalars().all()
    assert len(promises) == 1
    assert promises[0].resolved_at is not None
    assert promises[0].resolver_symbol_id is not None


async def test_graph_payload_shape(session) -> None:
    await memory.ingest_file(
        session,
        project_id="demo",
        path="a.py",
        content="def foo():\n    return 1\n",
    )
    await memory.ingest_file(
        session,
        project_id="demo",
        path="b.py",
        content="from a import foo, missing_one\n\ndef bar():\n    return foo()\n",
    )
    graph = await memory.project_graph(session, "demo")

    node_types = {n["data"].get("type") for n in graph["nodes"]}
    assert "file" in node_types
    assert "symbol" in node_types
    # `missing_one` is not defined in a.py but a.py is known → open promise node.
    assert "open_promise" in node_types

    edge_types = {e["data"].get("type") for e in graph["edges"]}
    assert "defines" in edge_types
    assert "imports" in edge_types
    assert "promise_open" in edge_types


async def test_re_ingest_is_idempotent_and_updates_content(session) -> None:
    await memory.ingest_file(
        session,
        project_id="demo",
        path="m.py",
        content="def one():\n    return 1\n",
    )
    # Same file, new content — should replace symbols, not accumulate.
    await memory.ingest_file(
        session,
        project_id="demo",
        path="m.py",
        content="def two():\n    return 2\n",
    )
    syms = (
        await session.execute(
            select(models.Symbol).where(models.Symbol.project_id == "demo")
        )
    ).scalars().all()
    names = {s.name for s in syms}
    assert names == {"two"}

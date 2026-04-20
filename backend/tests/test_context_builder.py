"""Tests for the chat context assembler."""

from __future__ import annotations

from app import memory
from app.context_builder import build_chat_context


async def test_context_includes_open_promises_and_symbols(session) -> None:
    project_id = "ctx-proj"
    # utils.py exists but empty → known intra-project module
    await memory.ingest_file(
        session,
        project_id=project_id,
        path="utils.py",
        content="# empty\n",
    )
    # main.py imports helper from utils → opens a promise
    await memory.ingest_file(
        session,
        project_id=project_id,
        path="main.py",
        content="from utils import helper\n\ndef go():\n    return helper()\n",
    )
    ctx = await build_chat_context(session, project_id, "where is helper?")
    assert "## Project" in ctx
    assert "## Open promises" in ctx
    assert "helper" in ctx  # open promise mentions it

    # Resolve the promise by defining helper
    await memory.ingest_file(
        session,
        project_id=project_id,
        path="utils.py",
        content='def helper() -> int:\n    """Return 1."""\n    return 1\n',
    )
    ctx = await build_chat_context(session, project_id, "tell me about helper")
    assert "Relevant symbols" in ctx
    assert "helper" in ctx


async def test_context_empty_for_unknown_project(session) -> None:
    ctx = await build_chat_context(session, "no-such-project", "hello?")
    assert ctx == ""

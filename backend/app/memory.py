"""Core memory operations — ingest, resolve promises, query graph."""

from __future__ import annotations

import hashlib
import posixpath
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import models
from .ast_extractor import ExtractedImport, ExtractedSymbol, extract_python


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _resolve_relative_module(source_path: str, module: str) -> str:
    """Resolve ``from .sibling import x`` against the file's own location.

    For ``module='..pkg.mod'`` in file ``a/b/c.py``, returns ``a/pkg/mod``.
    Absolute imports (no leading dot) are returned unchanged.
    """
    if not module.startswith("."):
        return module
    dots = len(module) - len(module.lstrip("."))
    remainder = module[dots:]
    parts = source_path.split("/")
    # drop the file itself, then go up (dots-1) more levels
    base = parts[:-1]
    up = max(0, dots - 1)
    if up:
        base = base[: len(base) - up] if up <= len(base) else []
    if remainder:
        return posixpath.normpath("/".join([*base, *remainder.split(".")]))
    return posixpath.normpath("/".join(base) or ".")


def _module_candidates_for_file(path: str) -> list[str]:
    """What module names could reach this file?

    For ``utils.py`` we accept ``utils`` as a candidate; for ``pkg/utils.py``
    we accept ``pkg.utils``, ``pkg/utils`` and ``utils`` (flat layout). The
    resolver tries every candidate against recorded imports.
    """
    base = path[:-3] if path.endswith(".py") else path
    base = base.replace("\\", "/").strip("/")
    if base.endswith("/__init__"):
        base = base[: -len("/__init__")]
    candidates = {base, base.replace("/", ".")}
    tail = base.rsplit("/", 1)[-1]
    candidates.add(tail)
    return [c for c in candidates if c]


@dataclass
class IngestReport:
    file_id: int
    created: bool
    symbols: int
    imports: int
    new_promises: int
    resolved_promises: int
    parse_error: str = ""


async def ensure_project(session: AsyncSession, project_id: str, name: str = "") -> models.Project:
    project = await session.get(models.Project, project_id)
    if project is None:
        project = models.Project(
            id=project_id,
            name=name or project_id,
        )
        session.add(project)
        await session.flush()
    elif name and project.name != name:
        project.name = name
    return project


async def ingest_file(
    session: AsyncSession,
    *,
    project_id: str,
    path: str,
    content: str,
    language: str = "python",
) -> IngestReport:
    """Ingest or update a single file and update the promise graph.

    The operation is idempotent: re-ingesting the same content is a no-op
    beyond bumping ``last_ingested_at``. Re-ingesting changed content
    replaces the file's symbols and imports, then resolves any promises
    that are newly satisfied and creates any that are newly unsatisfied.
    """
    await ensure_project(session, project_id)

    norm_path = posixpath.normpath(path.replace("\\", "/"))
    content_hash = _hash(content)

    existing = (
        await session.execute(
            select(models.File).where(
                models.File.project_id == project_id,
                models.File.path == norm_path,
            )
        )
    ).scalar_one_or_none()

    created = existing is None
    if existing is None:
        existing = models.File(
            project_id=project_id,
            path=norm_path,
            language=language,
            content_hash=content_hash,
            content=content,
            lines=len(content.splitlines()),
        )
        session.add(existing)
        await session.flush()
    else:
        existing.language = language
        existing.content = content
        existing.content_hash = content_hash
        existing.lines = len(content.splitlines())
        existing.last_ingested_at = _utcnow()
        # wipe derived rows; we'll recompute below
        await session.execute(
            delete(models.Symbol).where(models.Symbol.file_id == existing.id)
        )
        await session.execute(
            delete(models.Import).where(models.Import.file_id == existing.id)
        )
        await session.flush()

    extraction = extract_python(content) if language == "python" else None

    report = IngestReport(
        file_id=existing.id,
        created=created,
        symbols=0,
        imports=0,
        new_promises=0,
        resolved_promises=0,
        parse_error=extraction.parse_error if extraction else "",
    )

    if extraction is not None:
        _persist_symbols(session, existing, extraction.symbols, project_id)
        _persist_imports(session, existing, extraction.imports, project_id, norm_path)
        report.symbols = len(extraction.symbols)
        report.imports = len(extraction.imports)
        await session.flush()

    resolved = await _resolve_promises_after_write(session, project_id, existing)
    new = await _create_promises_for_unresolved_imports(session, project_id, existing)
    report.resolved_promises = resolved
    report.new_promises = new

    return report


def _persist_symbols(
    session: AsyncSession,
    file: models.File,
    symbols: list[ExtractedSymbol],
    project_id: str,
) -> None:
    for s in symbols:
        session.add(
            models.Symbol(
                project_id=project_id,
                file_id=file.id,
                name=s.name,
                kind=s.kind,
                signature=s.signature,
                docstring=s.docstring,
                line_start=s.line_start,
                line_end=s.line_end,
            )
        )


def _persist_imports(
    session: AsyncSession,
    file: models.File,
    imports: list[ExtractedImport],
    project_id: str,
    source_path: str,
) -> None:
    for imp in imports:
        resolved_module = _resolve_relative_module(source_path, imp.module)
        session.add(
            models.Import(
                project_id=project_id,
                file_id=file.id,
                module=resolved_module,
                name=imp.name,
                alias=imp.alias,
                line=imp.line,
            )
        )


async def _resolve_promises_after_write(
    session: AsyncSession, project_id: str, file: models.File
) -> int:
    """Mark open promises satisfied by the newly-written file's symbols."""
    candidates = _module_candidates_for_file(file.path)
    if not candidates:
        return 0
    symbols = (
        await session.execute(
            select(models.Symbol).where(models.Symbol.file_id == file.id)
        )
    ).scalars().all()
    if not symbols:
        return 0

    resolved = 0
    for sym in symbols:
        open_promises = (
            await session.execute(
                select(models.Promise).where(
                    models.Promise.project_id == project_id,
                    models.Promise.resolved_at.is_(None),
                    models.Promise.expected_name == sym.name,
                    models.Promise.expected_module.in_(candidates),
                )
            )
        ).scalars().all()
        for pr in open_promises:
            pr.resolved_at = _utcnow()
            pr.resolver_symbol_id = sym.id
            resolved += 1
    return resolved


async def _create_promises_for_unresolved_imports(
    session: AsyncSession, project_id: str, file: models.File
) -> int:
    """For imports with no matching symbol in the project, open Promises.

    We deliberately only do this for *intra-project* imports (relative
    imports, or imports that match another file's candidate module names).
    External packages (``fastapi``, ``numpy``, ...) are not our problem.
    """
    imports = (
        await session.execute(
            select(models.Import).where(models.Import.file_id == file.id)
        )
    ).scalars().all()
    if not imports:
        return 0

    all_files = (
        await session.execute(
            select(models.File).where(models.File.project_id == project_id)
        )
    ).scalars().all()
    module_to_file: dict[str, models.File] = {}
    for f in all_files:
        for cand in _module_candidates_for_file(f.path):
            module_to_file[cand] = f

    new = 0
    for imp in imports:
        target_file = module_to_file.get(imp.module)
        if target_file is None:
            # Not a known intra-project module; skip (could be stdlib/3rd-party).
            continue
        # Match by symbol name in that file
        has_symbol = (
            await session.execute(
                select(models.Symbol.id).where(
                    models.Symbol.file_id == target_file.id,
                    models.Symbol.name == imp.name,
                )
            )
        ).first()
        if has_symbol:
            continue
        # Already an open promise for this exact (source_file, module, name)?
        existing_promise = (
            await session.execute(
                select(models.Promise).where(
                    models.Promise.project_id == project_id,
                    models.Promise.source_file_id == file.id,
                    models.Promise.expected_module == imp.module,
                    models.Promise.expected_name == imp.name,
                    models.Promise.resolved_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if existing_promise is not None:
            continue
        session.add(
            models.Promise(
                project_id=project_id,
                source_file_id=file.id,
                expected_module=imp.module,
                expected_name=imp.name,
            )
        )
        new += 1
    return new


async def project_graph(
    session: AsyncSession, project_id: str
) -> dict[str, list[dict]]:
    """Return a Cytoscape-compatible payload of the project graph."""
    files = (
        await session.execute(
            select(models.File).where(models.File.project_id == project_id)
        )
    ).scalars().all()
    file_ids = {f.id: f for f in files}

    symbols = (
        await session.execute(
            select(models.Symbol).where(models.Symbol.project_id == project_id)
        )
    ).scalars().all()
    imports = (
        await session.execute(
            select(models.Import).where(models.Import.project_id == project_id)
        )
    ).scalars().all()
    promises = (
        await session.execute(
            select(models.Promise).where(models.Promise.project_id == project_id)
        )
    ).scalars().all()

    nodes: list[dict] = []
    edges: list[dict] = []

    for f in files:
        nodes.append(
            {
                "data": {
                    "id": f"file:{f.id}",
                    "label": f.path,
                    "type": "file",
                    "lines": f.lines,
                }
            }
        )
    for s in symbols:
        nodes.append(
            {
                "data": {
                    "id": f"symbol:{s.id}",
                    "label": s.name,
                    "type": "symbol",
                    "kind": s.kind,
                    "signature": s.signature,
                    "file": file_ids[s.file_id].path if s.file_id in file_ids else "",
                }
            }
        )
        edges.append(
            {
                "data": {
                    "id": f"defines:{s.id}",
                    "source": f"file:{s.file_id}",
                    "target": f"symbol:{s.id}",
                    "type": "defines",
                }
            }
        )
    for imp in imports:
        # Link file → whichever symbol it imports, if resolvable right now.
        key = f"imp:{imp.id}"
        target = f"module:{imp.module}:{imp.name}"
        edges.append(
            {
                "data": {
                    "id": key,
                    "source": f"file:{imp.file_id}",
                    "target": target,
                    "type": "imports",
                    "label": f"{imp.module}::{imp.name}",
                }
            }
        )
    for pr in promises:
        if pr.resolved_at is not None and pr.resolver_symbol_id is not None:
            edges.append(
                {
                    "data": {
                        "id": f"promise:{pr.id}",
                        "source": f"file:{pr.source_file_id}",
                        "target": f"symbol:{pr.resolver_symbol_id}",
                        "type": "promise_resolved",
                        "label": f"{pr.expected_module}::{pr.expected_name}",
                    }
                }
            )
        else:
            target = f"open_promise:{pr.id}"
            nodes.append(
                {
                    "data": {
                        "id": target,
                        "label": f"{pr.expected_module}::{pr.expected_name}",
                        "type": "open_promise",
                    }
                }
            )
            edges.append(
                {
                    "data": {
                        "id": f"promise:{pr.id}",
                        "source": f"file:{pr.source_file_id}",
                        "target": target,
                        "type": "promise_open",
                        "label": f"{pr.expected_module}::{pr.expected_name}",
                    }
                }
            )

    return {"nodes": nodes, "edges": edges}

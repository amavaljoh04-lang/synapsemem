"""Read-only query routes: projects, graph, symbols, promises."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import embeddings, memory, models
from ..database import get_session
from ..schemas import (
    GraphPayload,
    ProjectSummary,
    PromiseSummary,
    SymbolSummary,
)

router = APIRouter(tags=["query"])


@router.get("/projects", response_model=list[ProjectSummary])
async def list_projects(
    session: AsyncSession = Depends(get_session),
) -> list[ProjectSummary]:
    rows = (await session.execute(select(models.Project))).scalars().all()
    return [ProjectSummary.model_validate(p) for p in rows]


@router.get("/projects/{project_id}", response_model=ProjectSummary)
async def get_project(
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> ProjectSummary:
    project = await session.get(models.Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return ProjectSummary.model_validate(project)


@router.get("/projects/{project_id}/graph", response_model=GraphPayload)
async def get_project_graph(
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> GraphPayload:
    project = await session.get(models.Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    payload = await memory.project_graph(session, project_id)
    return GraphPayload(**payload)


@router.get("/projects/{project_id}/symbols", response_model=list[SymbolSummary])
async def list_symbols(
    project_id: str,
    q: str = Query("", description="Case-insensitive substring filter on symbol name"),
    limit: int = Query(200, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
) -> list[SymbolSummary]:
    stmt = select(models.Symbol, models.File).join(
        models.File, models.Symbol.file_id == models.File.id
    ).where(models.Symbol.project_id == project_id)
    if q:
        stmt = stmt.where(models.Symbol.name.ilike(f"%{q}%"))
    stmt = stmt.order_by(models.Symbol.name).limit(limit)

    rows = (await session.execute(stmt)).all()
    out: list[SymbolSummary] = []
    for sym, file in rows:
        out.append(
            SymbolSummary(
                id=sym.id,
                name=sym.name,
                kind=sym.kind,
                signature=sym.signature,
                docstring=sym.docstring,
                file_path=file.path,
            )
        )
    return out


@router.post("/projects/{project_id}/reindex")
async def reindex_project(
    project_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Ensure every Symbol in this project has an up-to-date embedding.

    Idempotent: a second call right after the first returns
    ``{"embedded": 0}``. Silently swallows Ollama errors so a broken
    embedding endpoint never 500s the UI — the response carries what
    was actually computed.
    """
    project = await session.get(models.Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    n = await embeddings.ensure_symbols_embedded(session, project_id)
    return {"project_id": project_id, "embedded": n}


@router.get("/projects/{project_id}/promises", response_model=list[PromiseSummary])
async def list_promises(
    project_id: str,
    unresolved: bool = Query(
        default=False,
        description="If true, return only open (unresolved) promises.",
    ),
    limit: int = Query(500, ge=1, le=5000),
    session: AsyncSession = Depends(get_session),
) -> list[PromiseSummary]:
    stmt = select(models.Promise, models.File).join(
        models.File, models.Promise.source_file_id == models.File.id
    ).where(models.Promise.project_id == project_id)
    if unresolved:
        stmt = stmt.where(models.Promise.resolved_at.is_(None))
    stmt = stmt.order_by(models.Promise.created_at.desc()).limit(limit)

    rows = (await session.execute(stmt)).all()
    out: list[PromiseSummary] = []
    for pr, file in rows:
        out.append(
            PromiseSummary(
                id=pr.id,
                expected_module=pr.expected_module,
                expected_name=pr.expected_name,
                source_file_path=file.path,
                created_at=pr.created_at,
                resolved_at=pr.resolved_at,
            )
        )
    return out

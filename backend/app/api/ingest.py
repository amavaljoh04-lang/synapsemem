"""Ingestion routes — accept individual files or scan a directory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from .. import memory
from ..database import get_session
from ..schemas import (
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestReportResponse,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _to_response(report: memory.IngestReport, path: str) -> IngestReportResponse:
    return IngestReportResponse(
        file_id=report.file_id,
        path=path,
        created=report.created,
        symbols=report.symbols,
        imports=report.imports,
        new_promises=report.new_promises,
        resolved_promises=report.resolved_promises,
        parse_error=report.parse_error,
    )


@router.post("/file", response_model=IngestReportResponse)
async def ingest_file(
    req: IngestFileRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestReportResponse:
    report = await memory.ingest_file(
        session,
        project_id=req.project_id,
        path=req.path,
        content=req.content,
        language=req.language,
    )
    return _to_response(report, req.path)


@router.post("/directory", response_model=IngestDirectoryResponse)
async def ingest_directory(
    req: IngestDirectoryRequest,
    session: AsyncSession = Depends(get_session),
) -> IngestDirectoryResponse:
    root = Path(req.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {root}",
        )

    extensions = {"python": (".py",)}.get(req.language, ())
    if not extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {req.language}",
        )

    ingested: list[IngestReportResponse] = []
    skipped: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden / virtualenv / cache dirs.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in {"__pycache__", "node_modules", "venv", ".venv"}
        ]
        for fname in sorted(filenames):
            if not fname.endswith(extensions):
                continue
            fpath = Path(dirpath) / fname
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                skipped.append(f"{fpath}: {exc}")
                continue
            rel = str(fpath.resolve().relative_to(root))
            report = await memory.ingest_file(
                session,
                project_id=req.project_id,
                path=rel,
                content=content,
                language=req.language,
            )
            ingested.append(_to_response(report, rel))

    return IngestDirectoryResponse(
        project_id=req.project_id,
        ingested=ingested,
        skipped=skipped,
    )

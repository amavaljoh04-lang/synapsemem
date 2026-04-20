"""Ingestion routes — accept individual files or scan a directory."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from .. import memory
from ..database import get_session
from ..schemas import (
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestReportResponse,
)

_LANG_EXTS: dict[str, tuple[str, ...]] = {"python": (".py",)}
_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
_SKIP_DIR_NAMES = {
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    ".git",
}

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


@router.post("/archive", response_model=IngestDirectoryResponse)
async def ingest_archive(
    project_id: str = Form(...),
    language: str = Form("python"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> IngestDirectoryResponse:
    """Accept a ``.zip`` dropped from the UI and ingest its code files.

    Only *text* source files are kept (filtered by language extension),
    binaries/hidden/virtualenv dirs are skipped, and traversal is capped
    at ``_MAX_ARCHIVE_BYTES`` uncompressed to prevent zip bombs.
    """
    extensions = _LANG_EXTS.get(language, ())
    if not extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language: {language}",
        )

    raw = await file.read()
    if len(raw) > _MAX_ARCHIVE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Archive too big: {len(raw)} bytes",
        )
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not a valid zip archive: {exc}",
        ) from exc

    total_uncompressed = sum(i.file_size for i in zf.infolist())
    if total_uncompressed > _MAX_ARCHIVE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Uncompressed archive too big: {total_uncompressed} bytes "
                f"(limit {_MAX_ARCHIVE_BYTES})"
            ),
        )

    # Trim the common top-level folder GitHub/Zip exports usually add.
    names = [n for n in zf.namelist() if not n.endswith("/")]
    common_prefix = _common_top_dir(names)

    ingested: list[IngestReportResponse] = []
    skipped: list[str] = []
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        if info.is_dir():
            continue
        rel = _strip_prefix(info.filename, common_prefix)
        parts = PurePosixPath(rel).parts
        if not parts or any(
            p.startswith(".") or p in _SKIP_DIR_NAMES for p in parts[:-1]
        ):
            continue
        if not rel.endswith(extensions):
            continue
        try:
            content = zf.read(info).decode("utf-8", errors="replace")
        except (KeyError, RuntimeError) as exc:
            skipped.append(f"{rel}: {exc}")
            continue
        report = await memory.ingest_file(
            session,
            project_id=project_id,
            path=rel,
            content=content,
            language=language,
        )
        ingested.append(_to_response(report, rel))

    return IngestDirectoryResponse(
        project_id=project_id,
        ingested=ingested,
        skipped=skipped,
    )


def _common_top_dir(names: list[str]) -> str:
    if not names:
        return ""
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    if len(tops) == 1 and not any("/" not in n for n in names):
        return next(iter(tops)) + "/"
    return ""


def _strip_prefix(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    return name

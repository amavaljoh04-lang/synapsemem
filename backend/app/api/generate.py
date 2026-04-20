"""Project-generation endpoints.

* ``POST /projects/{project_id}/generate`` — streams NDJSON events while
  SynapseMem iteratively prompts the LLM, ingests files, runs checks,
  and loops using the memory feedback.
* ``GET  /projects/{project_id}/download`` — returns the latest zip of
  the generated project's files.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_session
from ..generator import (
    GenerationOptions,
    make_project_zip,
    project_workdir,
    stream_generation,
)

router = APIRouter(prefix="/projects", tags=["generate"])


class GenerateRequest(BaseModel):
    goal: str = Field(min_length=4, max_length=4000)
    model: str | None = Field(
        default=None,
        description="Override the default coder model from settings.",
    )
    max_iterations: int = Field(default=4, ge=1, le=10)


@router.post("/{project_id}/generate")
async def generate(
    project_id: str,
    req: GenerateRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    options = GenerationOptions(
        goal=req.goal,
        model=req.model or settings.extractor_model,
        max_iterations=req.max_iterations,
    )
    return StreamingResponse(
        stream_generation(session, project_id, options),
        media_type="application/x-ndjson",
    )


@router.get("/{project_id}/download")
async def download(project_id: str) -> FileResponse:
    workdir = project_workdir(project_id)
    if not any(workdir.iterdir()):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No generated files for this project yet.",
        )
    zip_path = make_project_zip(project_id)
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{project_id}.zip",
    )

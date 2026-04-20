"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IngestFileRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=1024)
    content: str
    language: str = "python"


class IngestDirectoryRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, description="Absolute path to the source directory")
    language: str = "python"


class IngestReportResponse(BaseModel):
    file_id: int
    path: str
    created: bool
    symbols: int
    imports: int
    new_promises: int
    resolved_promises: int
    parse_error: str = ""


class IngestDirectoryResponse(BaseModel):
    project_id: str
    ingested: list[IngestReportResponse]
    skipped: list[str]


class ProjectSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime


class SymbolSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    kind: str
    signature: str
    docstring: str = ""
    file_path: str = ""


class PromiseSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    expected_module: str
    expected_name: str
    source_file_path: str = ""
    created_at: datetime
    resolved_at: datetime | None = None


class GraphPayload(BaseModel):
    nodes: list[dict]
    edges: list[dict]

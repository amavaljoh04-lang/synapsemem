"""Chat endpoint — memory-augmented single-turn completion.

Flow per request:

1. Persist the user message as an ``Episode`` (role=``user``).
2. Build a memory context block from the project's current state.
3. Call Ollama with ``[system, context, user]``.
4. Persist the assistant reply as another ``Episode`` (role=``assistant``).
5. Return the reply together with a digest of what was retrieved so the
   UI can display "citations" (which promises/symbols/episodes informed
   the answer).

Streaming variant is ``POST /chat/stream`` — same input shape, returns
a newline-delimited stream so the frontend can render progressively.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models
from ..config import settings
from ..context_builder import build_chat_context
from ..database import get_session
from ..ollama_client import OllamaClient, OllamaError

router = APIRouter(prefix="/chat", tags=["chat"])


SYSTEM_PROMPT = (
    "You are SynapseMem, a coding agent with persistent project memory. "
    "Before writing code, read the 'Open promises' section — those are "
    "symbols earlier files already reference and that YOU owe to this "
    "project. When you produce code, keep names, signatures, and import "
    "paths consistent with the 'Relevant symbols' section: never rename "
    "an existing public symbol unless the user explicitly asks for it. "
    "When you create a new symbol that another file references, say so "
    "explicitly so the promise tracker can resolve it. Answer in the same "
    "language as the user."
)


class ChatRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    model: str | None = Field(
        default=None,
        description="Override the extractor_model from settings (optional).",
    )


class ChatResponse(BaseModel):
    project_id: str
    reply: str
    model: str
    context_chars: int
    context: str = Field(
        default="",
        description=(
            "The assembled memory block that was prepended to the user "
            "message. Returned for observability/UI citation. May be empty "
            "for projects that have no memory yet."
        ),
    )


async def _record_episode(
    session: AsyncSession,
    *,
    project_id: str,
    role: str,
    kind: str,
    content: str,
) -> None:
    await _ensure_project(session, project_id)
    session.add(
        models.Episode(
            project_id=project_id,
            role=role,
            kind=kind,
            content=content,
        )
    )
    await session.flush()


async def _ensure_project(session: AsyncSession, project_id: str) -> None:
    existing = await session.get(models.Project, project_id)
    if existing is None:
        session.add(
            models.Project(id=project_id, name=project_id, description="")
        )
        await session.flush()


def _build_messages(context: str, user_message: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        msgs.append(
            {
                "role": "system",
                "content": "Project memory:\n\n" + context,
            }
        )
    msgs.append({"role": "user", "content": user_message})
    return msgs


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    await _record_episode(
        session,
        project_id=req.project_id,
        role="user",
        kind="chat",
        content=req.message,
    )
    context = await build_chat_context(session, req.project_id, req.message)
    messages = _build_messages(context, req.message)

    model = req.model or settings.extractor_model
    client = OllamaClient(settings.ollama_base_url)
    try:
        response = await client.chat(model=model, messages=messages)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    reply = (response.get("message") or {}).get("content", "")
    await _record_episode(
        session,
        project_id=req.project_id,
        role="assistant",
        kind="chat",
        content=reply,
    )

    return ChatResponse(
        project_id=req.project_id,
        reply=reply,
        model=model,
        context_chars=len(context),
        context=context,
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    await _record_episode(
        session,
        project_id=req.project_id,
        role="user",
        kind="chat",
        content=req.message,
    )
    context = await build_chat_context(session, req.project_id, req.message)
    messages = _build_messages(context, req.message)
    await session.commit()  # commit user episode before the long stream

    model = req.model or settings.extractor_model
    client = OllamaClient(settings.ollama_base_url)

    async def _iterate():
        preface = {
            "type": "context",
            "project_id": req.project_id,
            "model": model,
            "context_chars": len(context),
            "context": context,
        }
        yield json.dumps(preface, ensure_ascii=False) + "\n"
        buffer: list[str] = []
        try:
            async for piece in client.chat_stream(model=model, messages=messages):
                buffer.append(piece)
                yield json.dumps({"type": "token", "text": piece}) + "\n"
        except OllamaError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            return

        # persist assistant episode once streaming is done
        from ..database import session_scope

        full = "".join(buffer)
        async with session_scope() as s2:
            await _record_episode(
                s2,
                project_id=req.project_id,
                role="assistant",
                kind="chat",
                content=full,
            )
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(_iterate(), media_type="application/x-ndjson")


@router.get("/models", tags=["chat"])
async def list_models() -> dict:
    """Proxy to Ollama /api/tags so the UI can populate a model picker."""
    client = OllamaClient(settings.ollama_base_url)
    try:
        models_list = await client.tags()
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "default": settings.extractor_model,
        "models": [m.get("name", "") for m in models_list if m.get("name")],
    }

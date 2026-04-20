"""Embedding layer for SynapseMem.

We keep this deliberately minimal for v0.1:

* The model is whatever ``settings.embeddings_model`` names, served by
  the same Ollama instance that serves the chat model.
* One vector per :class:`models.Symbol`, stored as raw float32 bytes in
  :class:`models.SymbolEmbedding`.
* No FAISS yet — projects fit in memory by several orders of magnitude
  at this scale, so a plain NumPy cosine similarity is both simpler and
  faster than the overhead of maintaining an index.

Two entry points:

* :func:`ensure_symbols_embedded` — idempotent backfill for a given
  project. Called before each semantic retrieval so symbols added since
  the last query always participate.
* :func:`semantic_symbol_search` — cosine top-K against the stored
  vectors for a free-form query.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import models
from .config import settings
from .ollama_client import OllamaClient, OllamaError

# How many symbol texts we embed per Ollama call. ``nomic-embed-text``
# handles a few hundred short inputs per second; we call sequentially
# because the official endpoint is one text per call.
_BACKFILL_BATCH = 32


def _symbol_text(sym: models.Symbol, path: str) -> str:
    """Textual representation we feed into the embedding model."""
    parts = [f"{sym.kind} {sym.name}", f"in {path}"]
    if sym.signature:
        parts.append(sym.signature)
    if sym.docstring:
        parts.append(sym.docstring.strip())
    return "\n".join(parts)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _pack(vec: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    import struct

    return list(struct.unpack(f"<{dim}f", blob))


async def ensure_symbols_embedded(
    session: AsyncSession,
    project_id: str,
    *,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> int:
    """Embed any Symbol of ``project_id`` whose vector is stale/missing.

    Returns the number of embeddings computed. Silently skips when the
    Ollama endpoint is unreachable so UI/chat paths never break because
    of an embedding failure.
    """
    model = model or settings.embeddings_model
    close_client = False
    if client is None:
        client = OllamaClient(str(settings.ollama_base_url))
        close_client = True

    rows = (
        await session.execute(
            select(models.Symbol, models.File.path)
            .join(models.File, models.Symbol.file_id == models.File.id)
            .where(models.Symbol.project_id == project_id)
        )
    ).all()
    if not rows:
        return 0

    # Fetch existing embeddings for this project in one query so we can
    # decide per-symbol whether to re-embed.
    existing_rows = (
        await session.execute(
            select(models.SymbolEmbedding).where(
                models.SymbolEmbedding.project_id == project_id
            )
        )
    ).scalars().all()
    existing = {e.symbol_id: e for e in existing_rows}

    computed = 0
    try:
        for sym, path in rows:
            text = _symbol_text(sym, path)
            digest = _content_hash(text)
            prev = existing.get(sym.id)
            if prev and prev.model == model and prev.content_hash == digest:
                continue  # still valid
            try:
                vec = await client.embed(model=model, text=text)
            except OllamaError:
                # Abort the whole backfill — the caller will get what we
                # managed to embed so far and can retry later.
                break
            if not vec:
                continue
            blob = _pack(vec)
            if prev is None:
                session.add(
                    models.SymbolEmbedding(
                        symbol_id=sym.id,
                        project_id=project_id,
                        model=model,
                        dim=len(vec),
                        vector=blob,
                        content_hash=digest,
                    )
                )
            else:
                prev.model = model
                prev.dim = len(vec)
                prev.vector = blob
                prev.content_hash = digest
            computed += 1
            if computed % _BACKFILL_BATCH == 0:
                await session.flush()
    finally:
        if close_client:
            await client.aclose()
    if computed:
        await session.commit()
    return computed


@dataclass
class SemanticHit:
    symbol: models.Symbol
    path: str
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


async def semantic_symbol_search(
    session: AsyncSession,
    project_id: str,
    query: str,
    *,
    k: int = 10,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> list[SemanticHit]:
    """Return top-K symbols most similar to ``query``.

    Returns an empty list when embeddings are not yet available or the
    Ollama endpoint is unreachable — callers should combine this with
    the keyword-based retrieval rather than rely on semantic alone.
    """
    model = model or settings.embeddings_model
    close_client = False
    if client is None:
        client = OllamaClient(str(settings.ollama_base_url))
        close_client = True
    try:
        try:
            q_vec = await client.embed(model=model, text=query)
        except OllamaError:
            return []
    finally:
        if close_client and client is not None:
            # leave open if we actually need it for more calls below
            pass
    if not q_vec:
        if close_client:
            await client.aclose()
        return []

    rows = (
        await session.execute(
            select(models.SymbolEmbedding, models.Symbol, models.File.path)
            .join(models.Symbol, models.Symbol.id == models.SymbolEmbedding.symbol_id)
            .join(models.File, models.Symbol.file_id == models.File.id)
            .where(models.SymbolEmbedding.project_id == project_id)
        )
    ).all()
    if close_client:
        await client.aclose()

    scored: list[SemanticHit] = []
    for emb, sym, path in rows:
        vec = _unpack(emb.vector, emb.dim)
        score = _cosine(q_vec, vec)
        if score <= 0:
            continue
        scored.append(SemanticHit(symbol=sym, path=path, score=score))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k]

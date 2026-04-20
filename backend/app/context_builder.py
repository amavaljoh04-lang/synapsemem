"""Assemble a memory-backed context block for the chat LLM.

The context is NOT a generic RAG dump — it is four targeted sections the
coder-memory actually cares about, each cheap to build from SQL:

1. **Project summary** (name / description / file + symbol counts).
2. **Open promises** — unresolved references. The LLM must pay these back
   before closing the session.
3. **Recent symbols** — signatures + docstrings of symbols in files the
   user mentions (naive keyword match for v0.1; embedding-based retrieval
   lands when we wire the embeddings model).
4. **Recent episodes** — last N user prompts + agent answers, so the
   model has the conversational thread without re-sending it each turn.

All four sections are clearly labelled so we can later measure
per-section contribution to benchmark uplift.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import embeddings, models


async def build_chat_context(
    session: AsyncSession,
    project_id: str,
    user_message: str,
    *,
    recent_episodes: int = 10,
    max_symbols: int = 20,
) -> str:
    """Return a ready-to-insert system-prompt block summarising memory."""
    sections: list[str] = []

    # --- 1. Project summary ----------------------------------------------
    project = await session.get(models.Project, project_id)
    if project is None:
        return ""  # no memory yet; chat falls back to raw prompt
    file_count = (
        await session.execute(
            select(func.count(models.File.id)).where(
                models.File.project_id == project_id
            )
        )
    ).scalar_one()
    symbol_count = (
        await session.execute(
            select(func.count(models.Symbol.id)).where(
                models.Symbol.project_id == project_id
            )
        )
    ).scalar_one()
    sections.append(
        "## Project\n"
        f"- id: {project.id}\n"
        f"- name: {project.name}\n"
        f"- files: {file_count}\n"
        f"- symbols: {symbol_count}"
    )

    # --- 2. Open promises -------------------------------------------------
    open_promises = (
        await session.execute(
            select(models.Promise, models.File.path)
            .join(models.File, models.Promise.source_file_id == models.File.id)
            .where(
                models.Promise.project_id == project_id,
                models.Promise.resolved_at.is_(None),
            )
            .order_by(models.Promise.created_at.desc())
            .limit(50)
        )
    ).all()
    if open_promises:
        lines = ["## Open promises (unresolved imports — you OWE these)"]
        for promise, path in open_promises:
            lines.append(
                f"- `{promise.expected_name}` expected in module "
                f"`{promise.expected_module}` (imported by `{path}`)"
            )
        sections.append("\n".join(lines))
    else:
        sections.append("## Open promises\n(none — every import resolves.)")

    # --- 3. Relevant symbols ---------------------------------------------
    # First: semantic retrieval if the embeddings layer is ready. Silently
    # skipped when Ollama is unreachable; keyword match fills the rest.
    symbols: list[tuple[models.Symbol, str]] = []
    seen_ids: set[int] = set()
    try:
        hits = await embeddings.semantic_symbol_search(
            session, project_id, user_message, k=max_symbols
        )
    except Exception:
        hits = []
    for hit in hits:
        if hit.symbol.id in seen_ids:
            continue
        seen_ids.add(hit.symbol.id)
        symbols.append((hit.symbol, hit.path))
        if len(symbols) >= max_symbols:
            break

    keywords = _extract_keywords(user_message)
    for kw in keywords:
        if len(symbols) >= max_symbols:
            break
        rows = (
            await session.execute(
                select(models.Symbol, models.File.path)
                .join(models.File, models.Symbol.file_id == models.File.id)
                .where(
                    models.Symbol.project_id == project_id,
                    models.Symbol.name.ilike(f"%{kw}%"),
                )
                .limit(max_symbols)
            )
        ).all()
        for sym, path in rows:
            if sym.id in seen_ids:
                continue
            seen_ids.add(sym.id)
            symbols.append((sym, path))
            if len(symbols) >= max_symbols:
                break
    if symbols:
        lines = ["## Relevant symbols already defined in this project"]
        for sym, path in symbols:
            header = f"- `{sym.kind} {sym.name}` in `{path}`"
            if sym.signature:
                header += f" — `{sym.signature}`"
            if sym.docstring:
                header += f"\n    {_one_line(sym.docstring)}"
            lines.append(header)
        sections.append("\n".join(lines))

    # --- 4. Recent episodes ----------------------------------------------
    episodes = (
        await session.execute(
            select(models.Episode)
            .where(models.Episode.project_id == project_id)
            .order_by(models.Episode.created_at.desc())
            .limit(recent_episodes)
        )
    ).scalars().all()
    if episodes:
        lines = ["## Recent conversation"]
        for ep in reversed(episodes):
            snippet = _one_line(ep.content, width=300)
            lines.append(f"- [{ep.role}] {snippet}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "what", "when", "where",
        "how", "does", "into", "from", "your", "would", "should", "could",
        "about", "there", "have", "has", "had", "was", "were", "been", "being",
        "are", "is", "am", "a", "an", "of", "to", "in", "on", "by", "it", "be",
        "as", "or", "if", "so", "do", "i", "you", "we", "they", "me", "my",
        "le", "la", "les", "de", "du", "des", "un", "une", "je", "tu", "il",
        "elle", "nous", "vous", "ils", "elles", "que", "qui", "quoi", "pour",
        "avec", "sans", "sur", "sous", "dans", "et", "ou", "mais", "donc",
        "car", "ni", "est", "sont", "avoir", "être", "faire", "ça", "ce",
        "cet", "cette", "ces",
    }
)


def _extract_keywords(text: str) -> list[str]:
    """Rough keyword extraction — used until we wire proper embeddings."""
    out: list[str] = []
    seen: set[str] = set()
    for token in text.split():
        token = token.strip(" \t\n`'\",.!?;:()[]{}<>").lower()
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 10:
            break
    return out


def _one_line(text: str, width: int = 120) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    if len(line) > width:
        return line[: width - 1] + "…"
    return line

"""Three retrievers: baseline (none), BM25, SynapseMem.

Each returns a text block to prepend to the completion prompt. Keeping
the interface identical makes the runner simple and makes it trivial to
plug a fourth approach later (e.g. dense retrieval, GraphRAG).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol


class Retriever(Protocol):
    name: str

    def build(self, repo_files: dict[str, str]) -> None:
        """Index the repository. Called once per task (or per repo)."""

    def retrieve(self, query: str, *, target_path: str, max_chars: int) -> str:
        """Return the cross-file context block for the given query."""


@dataclass
class NoopRetriever:
    """Baseline: no cross-file context at all."""

    name: str = "baseline"

    def build(self, repo_files: dict[str, str]) -> None:
        return None

    def retrieve(self, query: str, *, target_path: str, max_chars: int) -> str:
        return ""


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class BM25Retriever:
    """Vanilla BM25 over file chunks. The classic RAG baseline."""

    name: str = "rag_bm25"
    chunk_lines: int = 40
    k1: float = 1.5
    b: float = 0.75
    _chunks: list[tuple[str, str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0
    _n_docs: int = 0
    _doc_tokens: list[list[str]] = field(default_factory=list)

    def build(self, repo_files: dict[str, str]) -> None:
        self._chunks = []
        self._doc_tokens = []
        self._df = Counter()
        for path, content in repo_files.items():
            lines = content.splitlines()
            for start in range(0, len(lines), self.chunk_lines):
                chunk = "\n".join(lines[start : start + self.chunk_lines])
                if not chunk.strip():
                    continue
                self._chunks.append((path, chunk))
                toks = _tokenize(chunk)
                self._doc_tokens.append(toks)
                for t in set(toks):
                    self._df[t] += 1
        self._n_docs = len(self._chunks)
        self._avgdl = (
            sum(len(t) for t in self._doc_tokens) / self._n_docs
            if self._n_docs
            else 0.0
        )

    def retrieve(self, query: str, *, target_path: str, max_chars: int) -> str:
        if not self._chunks:
            return ""
        q_tokens = _tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tokens in enumerate(self._doc_tokens):
            path, _ = self._chunks[i]
            if path == target_path:
                continue  # must NOT leak from the file we're predicting into
            scored.append((self._score(q_tokens, tokens), i))
        scored.sort(reverse=True)
        out: list[str] = []
        used = 0
        for _, i in scored:
            if used >= max_chars:
                break
            path, chunk = self._chunks[i]
            block = f"# file: {path}\n{chunk}\n"
            if used + len(block) > max_chars:
                break
            out.append(block)
            used += len(block)
        return "".join(out)

    def _score(self, q: Iterable[str], tokens: list[str]) -> float:
        tf = Counter(tokens)
        dl = len(tokens) or 1
        score = 0.0
        for term in q:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return score


@dataclass
class SynapseMemRetriever:
    """Ingest the repo into SynapseMem and query its chat context builder.

    Notes
    -----
    We instantiate a fresh in-memory SQLite DB per task so there is no
    leakage across tasks. The retriever imports the backend modules
    lazily so this file remains safe to import in environments without
    the backend deps (e.g. during frontend-only tooling).
    """

    name: str = "synapsemem"
    max_symbols: int = 25

    def __init__(self, project_id: str = "ccebench") -> None:
        self.project_id = project_id
        self._repo_files: dict[str, str] = {}

    def build(self, repo_files: dict[str, str]) -> None:
        self._repo_files = dict(repo_files)

    def retrieve(self, query: str, *, target_path: str, max_chars: int) -> str:
        import asyncio

        return asyncio.run(
            self._retrieve_async(query, target_path=target_path, max_chars=max_chars)
        )

    async def _retrieve_async(
        self, query: str, *, target_path: str, max_chars: int
    ) -> str:
        # Lazy imports so this module is safe to import without the backend.
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from app import memory, models  # noqa: F401  (register tables)
        from app.context_builder import build_chat_context
        from app.database import Base

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:", future=True
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine, expire_on_commit=False
        )

        async with maker() as session:
            for path, content in self._repo_files.items():
                if path == target_path:
                    continue  # never ingest the target file (would leak)
                await memory.ingest_file(
                    session,
                    project_id=self.project_id,
                    path=path,
                    content=content,
                )
            await session.commit()
            block = await build_chat_context(
                session,
                self.project_id,
                query,
                max_symbols=self.max_symbols,
            )
        await engine.dispose()
        if len(block) > max_chars:
            return block[:max_chars]
        return block

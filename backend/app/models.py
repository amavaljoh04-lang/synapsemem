"""ORM models — episodic layer, semantic graph, procedural rules.

Design notes
------------
Four cognitively-inspired layers:

* **Episode**: ground-truth stream of every interaction (user prompt, file
  write, agent note). Immutable, cheap, append-only. Retained verbatim so
  we can always recover detail if the derived layers get something wrong.

* **File / Symbol / Import / Promise**: the semantic layer. This is
  explicitly NOT a free-form triplet graph — we start from the structure
  that actually matters for code (who defines what, who imports what, who
  owes what). A free-form triplet extension lives in ``Edge`` for
  the NL extractor to write into later.

* **Preference**: procedural layer. Rules like ``(Johnny, uses-cli-lib,
  argparse, confidence=0.9)``. The reflection worker promotes strong
  recurring patterns from the semantic layer into procedural rules.

Promises are the keystone. When file A imports ``parse_config`` from
``utils`` and ``utils`` does not yet define ``parse_config``, we create a
Promise row. When ``utils`` is later written and defines ``parse_config``,
we mark the Promise resolved and link it to the resolver symbol. The
whole point of SynapseMem is that no Promise is ever forgotten.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Project(Base):
    """A logical collection of files (one per repo or per coding session)."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    files: Mapped[list[File]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Episode(Base):
    """Raw, append-only interaction log — the episodic ground truth."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class File(Base):
    """A source file within a project."""

    __tablename__ = "files"
    __table_args__ = (UniqueConstraint("project_id", "path", name="uq_file_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(1024))
    language: Mapped[str] = mapped_column(String(32), default="python")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    lines: Mapped[int] = mapped_column(Integer, default=0)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped[Project] = relationship(back_populates="files")
    symbols: Mapped[list[Symbol]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    imports: Mapped[list[Import]] = relationship(
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Symbol(Base):
    """A top-level named definition extracted from a file.

    ``kind`` is one of ``class``, ``function``, ``async_function``, ``variable``.
    ``signature`` is a compact one-line representation (e.g. ``def foo(x: int) -> str``).
    """

    __tablename__ = "symbols"
    __table_args__ = (
        Index("ix_symbols_project_name", "project_id", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(256))
    kind: Mapped[str] = mapped_column(String(32))
    signature: Mapped[str] = mapped_column(String(1024), default="")
    docstring: Mapped[str] = mapped_column(Text, default="")
    line_start: Mapped[int] = mapped_column(Integer, default=0)
    line_end: Mapped[int] = mapped_column(Integer, default=0)

    file: Mapped[File] = relationship(back_populates="symbols")


class Import(Base):
    """An import statement in a file.

    A single ``from x import a, b`` becomes one row with ``names=["a", "b"]``
    encoded as JSON-ish text. We keep the raw module string and the list of
    names separately so promise matching is straightforward.
    """

    __tablename__ = "imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE")
    )
    module: Mapped[str] = mapped_column(String(512))
    name: Mapped[str] = mapped_column(String(256))
    alias: Mapped[str] = mapped_column(String(256), default="")
    line: Mapped[int] = mapped_column(Integer, default=0)

    file: Mapped[File] = relationship(back_populates="imports")


class Promise(Base):
    """A reference not yet satisfied anywhere in the project.

    A Promise is created when an Import targets a symbol (module + name) that
    is not yet defined in any file of the project. It's resolved (and stays
    in history) when a Symbol matching the expected (module, name) appears.
    """

    __tablename__ = "promises"
    __table_args__ = (
        Index(
            "ix_promises_lookup",
            "project_id",
            "expected_module",
            "expected_name",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE")
    )
    expected_module: Mapped[str] = mapped_column(String(512))
    expected_name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolver_symbol_id: Mapped[int | None] = mapped_column(
        ForeignKey("symbols.id", ondelete="SET NULL"), nullable=True
    )


class Edge(Base):
    """Generic graph edge — used by the NL extractor in the next milestone.

    The semantic layer (files/symbols/imports/promises) covers the structured
    part. For free-form, LLM-extracted triplets from natural-language prompts,
    we use this table. It's intentionally flat to keep ingestion fast.
    """

    __tablename__ = "edges"
    __table_args__ = (
        Index("ix_edges_lookup", "project_id", "source", "relation", "target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(String(512))
    relation: Mapped[str] = mapped_column(String(128))
    target: Mapped[str] = mapped_column(String(512))
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    source_episode_id: Mapped[int | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"), nullable=True
    )


class Preference(Base):
    """Procedural-layer rule. Promoted by the reflection worker."""

    __tablename__ = "preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(256))
    value: Mapped[str] = mapped_column(String(1024))
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class SymbolEmbedding(Base):
    """Cached embedding vector for a Symbol.

    Stored as a raw little-endian float32 blob: ``numpy.asarray(vec,
    dtype=np.float32).tobytes()``. We don't use SQLAlchemy's JSON type
    because at query time we want to rebuild a matrix quickly for
    cosine similarity without parsing JSON for each row.
    """

    __tablename__ = "symbol_embeddings"

    symbol_id: Mapped[int] = mapped_column(
        ForeignKey("symbols.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[str] = mapped_column(String(128), index=True)
    model: Mapped[str] = mapped_column(String(128))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

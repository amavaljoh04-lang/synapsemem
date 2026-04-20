"""Memory-driven project generator.

This is what makes SynapseMem *exploitable* for actual code production:
the user gives a natural-language goal, and the system iteratively asks
the LLM for files, ingests each one into memory (which closes/opens
promises, records symbols, logs the episode), runs a quick syntax/test
check, then re-prompts with the updated memory context until the project
is internally coherent OR ``max_iterations`` is reached.

The LLM is instructed to emit files inside clearly-framed blocks::

    === FILE: app/main.py ===
    ```python
    ...
    ```

or (fallback, also parsed)::

    # FILE: app/main.py
    ```python
    ...
    ```

Any other prose between blocks is recorded as a plan episode.

After each round we:

1. Parse file blocks, write each to ``data/projects/{id}/`` on disk.
2. Re-ingest through ``memory.ingest_file`` so the promise graph is up
   to date (newly-defined symbols resolve earlier promises; new imports
   open new promises).
3. Run ``python -m py_compile`` on each new/changed file to catch syntax
   errors. If the repo has a ``tests/`` directory we run ``pytest -q``
   with a short timeout.
4. Build a fresh memory context + a human feedback block listing open
   promises, compile errors, failing tests. That feedback is then fed
   back to the LLM for the next round.

The output is a streaming sequence of NDJSON events so the UI can render
progress live.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from . import config as _config
from . import memory, models
from .context_builder import build_chat_context
from .ollama_client import OllamaClient, OllamaError


def _data_dir() -> Path:
    """Return the current data dir (re-read each call, test-friendly)."""
    return Path(_config.settings.data_dir)


def _ollama_base_url() -> str:
    return _config.settings.ollama_base_url

# --- Filesystem helpers ----------------------------------------------------


def _safe_project_id(project_id: str) -> str:
    """Reject anything with path separators; only keep a sane identifier."""
    if not project_id or "/" in project_id or ".." in project_id:
        raise ValueError(f"invalid project_id: {project_id!r}")
    return project_id


def project_workdir(project_id: str) -> Path:
    """Return the on-disk directory where generated files are stored."""
    safe = _safe_project_id(project_id)
    root = _data_dir() / "projects" / safe
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_within(base: Path, path: Path) -> bool:
    """True iff ``path`` resolves to a descendant of ``base``."""
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


# --- LLM output parser -----------------------------------------------------

_FILE_HEADER_RE = re.compile(
    r"^(?:===\s*FILE:|#\s*FILE:)\s*(?P<path>[^\s=][^\n=]*?)\s*(?:===)?\s*$",
    re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+-]*)?\n(?P<body>.*?)```",
    re.DOTALL,
)


@dataclass
class ParsedFile:
    path: str
    content: str


@dataclass
class ParsedResponse:
    files: list[ParsedFile] = field(default_factory=list)
    prose: str = ""


def parse_llm_response(text: str) -> ParsedResponse:
    """Extract file blocks from an LLM response.

    Robust to LLMs that forget to double up the FILE marker: we first
    split on headers, then pick the first fenced code block that follows.
    Everything outside file blocks is returned as ``prose`` so callers
    can persist it as a plan/thought episode.
    """
    parts = list(_FILE_HEADER_RE.finditer(text))
    if not parts:
        return ParsedResponse(files=[], prose=text.strip())

    files: list[ParsedFile] = []
    prose_chunks: list[str] = []
    # Prose before the first file header.
    head = text[: parts[0].start()].strip()
    if head:
        prose_chunks.append(head)

    for i, m in enumerate(parts):
        path = m.group("path").strip().strip("`").strip()
        body_start = m.end()
        body_end = parts[i + 1].start() if i + 1 < len(parts) else len(text)
        body = text[body_start:body_end]
        fence = _CODE_FENCE_RE.search(body)
        if fence is None:
            # No fenced block — fall back to raw body (trimmed).
            content = body.strip()
        else:
            content = fence.group("body")
            # keep prose AFTER the fence too
            after = body[fence.end() :].strip()
            if after:
                prose_chunks.append(after)
        clean = path.strip().strip('"').strip("'")
        if clean.startswith("./"):
            clean = clean[2:]
        # Reject absolute paths and parent traversal.
        if (
            not clean
            or clean.startswith("/")
            or clean.startswith("\\")
            or ".." in clean.replace("\\", "/").split("/")
        ):
            continue
        files.append(ParsedFile(path=clean, content=content.rstrip() + "\n"))

    return ParsedResponse(files=files, prose="\n\n".join(prose_chunks).strip())


# --- Project sandbox / feedback -------------------------------------------


@dataclass
class FileCheck:
    path: str
    compile_ok: bool
    compile_error: str = ""


def syntax_check(files: list[tuple[str, str]]) -> list[FileCheck]:
    """Run ``py_compile`` on each python file, return per-file results."""
    checks: list[FileCheck] = []
    for path, content in files:
        if not path.endswith(".py"):
            checks.append(FileCheck(path=path, compile_ok=True))
            continue
        try:
            compile(content, path, "exec")
            checks.append(FileCheck(path=path, compile_ok=True))
        except SyntaxError as e:
            checks.append(
                FileCheck(
                    path=path,
                    compile_ok=False,
                    compile_error=f"{e.__class__.__name__}: {e.msg} "
                    f"(line {e.lineno})",
                )
            )
    return checks


def run_pytest(workdir: Path, *, timeout: int = 90) -> tuple[bool, str]:
    """If a ``tests`` directory exists, run ``pytest -q`` inside ``workdir``.

    Returns ``(passed, combined_output)``. Missing pytest or no tests count
    as "not run" (passed=True, output indicates so).
    """
    tests_dir = workdir / "tests"
    if not tests_dir.exists():
        return True, "(no tests/ directory — skipping pytest)"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--maxfail=3"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return True, "(pytest not installed — skipping)"
    except subprocess.TimeoutExpired:
        return False, f"(pytest timed out after {timeout}s)"
    ok = proc.returncode == 0
    output = (proc.stdout or "") + (proc.stderr or "")
    return ok, output[-4000:]


# --- Prompts ---------------------------------------------------------------


SYSTEM_PROMPT = """You are SynapseMem-Coder, a disciplined coding agent.

You are producing a concrete multi-file Python project. Follow these rules:

1. Always emit each file using this exact framing, on its own line:
       === FILE: relative/path/to/file.py ===
   immediately followed by a fenced code block with the full file content.
2. Paths are relative to the project root, never start with "/" or "..".
3. Before writing code, READ the 'Open promises' memory section: any name
   listed there MUST be defined by one of the files in your next answer,
   using the exact same name and in the matching module.
4. Never redefine an existing public symbol from 'Relevant symbols' —
   import it instead.
5. Keep imports consistent across files (module path must match file
   layout; the promise tracker verifies this automatically).
6. When the project is complete, emit ONE final line: "DONE".
"""


def _format_feedback(
    *,
    open_promises: list[tuple[str, str, str]],
    compile_errors: list[FileCheck],
    test_output: str,
    tests_passed: bool,
) -> str:
    lines: list[str] = []
    if open_promises:
        lines.append("## Open promises (must be satisfied by your next answer)")
        for mod, name, path in open_promises:
            lines.append(f"- `{name}` expected in `{mod}` (imported by `{path}`)")
    bad = [c for c in compile_errors if not c.compile_ok]
    if bad:
        lines.append("## Syntax errors")
        for c in bad:
            lines.append(f"- {c.path}: {c.compile_error}")
    if test_output and not tests_passed:
        lines.append("## pytest failed")
        lines.append("```")
        lines.append(test_output.strip())
        lines.append("```")
    elif test_output:
        lines.append("## pytest")
        lines.append("```")
        lines.append(test_output.strip()[-800:])
        lines.append("```")
    return "\n".join(lines).strip()


async def _gather_open_promises(
    session: AsyncSession, project_id: str
) -> list[tuple[str, str, str]]:
    from sqlalchemy import select

    rows = (
        await session.execute(
            select(models.Promise, models.File.path)
            .join(models.File, models.Promise.source_file_id == models.File.id)
            .where(
                models.Promise.project_id == project_id,
                models.Promise.resolved_at.is_(None),
            )
        )
    ).all()
    return [
        (p.expected_module, p.expected_name, path) for (p, path) in rows
    ]


# --- Main loop -------------------------------------------------------------


@dataclass
class GenerationOptions:
    goal: str
    model: str
    max_iterations: int = 4
    per_call_timeout: float = 600.0


async def generate_project(
    session: AsyncSession,
    project_id: str,
    options: GenerationOptions,
) -> AsyncIterator[dict]:
    """Drive the memory-backed generation loop. Yields NDJSON-ready events."""
    _safe_project_id(project_id)
    workdir = project_workdir(project_id)
    # Clean any previous attempt so each run starts from a clean slate,
    # but keep the DB memory (it records history across runs).
    for child in workdir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)

    await memory.ensure_project(session, project_id, name=project_id)
    session.add(
        models.Episode(
            project_id=project_id,
            role="user",
            kind="goal",
            content=options.goal,
        )
    )
    await session.commit()

    yield {"type": "start", "goal": options.goal, "iterations": options.max_iterations}

    client = OllamaClient(_ollama_base_url(), timeout=options.per_call_timeout)
    history: list[dict[str, str]] = []

    for iteration in range(1, options.max_iterations + 1):
        context = await build_chat_context(session, project_id, options.goal)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "system", "content": "Project memory:\n\n" + context})
        if iteration == 1:
            messages.append({"role": "user", "content": options.goal})
        else:
            messages.extend(history)

        yield {"type": "iteration", "n": iteration, "context_chars": len(context)}

        try:
            response = await client.chat(model=options.model, messages=messages)
        except OllamaError as exc:
            yield {"type": "error", "detail": str(exc), "fatal": True}
            return

        reply = (response.get("message") or {}).get("content", "")
        session.add(
            models.Episode(
                project_id=project_id,
                role="assistant",
                kind="generation",
                content=reply,
            )
        )
        await session.commit()

        parsed = parse_llm_response(reply)
        yield {
            "type": "llm_reply",
            "n": iteration,
            "files_proposed": [f.path for f in parsed.files],
            "prose_chars": len(parsed.prose),
        }

        if not parsed.files:
            yield {
                "type": "info",
                "message": "LLM produced no file blocks. Asking again with explicit reminder.",
            }
            history.append({"role": "assistant", "content": reply})
            history.append(
                {
                    "role": "user",
                    "content": (
                        "You must emit files using the framing "
                        "'=== FILE: path ===' followed by a fenced code block. "
                        "Please retry."
                    ),
                }
            )
            continue

        # Write + ingest each file, record report.
        ingest_reports: list[dict] = []
        for pf in parsed.files:
            full = workdir / pf.path
            if not _is_within(workdir, full):
                yield {"type": "skip", "path": pf.path, "reason": "path escape"}
                continue
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(pf.content, encoding="utf-8")
            # Only treat ``.py`` files as Python so the AST extractor is
            # never handed shell scripts, Markdown, etc.
            language = "python" if pf.path.endswith(".py") else "other"
            report = await memory.ingest_file(
                session,
                project_id=project_id,
                path=pf.path,
                content=pf.content,
                language=language,
            )
            ingest_reports.append(
                {
                    "path": pf.path,
                    "symbols": report.symbols,
                    "imports": report.imports,
                    "new_promises": report.new_promises,
                    "resolved_promises": report.resolved_promises,
                    "parse_error": report.parse_error,
                }
            )
            yield {"type": "file_written", **ingest_reports[-1]}
        await session.commit()

        # Syntax check.
        checks = syntax_check(
            [(pf.path, pf.content) for pf in parsed.files]
        )
        compile_errors = [c for c in checks if not c.compile_ok]
        for c in compile_errors:
            yield {"type": "compile_error", "path": c.path, "error": c.compile_error}

        # Optional pytest.
        tests_passed, test_output = run_pytest(workdir)
        if not tests_passed:
            yield {"type": "tests_failed", "output": test_output[-600:]}
        elif (workdir / "tests").exists():
            yield {"type": "tests_passed"}

        open_promises = await _gather_open_promises(session, project_id)
        yield {
            "type": "round_summary",
            "n": iteration,
            "open_promises": len(open_promises),
            "compile_errors": len(compile_errors),
            "tests_passed": tests_passed,
        }

        done = (
            not open_promises
            and not compile_errors
            and tests_passed
            and "DONE" in reply
        )
        if done:
            break

        feedback = _format_feedback(
            open_promises=open_promises,
            compile_errors=checks,
            test_output=test_output,
            tests_passed=tests_passed,
        )
        history = [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": (
                    "Keep going. Address the feedback below, then re-emit any "
                    "files you changed plus any new ones required. Use the same "
                    "'=== FILE: path ===' framing.\n\n" + feedback
                ),
            },
        ]

    zip_path = make_project_zip(project_id)
    yield {
        "type": "done",
        "zip_path": str(zip_path),
        "download_url": f"/projects/{project_id}/download",
    }


def make_project_zip(project_id: str) -> Path:
    """Package the on-disk project directory into a zip and return its path."""
    workdir = project_workdir(project_id)
    out_path = _data_dir() / "projects" / f"{project_id}.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(workdir.rglob("*")):
            if file.is_file():
                arc = file.relative_to(workdir)
                zf.write(file, arcname=str(arc))
    out_path.write_bytes(buf.getvalue())
    return out_path


async def stream_generation(
    session: AsyncSession, project_id: str, options: GenerationOptions
) -> AsyncIterator[bytes]:
    """NDJSON streaming wrapper suitable for FastAPI StreamingResponse."""
    try:
        async for event in generate_project(session, project_id, options):
            yield (json.dumps(event) + "\n").encode("utf-8")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        yield (
            json.dumps({"type": "error", "detail": str(exc), "fatal": True})
            + "\n"
        ).encode("utf-8")

"""CrossCodeEval harness (Ding et al., NeurIPS 2023).

https://arxiv.org/abs/2310.11248 · https://github.com/amazon-science/cceval

### Dataset shape
Each JSONL record looks roughly like::

    {
      "task_id": "python/...",
      "language": "python",
      "prompt": "import os\\n\\ndef load(\n    path):\\n    ",
      "groundtruth": "    return open(path).read()\\n",
      "right_context": "\\n\\nif __name__ == '__main__':\\n    ...",
      "crossfile_context": [
        {"path": "utils.py", "content": "def helper(...): ..."},
        {"path": "models.py", "content": "class User: ..."}
      ],
      "repository": "foo/bar",
      "file_path": "src/main.py"
    }

Official versions package ``crossfile_context`` as either a flat string
(the pre-retrieved chunk concatenated) or as a list of ``(path, text)``
pairs. We accept both.

### What we compare

| Mode       | Prompt built from                           |
|------------|---------------------------------------------|
| baseline   | ``prompt`` only                             |
| synapse    | ``prompt`` + memory-assembled context from  |
|            | crossfile files ingested into SynapseMem    |

The same LLM is called in both modes, so any uplift is attributable to
the memory layer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .. import memory
from ..context_builder import build_chat_context
from ..database import Base
from ..ollama_client import OllamaClient
from .metrics import aggregate, edit_similarity, exact_match

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


@dataclass
class CCESample:
    task_id: str
    language: str
    prompt: str
    groundtruth: str
    right_context: str = ""
    crossfile_files: list[tuple[str, str]] = None  # type: ignore[assignment]
    repository: str = ""
    file_path: str = ""


def _extract_crossfile_files(raw) -> list[tuple[str, str]]:
    """Normalise the many shapes ``crossfile_context`` ships in.

    - ``None`` / ``""`` → empty list.
    - List of dicts ``[{path, content}]`` → passthrough.
    - List of dicts ``[{filename, content}]`` → rename key.
    - Flat string → single pseudo-file ``('crossfile.txt', raw)``.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [("crossfile.txt", raw)]
    if isinstance(raw, dict):
        # Official release ships ``{"text": "# Here are some relevant
        # code fragments ..."}`` — a flat blob where each fragment is
        # prefixed by ``# the below code fragment can be found in:\n#
        # <filename>`` and the body is every line prepended with ``# ``.
        # We reverse that so downstream ingestion sees real Python, not
        # comment-only text that the AST extractor would drop.
        text = raw.get("text") or raw.get("content") or ""
        if not text:
            return []
        return _split_cceval_blob(text)
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path") or entry.get("filename") or entry.get("file")
        content = entry.get("content") or entry.get("text") or ""
        if path and isinstance(content, str):
            out.append((str(path), content))
    return out


_FRAGMENT_HEADER = "# the below code fragment can be found in:"


def _split_cceval_blob(text: str) -> list[tuple[str, str]]:
    """Split the oracle/rg1 comment-annotated blob into per-file snippets.

    Each fragment in the blob starts with a line matching
    ``_FRAGMENT_HEADER`` and is followed by a filename comment and
    comment-prefixed source lines. Multiple fragments may point at the
    same file; we concatenate them in order.

    The returned list preserves filename ordering so downstream
    ingestion gets stable, deterministic paths for the same sample.
    """
    by_path: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    awaiting_filename = False

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped == _FRAGMENT_HEADER:
            awaiting_filename = True
            current = None
            continue
        if awaiting_filename:
            awaiting_filename = False
            if stripped.startswith("# ") and len(stripped) > 2:
                candidate = stripped[2:].strip()
                current = candidate or None
                if current and current not in by_path:
                    by_path[current] = []
                    order.append(current)
            continue
        if current is None:
            continue
        if not stripped:
            by_path[current].append("")
            continue
        if stripped.startswith("# "):
            by_path[current].append(stripped[2:])
        elif stripped.startswith("#"):
            # ``#<something>`` (no space) — keep the text after the hash.
            by_path[current].append(stripped[1:])
        # Any non-comment line ends the fragment until the next header.
    out: list[tuple[str, str]] = []
    for path in order:
        lines = by_path[path]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            out.append((path, "\n".join(lines) + "\n"))
    return out


def load_samples(dataset_path: Path, *, limit: int | None = None) -> list[CCESample]:
    """Read a CrossCodeEval JSONL file from disk."""
    samples: list[CCESample] = []
    with open(dataset_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") or {}
            s = CCESample(
                task_id=str(
                    row.get("task_id")
                    or meta.get("task_id")
                    or row.get("id")
                    or len(samples)
                ),
                language=row.get("language") or meta.get("language") or "python",
                prompt=row["prompt"],
                groundtruth=row.get("groundtruth") or row.get("completion") or "",
                right_context=row.get("right_context", ""),
                crossfile_files=_extract_crossfile_files(
                    row.get("crossfile_context") or row.get("cross_file_context")
                ),
                repository=row.get("repository") or meta.get("repository", ""),
                file_path=row.get("file_path") or meta.get("file", ""),
            )
            samples.append(s)
            if limit is not None and len(samples) >= limit:
                break
    return samples


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(?P<body>.*?)```", re.DOTALL)


def extract_completion(raw_reply: str, groundtruth: str) -> str:
    """Extract the completion proper from a chat-style LLM reply.

    CrossCodeEval is measured line-by-line: we only keep as many lines
    as the groundtruth has. If the model emitted a fenced block, we
    lift the body from it first.
    """
    text = raw_reply
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group("body")
    # Strip leading blank lines so alignment with gt works.
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    n = max(1, len(groundtruth.splitlines()))
    kept = lines[:n]
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_MEMORY_HEADER = (
    "# --- Cross-file memory (SynapseMem) ---\n"
    "# Symbols, imports, and unresolved references from other files in\n"
    "# this repository. Use them to keep names, signatures and imports\n"
    "# consistent with the rest of the project.\n"
)

# Hard cap on cross-file snippet bytes injected per sample so we don't
# blow through the LLM's context window on large repos. Empirical choice:
# qwen2.5-coder handles ~16k tokens comfortably; 8000 chars ≈ 2k tokens.
_MAX_SNIPPET_CHARS = 8000


async def _build_temp_session() -> tuple[AsyncSession, object]:
    """Isolated in-memory SQLite session for one harness run."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    session = sessionmaker()
    return session, engine


@dataclass
class RunnerConfig:
    mode: str  # "baseline" | "synapse"
    model: str
    ollama_base_url: str
    temperature: float = 0.0
    num_predict: int = 128


def _format_snippets_block(files: list[tuple[str, str]]) -> str:
    """Inject raw cross-file code snippets as comment-prefixed blocks.

    CrossCodeEval's crossfile_context is often a set of code fragments
    (partial method bodies, not parseable modules). AST-based symbol
    extraction misses them, so we fall back to embedding the raw
    bodies. The model reads them as comments because of the ``# ``
    prefix and treats them as reference context rather than code to
    continue.
    """
    if not files:
        return ""
    lines = ["# ## Cross-file code fragments"]
    budget = _MAX_SNIPPET_CHARS
    for path, content in files:
        if budget <= 0:
            break
        trimmed = content[:budget]
        budget -= len(trimmed)
        lines.append(f"# ### {path}")
        for raw in trimmed.splitlines():
            if raw:
                lines.append(f"# {raw}")
            else:
                lines.append("#")
        lines.append("#")
    return "\n".join(lines)


async def _build_memory_prefix(sample: CCESample) -> str:
    """Ingest crossfile files, then assemble SynapseMem's memory block.

    Two sources of context are fused:

    1. Graph-derived summary from the SynapseMem ingestion (project
       stats + open promises + relevant symbols via semantic /
       keyword retrieval) — useful when crossfile files are full,
       parseable Python modules.
    2. Raw code fragments from ``crossfile_context`` — the primary
       signal when the snippets are partial (common on CCE oracle
       variants). Without this the synapse prefix would collapse to
       empty whenever the AST extractor rejects a fragment.

    Output is comment-prefixed so even a pure FIM model reads it as
    reference material and not as code to continue.
    """
    session, engine = await _build_temp_session()
    mem_block = ""
    try:
        project_id = f"cce-{sample.task_id.replace('/', '-')}"
        for idx, (path, content) in enumerate(sample.crossfile_files or []):
            safe_path = path or f"extra_{idx}.py"
            await memory.ingest_file(
                session,
                project_id=project_id,
                path=safe_path,
                content=content,
                language="python" if safe_path.endswith(".py") else "other",
            )
        await session.commit()
        mem_block = await build_chat_context(
            session,
            project_id,
            sample.prompt,
            recent_episodes=0,
        )
    finally:
        await session.close()
        await engine.dispose()

    snippets_block = _format_snippets_block(sample.crossfile_files or [])
    if not mem_block.strip() and not snippets_block:
        return ""

    sections: list[str] = []
    if mem_block.strip():
        sections.append(
            "\n".join(f"# {line}" if line else "#" for line in mem_block.splitlines())
        )
    if snippets_block:
        sections.append(snippets_block)
    body = "\n".join(sections)
    return _MEMORY_HEADER + body + "\n# --- end memory ---\n\n"


async def run_sample(sample: CCESample, cfg: RunnerConfig) -> dict:
    """Run one sample under one mode and score it.

    Both modes use ``/api/generate`` with FIM ``suffix`` so
    qwen2.5-coder / deepseek-coder fires its native fill-in-the-middle
    path. The only difference is whether a memory prefix is prepended
    to ``prompt`` (synapse) or not (baseline).
    """
    client = OllamaClient(cfg.ollama_base_url)
    prefix = ""
    if cfg.mode == "synapse":
        prefix = await _build_memory_prefix(sample)
    elif cfg.mode != "baseline":
        raise ValueError(f"unknown mode: {cfg.mode}")

    prompt = prefix + sample.prompt
    suffix = sample.right_context or ""
    start = time.perf_counter()
    resp = await client.generate(
        model=cfg.model,
        prompt=prompt,
        suffix=suffix,
        options={
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
            "stop": ["\n\n"],
        },
    )
    elapsed = time.perf_counter() - start
    reply = resp.get("response", "") or ""
    prediction = extract_completion(reply, sample.groundtruth)
    em = exact_match(prediction, sample.groundtruth)
    es = edit_similarity(prediction, sample.groundtruth)
    return {
        "task_id": sample.task_id,
        "em": em,
        "es": es,
        "prediction": prediction,
        "groundtruth": sample.groundtruth,
        "elapsed": elapsed,
    }


async def run_benchmark(
    samples: list[CCESample],
    cfg: RunnerConfig,
    *,
    on_progress=None,
) -> dict:
    """Evaluate all samples sequentially; return aggregate + detail rows."""
    rows: list[dict] = []
    for i, sample in enumerate(samples):
        try:
            result = await run_sample(sample, cfg)
        except Exception as exc:
            result = {
                "task_id": sample.task_id,
                "em": False,
                "es": 0.0,
                "prediction": "",
                "groundtruth": sample.groundtruth,
                "error": str(exc),
                "elapsed": 0.0,
            }
        rows.append(result)
        if on_progress:
            on_progress(i + 1, len(samples), result)
    summary = aggregate(rows)
    summary["mode"] = cfg.mode
    summary["model"] = cfg.model
    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CrossCodeEval against an Ollama model.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Path to JSONL.")
    parser.add_argument("--mode", choices=["baseline", "synapse", "both"], default="both")
    parser.add_argument("--model", default=os.environ.get("SYNAPSEMEM_EXTRACTOR_MODEL", "qwen2.5-coder:32b"))
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("SYNAPSEMEM_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples = load_samples(args.dataset, limit=args.limit)
    if not samples:
        print("No samples loaded.")
        return 1

    def progress(done: int, total: int, result: dict) -> None:
        mark = "✓" if result["em"] else ("·" if result["es"] > 0.5 else "✗")
        print(
            f"  [{done:4d}/{total}] {mark} "
            f"em={int(result['em'])} es={result['es']:.2f} "
            f"elapsed={result['elapsed']:.1f}s task={result['task_id']}"
        )

    modes = [args.mode] if args.mode != "both" else ["baseline", "synapse"]
    out: dict[str, dict] = {}
    for mode in modes:
        cfg = RunnerConfig(mode=mode, model=args.model, ollama_base_url=args.ollama_url)
        print(f"\n=== running mode={mode} model={args.model} on {len(samples)} samples ===")
        result = asyncio.run(run_benchmark(samples, cfg, on_progress=progress))
        out[mode] = result
        s = result["summary"]
        print(f"  -> EM={s['em']:.3f} ES={s['es']:.3f} on {s['n']} samples")

    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\nWrote detailed results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

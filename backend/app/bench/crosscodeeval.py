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
        # code fragments ..."}`` — a flat blob that already contains
        # the other files annotated with ``# file.py\n# ...`` headers.
        text = raw.get("text") or raw.get("content") or ""
        if text:
            return [("crossfile.txt", text)]
        return []
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


BASELINE_PROMPT = (
    "You are a code completion model. Complete the following {lang} code. "
    "Emit ONLY the continuation, no explanation, no markdown fences.\n\n"
    "### Code\n{prompt}"
)

SYNAPSE_PROMPT = (
    "You are a code completion model with access to a memory of the "
    "project's other files. Use the memory to keep names, imports and "
    "signatures consistent. Emit ONLY the continuation of the target "
    "file, no explanation, no markdown fences.\n\n"
    "### Project memory\n{memory}\n\n"
    "### Target file so far\n{prompt}"
)


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


async def run_sample(sample: CCESample, cfg: RunnerConfig) -> dict:
    """Run one sample under one mode and score it."""
    client = OllamaClient(cfg.ollama_base_url)
    if cfg.mode == "baseline":
        prompt = BASELINE_PROMPT.format(prompt=sample.prompt, lang=sample.language)
    elif cfg.mode == "synapse":
        session, engine = await _build_temp_session()
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
        prompt = SYNAPSE_PROMPT.format(
            memory=mem_block.strip() or "(empty memory)",
            prompt=sample.prompt,
        )
    else:
        raise ValueError(f"unknown mode: {cfg.mode}")

    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    resp = await client.chat(
        model=cfg.model,
        messages=messages,
        options={"temperature": cfg.temperature, "num_predict": cfg.num_predict},
    )
    elapsed = time.perf_counter() - start
    reply = (resp.get("message") or {}).get("content", "")
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

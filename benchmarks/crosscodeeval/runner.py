"""Driver that runs one configuration end-to-end over a set of tasks.

Each run:

1. Builds the retriever on the task's repo (minus the target file).
2. Retrieves a cross-file context block.
3. Assembles a completion prompt and calls the model via Ollama.
4. Computes EM + ES against the groundtruth.
5. Writes per-task rows to a JSONL log.
6. Emits aggregate stats to stdout and (optionally) to BENCHMARKS.md.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import Task
from .metrics import edit_similarity, exact_match
from .retrievers import Retriever


@dataclass
class TaskResult:
    task_id: str
    config: str
    exact_match: float
    edit_similarity: float
    prediction: str
    groundtruth: str
    context_chars: int
    latency_s: float


@dataclass
class AggregateResult:
    config: str
    n: int
    em: float
    es: float


def run_config(
    tasks: Iterable[Task],
    retriever: Retriever,
    *,
    complete_fn,
    out_path: Path | None = None,
    max_context_chars: int = 8000,
) -> AggregateResult:
    results: list[TaskResult] = []
    fh = out_path.open("w", encoding="utf-8") if out_path else None
    try:
        for task in tasks:
            retriever.build(task.repo_files)
            query = _query_from_left_context(task.left_context)
            t0 = time.perf_counter()
            ctx = retriever.retrieve(
                query, target_path=task.path, max_chars=max_context_chars
            )
            prompt = _build_prompt(ctx, task)
            prediction = complete_fn(prompt)
            latency = time.perf_counter() - t0
            em = exact_match(prediction, task.groundtruth)
            es = edit_similarity(prediction, task.groundtruth)
            row = TaskResult(
                task_id=task.task_id,
                config=retriever.name,
                exact_match=em,
                edit_similarity=es,
                prediction=prediction[:500],
                groundtruth=task.groundtruth,
                context_chars=len(ctx),
                latency_s=round(latency, 3),
            )
            results.append(row)
            if fh:
                fh.write(json.dumps(asdict(row)) + "\n")
    finally:
        if fh:
            fh.close()
    n = len(results) or 1
    return AggregateResult(
        config=retriever.name,
        n=len(results),
        em=sum(r.exact_match for r in results) / n,
        es=sum(r.edit_similarity for r in results) / n,
    )


def _query_from_left_context(left: str, *, last_n_lines: int = 8) -> str:
    """Build a short query hint from the tail of the left context.

    The retriever only needs enough to anchor the search — typically the
    function signature + the last few lines tell us which symbols are
    relevant.
    """
    lines = [ln for ln in left.splitlines() if ln.strip()]
    return "\n".join(lines[-last_n_lines:])


def _build_prompt(ctx: str, task: Task) -> str:
    header = ""
    if ctx:
        header = (
            "# Cross-file context from the repo (read before completing):\n"
            f"{ctx}\n"
            "# ---- end of cross-file context ----\n\n"
        )
    return (
        f"{header}"
        f"# Complete the next single line of `{task.path}`. "
        "Reply with ONLY that line, no commentary.\n\n"
        f"{task.left_context}"
    )

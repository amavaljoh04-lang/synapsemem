"""CrossCodeEval dataset loader.

The official release (amazon-science/cceval) ships one ``data.jsonl``
per language plus a ``repo/`` directory with the source files of every
repository referenced by the tasks. This module reads both.

Expected layout (download once, cache on disk) — adapt ``--data-dir`` to
where you put the files::

    data/cceval/
      python/data.jsonl
      python/repo/<repo_name>/<path>/<file>
      java/data.jsonl
      java/repo/...

Each line of ``data.jsonl`` has (at minimum) the fields:

* ``task_id`` — unique per task.
* ``path`` — target file within the repo.
* ``left_context`` — text BEFORE the cursor inside that file.
* ``groundtruth`` — the single line the model must predict.
* ``right_context`` — text AFTER the cursor (unused by the model, but
  used by some metrics).
* ``repo`` — name of the repo directory under ``repo/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Task:
    task_id: str
    repo: str
    path: str
    left_context: str
    right_context: str
    groundtruth: str
    repo_files: dict[str, str]


def iter_tasks(
    data_dir: Path,
    language: str,
    *,
    limit: int | None = None,
    task_ids: set[str] | None = None,
) -> Iterator[Task]:
    """Yield :class:`Task` records for the requested language split.

    Parameters
    ----------
    data_dir:
        Directory containing ``<language>/data.jsonl`` and ``<language>/repo``.
    language:
        ``python`` / ``java`` / ``typescript`` / ``csharp``.
    limit:
        Optional cap on the number of tasks yielded (for smoke runs).
    task_ids:
        Optional set of task ids to restrict to.
    """
    split_dir = data_dir / language
    jsonl = split_dir / "data.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"CrossCodeEval data not found at {jsonl}. "
            "Download from https://github.com/amazon-science/cceval."
        )
    repo_root = split_dir / "repo"
    seen = 0
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if task_ids is not None and row["task_id"] not in task_ids:
                continue
            repo_name = row["repo"]
            files = _read_repo(repo_root / repo_name)
            yield Task(
                task_id=row["task_id"],
                repo=repo_name,
                path=row["path"],
                left_context=row.get("left_context", ""),
                right_context=row.get("right_context", ""),
                groundtruth=row["groundtruth"],
                repo_files=files,
            )
            seen += 1
            if limit is not None and seen >= limit:
                return


def _read_repo(repo_path: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not repo_path.exists():
        return files
    for p in repo_path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_path).as_posix()
        try:
            files[rel] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return files

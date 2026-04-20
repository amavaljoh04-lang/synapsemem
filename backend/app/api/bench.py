"""HTTP surface for running + inspecting SynapseMem benchmarks.

Endpoints:

* ``POST /bench/crosscodeeval/run`` — starts a CrossCodeEval run in a
  background task and streams NDJSON events to the caller by tailing
  the on-disk event log. Client disconnects do **not** abort the run;
  the background task keeps writing events to
  ``data/bench/live/{run_id}.ndjson`` until it finishes.
* ``GET  /bench/live`` — returns the currently active run (if any), so
  the UI can reattach on page load instead of losing progress.
* ``GET  /bench/live/{run_id}/stream`` — NDJSON stream of events for
  ``run_id``. Works for live **and** finished runs (replays from disk).
* ``GET  /bench/runs`` — list past runs (newest first).
* ``GET  /bench/runs/{run_id}`` — full detail (rows + summaries).
* ``GET  /bench/dataset/status`` — tells the UI whether the CCE files
  are installed and which languages/variants are available.

The runs are persisted on disk under ``settings.data_dir / bench`` so
the scoreboard survives container restarts without adding another DB
table.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..bench import dataset as cce_dataset
from ..bench.crosscodeeval import RunnerConfig, load_samples, run_sample
from ..bench.metrics import aggregate
from ..config import settings

router = APIRouter(prefix="/bench", tags=["bench"])

# Background tasks are kept alive explicitly to stop the asyncio GC from
# collecting them mid-run (ruff RUF006). This set holds refs until each
# task signals completion via its ``done`` callback.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


class CCERunRequest(BaseModel):
    model: str = Field(default="qwen2.5-coder:7b")
    # Upper bound covers the full CrossCodeEval Python split (~2 666
    # samples). Java/TS/C# are a similar order of magnitude.
    limit: int = Field(default=20, ge=1, le=10_000)
    mode: str = Field(default="both", pattern="^(baseline|synapse|both)$")
    language: str = Field(default=cce_dataset.DEFAULT_LANGUAGE)
    variant: str = Field(default=cce_dataset.DEFAULT_VARIANT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_predict: int = Field(default=128, ge=1, le=2048)


def _runs_dir() -> Path:
    d = settings.data_dir / "bench" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _live_dir() -> Path:
    d = settings.data_dir / "bench" / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _active_path() -> Path:
    return _live_dir() / "active.json"


def _events_path(run_id: str) -> Path:
    return _live_dir() / f"{run_id}.ndjson"


def _save_run(run: dict) -> None:
    (_runs_dir() / f"{run['id']}.json").write_text(json.dumps(run, indent=2))


# ---------------------------------------------------------------------------
# Dataset helpers (unchanged)
# ---------------------------------------------------------------------------


@router.get("/dataset/status")
async def dataset_status() -> dict:
    """Report whether the CCE dataset is already installed locally."""
    return {
        "installed": cce_dataset.is_installed(),
        "root": str(cce_dataset._root()),
        "languages": cce_dataset.available_languages(),
        "python_variants": cce_dataset.available_variants("python"),
    }


@router.post("/dataset/install")
async def dataset_install() -> dict:
    """Kick the (possibly long) download + extraction synchronously.

    The UI pings this once if ``status.installed`` is false. It blocks
    up to a few minutes on the first call, then returns instantly.
    """
    path = await cce_dataset.ensure_installed()
    return {"installed": True, "path": str(path)}


# ---------------------------------------------------------------------------
# Historical runs (unchanged)
# ---------------------------------------------------------------------------


@router.get("/runs")
async def list_runs(limit: int = 50) -> list[dict]:
    """Return the latest ``limit`` runs, newest first, summary-only."""
    runs: list[dict] = []
    for file in sorted(_runs_dir().glob("*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(file.read_text())
        except Exception:
            continue
        runs.append(
            {
                "id": data.get("id"),
                "model": data.get("model"),
                "mode": data.get("mode"),
                "limit": data.get("limit"),
                "language": data.get("language"),
                "variant": data.get("variant"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
                "summaries": data.get("summaries"),
            }
        )
    return runs


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    path = _runs_dir() / f"{run_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Live runs — background task + tailable event log
# ---------------------------------------------------------------------------


async def _generate_run_events(run_id: str, req: CCERunRequest) -> AsyncIterator[dict]:
    """Produce the NDJSON event stream for one CCE run.

    Pure generator — no side effects beyond reading the dataset and
    calling the LLM. The persistence + tee-to-file layer wraps this.
    """
    started_at = time.time()
    run: dict = {
        "id": run_id,
        "model": req.model,
        "mode": req.mode,
        "limit": req.limit,
        "language": req.language,
        "variant": req.variant,
        "started_at": started_at,
        "rows": {"baseline": [], "synapse": []},
        "summaries": {},
    }

    yield {"event": "status", "phase": "preparing"}
    try:
        if not cce_dataset.is_installed():
            yield {"event": "status", "phase": "downloading-dataset"}
            await cce_dataset.ensure_installed()
        ds_path = cce_dataset.dataset_path(req.language, req.variant)
        if not ds_path.is_file():
            yield {"event": "error", "error": f"dataset not found on disk: {ds_path}"}
            return

        samples = load_samples(ds_path, limit=req.limit)
        yield {
            "event": "status",
            "phase": "loaded",
            "n_samples": len(samples),
            "dataset": str(ds_path.name),
        }

        modes = [req.mode] if req.mode != "both" else ["baseline", "synapse"]
        for mode in modes:
            cfg = RunnerConfig(
                mode=mode,
                model=req.model,
                ollama_base_url=settings.ollama_base_url,
                temperature=req.temperature,
                num_predict=req.num_predict,
            )
            rows: list[dict] = []
            yield {
                "event": "status",
                "phase": "running",
                "mode": mode,
                "total": len(samples),
            }
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
                yield {
                    "event": "sample",
                    "mode": mode,
                    "done": i + 1,
                    "total": len(samples),
                    "result": result,
                }
                # Let the event loop flush bytes to the client so the
                # progress bar actually ticks in real time.
                await asyncio.sleep(0)
            summary = aggregate(rows)
            summary["mode"] = mode
            summary["model"] = req.model
            run["rows"][mode] = rows
            run["summaries"][mode] = summary
            yield {"event": "summary", "mode": mode, "summary": summary}

        run["finished_at"] = time.time()
        _save_run(run)
        yield {
            "event": "run",
            "run": {
                "id": run_id,
                "model": req.model,
                "mode": req.mode,
                "limit": req.limit,
                "language": req.language,
                "variant": req.variant,
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "summaries": run["summaries"],
            },
        }
    except Exception as exc:  # report + give up cleanly
        yield {"event": "error", "error": str(exc)}


async def _execute_and_persist(run_id: str, req: CCERunRequest) -> None:
    """Drive the generator, tee each event to disk and mark active state.

    Runs detached from any HTTP request so a client disconnect does not
    interrupt the benchmark. Events are appended to the run's NDJSON
    file; the ``active.json`` pointer is cleared once the run completes
    (success or error).
    """
    events_file = _events_path(run_id)
    events_file.parent.mkdir(parents=True, exist_ok=True)
    # Seed meta + active pointer so /bench/live can discover us
    # immediately, before the first event lands on disk.
    meta = {
        "id": run_id,
        "model": req.model,
        "mode": req.mode,
        "limit": req.limit,
        "language": req.language,
        "variant": req.variant,
        "started_at": time.time(),
    }
    _active_path().write_text(json.dumps(meta))
    try:
        async for evt in _generate_run_events(run_id, req):
            with events_file.open("a") as fh:
                fh.write(json.dumps(evt) + "\n")
    finally:
        # Mark the log as "closed" so tailers stop polling quickly.
        with events_file.open("a") as fh:
            fh.write(json.dumps({"event": "closed"}) + "\n")
        if _active_path().is_file():
            with contextlib.suppress(OSError):
                _active_path().unlink()


async def _tail_events(
    run_id: str,
    *,
    poll_interval: float = 0.25,
    idle_timeout: float = 30.0,
) -> AsyncIterator[bytes]:
    """Yield NDJSON bytes as they appear in the run's event log.

    Returns terminal when a ``closed`` / ``run`` / ``error`` event is
    seen OR when ``idle_timeout`` seconds pass without any new bytes
    (safety net for runs that died without writing a terminator).
    """
    path = _events_path(run_id)
    last_size = 0
    last_change = time.monotonic()
    saw_terminal = False

    while True:
        if path.is_file():
            size = path.stat().st_size
            if size > last_size:
                with path.open("rb") as fh:
                    fh.seek(last_size)
                    new_bytes = fh.read(size - last_size)
                last_size = size
                last_change = time.monotonic()
                # Yield whole lines only, buffering the partial tail.
                text = new_bytes.decode("utf-8", errors="replace")
                for line in text.splitlines(keepends=False):
                    if not line:
                        continue
                    yield (line + "\n").encode("utf-8")
                    try:
                        parsed = json.loads(line)
                    except Exception:
                        parsed = {}
                    ev = parsed.get("event")
                    if ev in {"closed", "run", "error"}:
                        saw_terminal = True
                if saw_terminal:
                    return
        if time.monotonic() - last_change > idle_timeout:
            return
        await asyncio.sleep(poll_interval)


def _read_active() -> dict | None:
    p = _active_path()
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text())
    except Exception:
        return None
    # Double-check the log file exists; otherwise the pointer is stale.
    if not _events_path(meta.get("id", "")).is_file():
        return None
    return meta


@router.get("/live")
async def live_status() -> dict:
    """Return the currently active run (if any) and whether it's running."""
    active = _read_active()
    return {"active": active}


@router.get("/live/{run_id}/stream")
async def live_stream(run_id: str) -> StreamingResponse:
    """Stream (or replay) a run's NDJSON event log."""
    if not _events_path(run_id).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no event log for run {run_id}",
        )
    return StreamingResponse(_tail_events(run_id), media_type="application/x-ndjson")


@router.post("/crosscodeeval/run")
async def crosscodeeval_run(req: CCERunRequest) -> dict:
    """Start a CrossCodeEval run in the background and return its id.

    The run keeps going even if the HTTP client disconnects. Use
    ``GET /bench/live/{run_id}/stream`` (or ``GET /bench/live`` on
    page load) to follow the NDJSON events.
    """
    if _read_active() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="another benchmark run is already in progress",
        )
    run_id = uuid.uuid4().hex[:12]
    # Touch the events file so /live/{run_id}/stream doesn't 404 while
    # the background task spins up.
    _events_path(run_id).touch()
    task = asyncio.create_task(_execute_and_persist(run_id, req))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return {
        "run_id": run_id,
        "stream_url": f"/bench/live/{run_id}/stream",
    }

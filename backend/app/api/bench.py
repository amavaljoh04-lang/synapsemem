"""HTTP surface for running + inspecting SynapseMem benchmarks.

Endpoints:

* ``POST /bench/crosscodeeval/run`` — streams NDJSON events while the
  CrossCodeEval harness executes. One line per sample, one line per
  mode summary, one final ``run`` line with the aggregate + run id.
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
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..bench import dataset as cce_dataset
from ..bench.crosscodeeval import RunnerConfig, load_samples, run_sample
from ..bench.metrics import aggregate
from ..config import settings

router = APIRouter(prefix="/bench", tags=["bench"])


class CCERunRequest(BaseModel):
    model: str = Field(default="qwen2.5-coder:7b")
    limit: int = Field(default=20, ge=1, le=2000)
    mode: str = Field(default="both", pattern="^(baseline|synapse|both)$")
    language: str = Field(default=cce_dataset.DEFAULT_LANGUAGE)
    variant: str = Field(default=cce_dataset.DEFAULT_VARIANT)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_predict: int = Field(default=128, ge=1, le=2048)


def _runs_dir() -> Path:
    d = settings.data_dir / "bench" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_run(run: dict) -> None:
    (_runs_dir() / f"{run['id']}.json").write_text(json.dumps(run, indent=2))


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


@router.post("/crosscodeeval/run")
async def crosscodeeval_run(req: CCERunRequest) -> StreamingResponse:
    """Run CrossCodeEval and stream NDJSON progress to the client.

    Event shapes (one JSON object per line):

    ``{"event": "status", "phase": "preparing", ...}``
        Pre-flight updates (dataset install, sample count, etc.).
    ``{"event": "sample", "mode": "...", "done": n, "total": N, "result": {...}}``
        One per sample x mode.
    ``{"event": "summary", "mode": "...", "summary": {...}}``
        One per mode at the end of that mode's pass.
    ``{"event": "run", "run": {...}}``
        Final — includes the run id and all summaries.
    """

    async def _events():
        run_id = uuid.uuid4().hex[:12]
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

        yield json.dumps({"event": "status", "phase": "preparing"}) + "\n"
        try:
            if not cce_dataset.is_installed():
                yield json.dumps({"event": "status", "phase": "downloading-dataset"}) + "\n"
                await cce_dataset.ensure_installed()
            ds_path = cce_dataset.dataset_path(req.language, req.variant)
            if not ds_path.is_file():
                yield (
                    json.dumps(
                        {
                            "event": "error",
                            "error": f"dataset not found on disk: {ds_path}",
                        }
                    )
                    + "\n"
                )
                return

            samples = load_samples(ds_path, limit=req.limit)
            yield (
                json.dumps(
                    {
                        "event": "status",
                        "phase": "loaded",
                        "n_samples": len(samples),
                        "dataset": str(ds_path.name),
                    }
                )
                + "\n"
            )

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
                yield (
                    json.dumps(
                        {
                            "event": "status",
                            "phase": "running",
                            "mode": mode,
                            "total": len(samples),
                        }
                    )
                    + "\n"
                )
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
                    yield (
                        json.dumps(
                            {
                                "event": "sample",
                                "mode": mode,
                                "done": i + 1,
                                "total": len(samples),
                                "result": result,
                            }
                        )
                        + "\n"
                    )
                    # Let the event loop flush bytes to the client so
                    # the progress bar actually ticks in real time.
                    await asyncio.sleep(0)
                summary = aggregate(rows)
                summary["mode"] = mode
                summary["model"] = req.model
                run["rows"][mode] = rows
                run["summaries"][mode] = summary
                yield json.dumps({"event": "summary", "mode": mode, "summary": summary}) + "\n"

            run["finished_at"] = time.time()
            _save_run(run)
            yield (
                json.dumps(
                    {
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
                )
                + "\n"
            )
        except Exception as exc:  # report + give up cleanly
            yield json.dumps({"event": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(_events(), media_type="application/x-ndjson")

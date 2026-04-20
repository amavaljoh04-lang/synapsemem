# Agents / Contributor conventions

This repo welcomes human and agent contributors. Please respect the
following conventions.

## Stack

* Python 3.12, FastAPI, SQLAlchemy 2.x async, SQLite.
* Ruff for lint (line length 100).
* pytest for tests, `asyncio_mode = "auto"`.
* Plain HTML/JS frontend for v0.1 (no build step). A Vite/React frontend
  comes in a later milestone.

## Before committing

From `backend/`:

```bash
ruff check .
pytest -q
```

Both must pass. CI runs exactly these commands (`.github/workflows/ci.yml`).

## Philosophy

* **Structured over free-form.** Whenever the data has shape (like source
  code), represent the shape explicitly in the schema. Reserve free-form
  triplets (`Edge`) for things we really can't structure (natural-language
  extraction output).
* **Promises over implicit assumptions.** Anything the agent
  expects-but-hasn't-delivered should be a first-class row. That's the
  single most important idea in the project.
* **Benchmarks or it didn't happen.** Features that claim to improve
  retrieval must be measured against a baseline on a recognized benchmark.

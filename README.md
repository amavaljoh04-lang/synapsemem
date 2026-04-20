# SynapseMem

> Cognitive-inspired memory for coding agents.
> When a model writes a 50-file project, it should **remember** what it already
> wrote, what it promised to expose, and what the user asked for — across every
> file, for the whole session, without re-reading the entire codebase on every
> prompt.

SynapseMem is a standalone service that augments any LLM (local Ollama or
remote API) with a persistent, introspectable project memory. It's designed
to be a **proof-of-concept that beats vector-only RAG on cross-file code
tasks**, measured on recognized benchmarks.

---

## The problem we're solving

Today, when you ask a code-generation agent to scaffold a multi-file project:

1. **File 1** imports `parse_config` from `utils.py`.
2. Several tasks later, when the agent actually writes `utils.py`, it has
   forgotten which symbols were promised — it invents a different name, a
   different signature, or it writes a helper that nobody asked for and
   silently drops `parse_config`.
3. You end up with `ImportError: cannot import name 'parse_config'` after a
   30-minute run.

Vector RAG helps a bit (you can recall the text of File 1), but it does not
*track the commitments*. What the agent needs is a **structured memory** that
says, at any point: "`parse_config` has been referenced in File 1 but is not
yet defined anywhere — next time you touch `utils.py`, you owe me this
symbol with this signature".

That's what SynapseMem tracks.

---

## Architecture at a glance

Four cognitively-inspired layers, served by a single local model:

```
┌───────────────────────────────────────────────────────────────┐
│  1. Episodic   (SQLite)     — every prompt/file/note, dated   │
│  2. Semantic   (graph)      — entities, symbols, relations    │
│  3. Procedural (rules)      — learned user preferences        │
│  4. Reflection (worker)     — consolidation, decay, contrad.  │
└───────────────────────────────────────────────────────────────┘
                   ▲                               │
                   │                               │
                   │         Retrieval             ▼
       ┌───────────┴────────────┐      ┌──────────────────┐
       │  Ingest endpoints      │      │  Query endpoints │
       │  /ingest/file /text    │      │  /graph /symbols │
       │  /ingest/directory     │      │  /promises /ask  │
       └───────────┬────────────┘      └─────────┬────────┘
                   │                              │
                   │                              ▼
              (any LLM)                  ┌──────────────────┐
                                         │  Chat + live     │
                                         │  graph UI        │
                                         └──────────────────┘
```

**Episodic** is ground truth — we never lose the raw data.

**Semantic** is a graph (NetworkX-backed, SQLite-persisted) of files,
symbols (classes / functions / variables), imports, and **promises** —
references that are not yet satisfied anywhere in the codebase.

**Procedural** captures things like "this user always uses `argparse`,
never `click`" as learned rules with confidence, so the next project
respects the preference without being re-told.

**Reflection** runs periodically (and on-demand) to merge redundant nodes,
decay unused edges, and surface contradictions ("file 7 uses click but
user's stated preference is argparse — one of these is wrong").

---

## Benchmarks

The whole point is to **prove** this works, on benchmarks the community
already uses. We target three:

| Benchmark | Why | Target |
|---|---|---|
| [CrossCodeEval](https://arxiv.org/abs/2310.11248) | Direct test of cross-file code completion — you have to know what another file defines to complete the current one. This is our exact use case. | +20 pts Exact Match vs same-model baseline |
| [RepoBench](https://arxiv.org/abs/2306.03091) | Repo-level completion (Python + Java), three difficulty levels including cross-file-random. | +15 pts EM on cross-file-random |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | Long-term conversational memory, multi-hop and temporal questions — checks that episodic + semantic layers work together. | Match or beat Mem0 / Zep on multi-hop |

The reported numbers in the README will always be against a **controlled
baseline** using the *exact same underlying LLM* (default:
`qwen2.5-coder:32b` via Ollama), so the delta measures the memory system,
not the model.

See [`benchmarks/`](./benchmarks) for the harness.

---

## Status

v0.1 scaffold — **not ready for production, not ready for benchmarks yet**.

- [x] Repo scaffold
- [x] FastAPI backend
- [x] SQLite storage, SQLAlchemy 2.x async
- [x] Python AST extractor (`defines`, `imports`, `calls`)
- [x] Promise tracking (unresolved references)
- [x] `/ingest/file`, `/ingest/directory`
- [x] `/projects/{id}/graph` (Cytoscape-compatible payload)
- [x] Minimal web frontend: graph viewer
- [x] Chat endpoint with memory-backed context assembly
- [x] Zip / directory ingestion (drag-drop in UI)
- [x] Mobile-first responsive UI (hamburger, keyboard-aware input)
- [x] Project-generation loop (plan → write → sandbox test → ZIP)
- [x] Embedding layer (`nomic-embed-text`) + semantic symbol retrieval
- [x] CrossCodeEval harness (EM + edit similarity, baseline vs synapse)
- [ ] Real CrossCodeEval run (pending dataset download on server)
- [ ] Preferences extractor + idle reflection worker

### Ollama models the server expects

- `qwen2.5-coder:32b` (default chat / generation / extractor)
- `nomic-embed-text` (symbol embeddings — `ollama pull nomic-embed-text`
  once on the host; ~270 MB, runs fine on a 3070 alongside the 32B on
  the 5070)

Changing the embedding model is a one-line change in
`backend/app/config.py` (`embeddings_model`). When you change it, hit
`POST /projects/{id}/reindex` to force a re-embedding of every symbol —
old vectors are replaced in place, no DB migration needed.
- [ ] CrossCodeEval harness + first numbers

---

## Run it

```bash
# Backend only, via Docker
docker compose up --build

# Or locally with uv
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 5555
```

Then open `http://localhost:5555/` for the graph viewer, or hit
`http://localhost:5555/docs` for the OpenAPI UI.

### Ingest your own project

```bash
curl -X POST http://localhost:5555/ingest/directory \
  -H "content-type: application/json" \
  -d '{"project_id": "my-repo", "path": "/absolute/path/to/repo"}'
```

Then GET `http://localhost:5555/projects/my-repo/graph` and watch it
render in the UI.

---

## License

MIT.

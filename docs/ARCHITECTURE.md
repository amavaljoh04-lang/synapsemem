# Architecture

This document is the technical companion to the [README](../README.md). It
describes the data model, the ingestion path, and how promise resolution
actually works.

## The four layers

```
            ┌──────────────────────────────────────────┐
   slow     │ 4. Reflection (worker)                   │  writes back into 2+3
   (minutes)│    merge / decay / contradictions        │
            ├──────────────────────────────────────────┤
            │ 3. Procedural (Preferences)              │  Johnny→uses→argparse
            │    learned rules, confidence-weighted    │
            ├──────────────────────────────────────────┤
            │ 2. Semantic (Files, Symbols, Imports,    │  structured, typed
            │     Promises, free-form Edges)           │
            ├──────────────────────────────────────────┤
   fast     │ 1. Episodic (Episodes)                   │  immutable log
  (ms)      │    every prompt, every file write        │
            └──────────────────────────────────────────┘
```

Episodic is written on every ingest. Semantic is derived *synchronously* for
the well-structured parts (code → AST) and *asynchronously* for the rest
(natural language → LLM extraction, not yet implemented). Procedural is
promoted from the semantic layer by the reflection worker when a pattern
recurs with high confidence.

## Why Promises?

A promise is an unresolved reference. It says: "some file **expects** a
symbol named `parse_config` to live in module `utils`, but right now no
such symbol exists in the project".

Existing systems fall into two camps:

* **Post-hoc linting** (pyflakes, pyright): tells you *after* you've written
  everything that something's broken.
* **Vector RAG** (Aider, Cursor): recalls text but doesn't reason about
  *commitments* — if file A imports `X`, the retriever might surface A but
  won't proactively tell the agent "you owe me X".

Promises make the *commitment* itself a first-class object in the graph.
When the coding agent asks "what do I need to write next?", the answer
includes every open promise directed at the file it's about to touch.

## Ingestion flow (Python)

1. `POST /ingest/file` with `(project_id, path, content)`.
2. `File` row is upserted (content hashed for idempotency).
3. Python source is parsed by `ast_extractor.extract_python`:
   * Top-level `def` / `async def` / `class` → `Symbol` rows.
   * Top-level `Name = ...` and `Name: T = ...` → `Symbol(kind="variable")`.
   * `import x` / `from x import a, b` → `Import` rows (one per name).
4. Relative imports like `from .utils import x` are resolved against the
   source file's own path, so `a/b/c.py` + `from .utils import x` → module
   `a/b/utils`.
5. **Promise resolution pass**: for every symbol freshly written, close any
   open promise whose `(expected_module, expected_name)` matches.
6. **Promise creation pass**: for every import in this file whose target
   module is known in the project (has an existing `File`) but whose target
   name is not yet defined there, create a new open Promise.

Step 5 happens *before* step 6 so a file that both satisfies existing
demand and creates new demand is handled in one pass.

## Why not just re-parse everything?

We could, and at small scale that's fine. The reason we track derived state
(symbols, imports, promises) explicitly is:

* **Queries become cheap and structured**: `GET
  /projects/{id}/promises?unresolved=true` is a single indexed SELECT, not
  a re-parse of N files.
* **Reflection has something to chew on**: decay weights on unused edges,
  merge equivalent symbols, track when a promise was created vs. resolved —
  all of that needs a persistent, typed graph.
* **The coding agent's prompt stays small**: on every task we inject only
  the relevant neighborhood of the graph + the content of files it depends
  on, not the whole repo.

## Next milestones

* **LLM extractor**: parse natural-language prompts into requirements, link
  each requirement to the task(s) that are supposed to satisfy it. Mirrors
  the Promise mechanism but for user intent instead of code.
* **Chat endpoint**: `/chat` that takes a user message, pulls
  memory context (open promises + relevant symbols + recent episodes),
  and answers with the LLM while citing which memory elements it used.
* **Reflection worker**: periodic pass that decays stale edges, merges
  equivalent entities, and promotes recurring patterns into the
  procedural layer.
* **Benchmarks**: CrossCodeEval harness first (closest to our use case),
  then RepoBench, then LongMemEval.

# Benchmarks

The goal of SynapseMem is to **prove** that a structured-memory layer
improves a local LLM on real, widely-used benchmarks — not just to feel
better. This doc tracks which benchmarks we target, how we run them, and
what the current numbers look like.

## Target benchmarks

### 1. CrossCodeEval (primary)

* Paper: [CrossCodeEval: A Diverse and Multilingual Benchmark for
  Cross-File Code Completion](https://arxiv.org/abs/2310.11248) (NeurIPS
  2023).
* What it measures: given a partially-written file with a cursor, predict
  the rest of the line/block. Critically, the correct completion **depends
  on definitions in other files** of the same repo. This is literally the
  use case SynapseMem is built for.
* Metric we track: Exact Match (EM) and Edit Similarity (ES) on the
  cross-file-dependent subset.
* Target: **+20 points EM** over the baseline model (same model, no
  memory) on the Python split.

### 2. RepoBench

* Paper: [RepoBench: Benchmarking Repository-Level Code
  Auto-Completion](https://arxiv.org/abs/2306.03091) (ICLR 2024).
* What it measures: repo-level completion across Python and Java, three
  difficulty levels (in-file, cross-file-first, cross-file-random).
* Metric: EM on the cross-file-random tier, which is the hardest.
* Target: **+15 points EM** over baseline.

### 3. LongMemEval (secondary, for conversational use)

* Paper: [LongMemEval: Benchmarking Chat Assistants on Long-Term
  Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025).
* What it measures: 500 chat sessions with questions that require recalling
  information from earlier turns, including multi-hop and temporal
  queries.
* Why it's secondary: SynapseMem's primary story is code. LongMemEval
  tests the episodic + semantic interaction, which is still useful but
  farther from our core claim.

## Harness

The harness lives in `benchmarks/` and will:

1. Spin up a SynapseMem instance backed by SQLite (in-memory if possible).
2. For each task:
   a. Ingest the repo's other files into SynapseMem.
   b. Query SynapseMem for the context relevant to the cursor file.
   c. Prompt the LLM (via Ollama) with the SynapseMem-augmented context.
   d. Score the completion.
3. Run the same tasks with a **baseline** (same LLM, no memory) and a
   **vanilla RAG baseline** (FAISS over raw file text, no graph).
4. Report all three side by side.

This is explicitly not a "cherry-pick a prompt and declare victory"
setup. If SynapseMem doesn't beat vanilla RAG, it's not worth keeping.

## Current numbers

_TBD — the harness isn't wired up yet. This table will be populated in the
next milestone and kept current. The model column names the exact Ollama
tag so everyone can reproduce._

| Benchmark           | Model                    | Baseline EM | +RAG EM | +SynapseMem EM |
|---------------------|--------------------------|-------------|---------|----------------|
| CrossCodeEval (Py)  | qwen2.5-coder:32b        | TBD         | TBD     | TBD            |
| RepoBench (Py, XF)  | qwen2.5-coder:32b        | TBD         | TBD     | TBD            |
| LongMemEval         | qwen2.5-coder:32b        | TBD         | TBD     | TBD            |

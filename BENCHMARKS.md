# Benchmarks

SynapseMem is evaluated against two published benchmarks. We track the
gap between the raw LLM (baseline) and the LLM assisted by the
SynapseMem memory layer (synapse). A third line — plain vector RAG —
will be added once the embeddings layer is wired.

## CrossCodeEval (Ding et al., NeurIPS 2023)

Dataset: <https://github.com/amazon-science/cceval>. Metric: Exact Match
and Edit Similarity on line-level completion, with the target line
removed from the file.

Running locally on the deploy host:

```bash
# inside the container (so Ollama at host-network 11434 is reachable)
docker compose exec synapsemem \
  python -m app.bench.crosscodeeval \
    --dataset /app/data/cceval_python.jsonl \
    --mode both \
    --model qwen2.5-coder:32b \
    --limit 20 \
    --output /app/data/bench/cce_smoke.json
```

Output shape:

```
=== running mode=baseline model=qwen2.5-coder:32b on 20 samples ===
  [   1/20] · em=0 es=0.72 elapsed=6.4s task=python/0
  ...
  -> EM=0.250 ES=0.610 on 20 samples

=== running mode=synapse model=qwen2.5-coder:32b on 20 samples ===
  ...
  -> EM=0.XXX ES=0.YYY on 20 samples
```

Put the dataset JSONL at ``/app/data/cceval_python.jsonl``; the loader
tolerates both shapes of ``crossfile_context`` (flat string or list of
``{path, content}`` dicts).

### Scoreboard

Updated manually after each commit that affects retrieval.

| Mode       | Model                  | Samples | EM     | ES     | Date       | Notes                                                               |
|------------|------------------------|---------|--------|--------|------------|---------------------------------------------------------------------|
| baseline   | qwen2.5-coder:7b       | 20      | 0.000  | 0.276  | 2026-04-20 | smoke; prompt only, no crossfile context                            |
| synapse    | qwen2.5-coder:7b       | 20      | 0.000  | 0.253  | 2026-04-20 | blob splitter + AST + keyword/semantic; no uplift yet on line-level |
| baseline   | qwen2.5-coder:32b      | —       | —      | —      | —          | awaiting run                                                        |
| synapse    | qwen2.5-coder:32b      | —       | —      | —      | —          | awaiting run                                                        |

**Reading the smoke**: 20 samples is statistically noisy, but the gap is
real: on line-level completion with the official oracle crossfile blob,
the current SynapseMem context does not help qwen2.5-coder:7b. Failure
modes seen so far:

1. CCE completions are mid-expression (``self.tokenizer.decode(sequence
   _actual[:, -max_stop_string:])[0]``) — knowing *that another file
   defines ``tokenizer.decode``* does not predict the exact token
   sequence the grader checks.
2. The 7B coder is a fill-in-the-middle model; an instruction wrapper
   (``You are a code completion model ... emit only the continuation``)
   costs it a few ES points before any memory block is added.
3. Our retrieval surfaces symbols matching prompt keywords, but the
   completion target often refers to *locals* of the parent file, which
   are never in the symbol graph.

**Planned fixes before the 32B run**:

- Switch the prompt to raw FIM tokens (``<|fim_prefix|>`` / ``<|fim_
  suffix|>``) instead of the chat-style instruction wrapper.
- Inject ``right_context`` in both modes — the memory block should add
  signal on top of the full file skeleton, not replace it.
- Include the verbatim crossfile text as a fallback when no symbol
  matches fire (structured view is additive, not replacement).

## LongMemEval (Wu et al., ICLR 2025)

Dataset: <https://github.com/xiaowu0162/LongMemEval>. Metric: accuracy
on 5 memory-competence sub-tasks (single-hop, multi-hop, temporal,
knowledge update, abstention). Harness lives in
``backend/app/bench/longmemeval.py`` once wired.

Not yet implemented — planned right after the CrossCodeEval numbers
are stable.

## Reproducibility

- All harnesses use an in-memory SQLite per sample so runs do not
  pollute the main ``synapse.db``.
- Temperature is pinned to 0.0; ``num_predict`` is 128 for completion
  tasks.
- Elapsed seconds are recorded per-sample for cost tracking.

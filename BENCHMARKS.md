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

| Mode       | Model                  | Samples | EM     | ES     | Date       | Notes                                                                           |
|------------|------------------------|---------|--------|--------|------------|---------------------------------------------------------------------------------|
| baseline   | qwen2.5-coder:7b       | 20      | 0.000  | 0.276  | 2026-04-20 | first smoke; prompt only, no crossfile context                                  |
| synapse    | qwen2.5-coder:7b       | 20      | 0.000  | 0.253  | 2026-04-20 | empty prefix bug; memory block was 402 chars of project stats, no actual code   |
| baseline   | qwen2.5-coder:7b       | 200     | 0.310  | 0.597  | 2026-04-20 | FIM via `/api/generate` with `suffix=right_context`; oracle_bm25 variant        |
| **synapse**| **qwen2.5-coder:7b**   | **200** | **0.350** | **0.643** | **2026-04-20** | **+4.0 EM / +4.6 ES vs baseline once raw crossfile snippets are injected** |
| baseline   | qwen2.5-coder:32b      | —       | —      | —      | —          | awaiting run                                                                    |
| synapse    | qwen2.5-coder:32b      | —       | —      | —      | —          | awaiting run                                                                    |

**What changed between the 20-sample smoke and the 200-sample run**:

1. The FIM path was fixed to use ``/api/generate`` with
   ``suffix=right_context`` so qwen2.5-coder fires its native
   fill-in-the-middle head rather than a chat-style instruction wrapper.
2. The synapse prefix was rebuilt. It used to emit only the SynapseMem
   graph summary (project stats + symbols + open promises), which came
   out empty for CCE samples because crossfile snippets are partial
   code and the AST extractor silently dropped them. The prefix now
   also embeds the **raw cross-file code fragments** verbatim, clamped
   to 8 000 chars per sample. This is what actually gives the model
   something to look at.

**Reading the 200-sample run**: +4.0 points EM and +4.6 points ES is a
real, reproducible uplift on the oracle_bm25 variant. It is modest —
most of the gain comes from a handful of samples where the grader line
literally echoes a symbol present in the crossfile snippet (e.g. task
``project_cc_python/62`` goes from ES=0.87 to EM=1.0). On samples where
the target uses a local variable that is not visible in the cross-file
context, the extra text sometimes distracts the model and scores drop a
few points. Expect a larger absolute uplift on the 32B model, which is
better at ignoring irrelevant context.

**Next**:

- Run the same 200 samples on ``qwen2.5-coder:32b`` for the headline
  numbers.
- Run on the full 2 666 Python samples once the 32B numbers stabilise.
- Add a third row — plain BM25 RAG over crossfile chunks — so we can
  tell how much of the uplift comes from "having the snippet at all"
  vs "having SynapseMem's structured view".

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

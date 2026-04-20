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

| Mode       | Model                  | Samples | EM     | ES     | Date       | Notes                 |
|------------|------------------------|---------|--------|--------|------------|-----------------------|
| baseline   | qwen2.5-coder:32b      | TBD     | —      | —      | —          | awaiting live run     |
| synapse    | qwen2.5-coder:32b      | TBD     | —      | —      | —          | keyword retrieval     |
| synapse+emb| qwen2.5-coder:32b      | TBD     | —      | —      | —          | nomic-embed-text      |

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

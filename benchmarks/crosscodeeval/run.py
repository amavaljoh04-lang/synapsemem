"""CLI entrypoint: ``python -m benchmarks.crosscodeeval.run``.

Examples
--------
Smoke run, 5 Python tasks, baseline only::

    python -m benchmarks.crosscodeeval.run \\
        --data-dir data/cceval --language python --limit 5 \\
        --configs baseline --model qwen2.5-coder:7b

Full comparison (baseline vs RAG vs SynapseMem) on 200 tasks::

    python -m benchmarks.crosscodeeval.run \\
        --data-dir data/cceval --language python --limit 200 \\
        --configs baseline,rag_bm25,synapsemem \\
        --model qwen2.5-coder:32b --output-dir runs/2026-04-19
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict
from pathlib import Path

from .dataset import iter_tasks
from .retrievers import BM25Retriever, NoopRetriever, SynapseMemRetriever
from .runner import AggregateResult, run_config

_RETRIEVERS = {
    "baseline": NoopRetriever,
    "rag_bm25": BM25Retriever,
    "synapsemem": SynapseMemRetriever,
}


def _ollama_complete_factory(base_url: str, model: str):
    from app.ollama_client import OllamaClient

    client = OllamaClient(base_url)

    def complete(prompt: str) -> str:
        async def _call() -> str:
            resp = await client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 256},
            )
            return (resp.get("message") or {}).get("content", "")

        return asyncio.run(_call())

    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossCodeEval runner for SynapseMem")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--configs",
        default="baseline,rag_bm25,synapsemem",
        help="Comma-separated list of retrievers to run.",
    )
    parser.add_argument("--model", default="qwen2.5-coder:32b")
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("SYNAPSEMEM_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--max-context-chars", type=int, default=8000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = _ollama_complete_factory(args.ollama_base_url, args.model)

    aggregates: list[AggregateResult] = []
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if config not in _RETRIEVERS:
            raise SystemExit(f"Unknown retriever: {config}")
        # Each config sees its own fresh iterator (reads JSONL twice — fine).
        tasks = list(
            iter_tasks(args.data_dir, args.language, limit=args.limit)
        )
        retriever = _RETRIEVERS[config]()
        out = args.output_dir / f"{config}.jsonl"
        print(f"→ running {config} ({len(tasks)} tasks) → {out}")
        agg = run_config(
            tasks,
            retriever,
            complete_fn=complete,
            out_path=out,
            max_context_chars=args.max_context_chars,
        )
        aggregates.append(agg)
        print(
            f"  {config:12s}  EM={agg.em*100:5.2f}%  ES={agg.es*100:5.2f}%  "
            f"n={agg.n}"
        )

    summary = args.output_dir / "summary.json"
    summary.write_text(
        "\n".join(
            [
                "# CrossCodeEval — SynapseMem benchmark",
                f"model: {args.model}",
                f"language: {args.language}",
                f"n: {aggregates[0].n if aggregates else 0}",
                "",
                *[str(asdict(a)) for a in aggregates],
            ]
        )
    )
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

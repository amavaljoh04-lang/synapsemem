"""CrossCodeEval harness for SynapseMem.

NeurIPS 2023 benchmark for *cross-file* code completion:
https://arxiv.org/abs/2310.11248 — repo at github.com/amazon-science/cceval.

For each task the model sees the left context of the cursor inside one
file and must predict the next line. The completion is only possible if
the model knows about symbols defined in *other* files of the repo —
which is exactly the problem SynapseMem is built to solve.

This harness supports three configurations so we can measure uplift:

* ``baseline`` — raw model call, no cross-file info.
* ``rag_bm25`` — BM25 over all other files in the repo, top-k chunks
  prepended to the prompt.
* ``synapsemem`` — ingest the whole repo into SynapseMem, retrieve the
  memory context for the target file/line, prepend it.

Metrics follow the paper: Exact Match (line-level) and Edit Similarity.
"""

"""Benchmark harnesses for SynapseMem.

The harnesses are isolated from the live FastAPI service: they spin up
a throw-away SQLite DB, ingest the dataset's cross-file context into
SynapseMem, then compare model predictions with and without the
memory-augmented prompt using the official metrics from each
benchmark's paper.
"""

"""Unit tests for CrossCodeEval metrics and helper extractors."""

from __future__ import annotations

import json
from pathlib import Path

from app.bench.crosscodeeval import (
    _extract_crossfile_files,
    extract_completion,
    load_samples,
)
from app.bench.metrics import (
    aggregate,
    edit_similarity,
    exact_match,
)


def test_exact_match_ignores_trailing_whitespace() -> None:
    assert exact_match("def f():\n    return 1  ", "def f():\n    return 1")
    assert not exact_match("def f():\n    return 2", "def f():\n    return 1")


def test_edit_similarity_bounds() -> None:
    assert edit_similarity("abc", "abc") == 1.0
    assert 0.0 < edit_similarity("abcd", "abce") < 1.0
    assert edit_similarity("", "") == 1.0


def test_aggregate_computes_mean() -> None:
    rows = [
        {"em": True, "es": 1.0},
        {"em": False, "es": 0.5},
        {"em": False, "es": 0.0},
    ]
    summary = aggregate(rows)
    assert summary["n"] == 3
    assert summary["em"] == 1 / 3
    assert abs(summary["es"] - 0.5) < 1e-9


def test_extract_completion_line_limit() -> None:
    # Model emitted 5 lines, groundtruth had 2 → only 2 kept.
    reply = "line1\nline2\nline3\nline4\nline5"
    gt = "line1\nline2"
    assert extract_completion(reply, gt) == "line1\nline2"


def test_extract_completion_strips_fenced_block() -> None:
    reply = "sure thing:\n```python\nprint('hi')\n```\n"
    gt = "print('hi')"
    assert extract_completion(reply, gt) == "print('hi')"


def test_crossfile_normalisation() -> None:
    # list of dicts with filename key
    raw = [{"filename": "a.py", "content": "A"}, {"path": "b.py", "text": "B"}]
    out = _extract_crossfile_files(raw)
    assert out == [("a.py", "A"), ("b.py", "B")]
    # flat string fallback
    assert _extract_crossfile_files("raw chunk") == [("crossfile.txt", "raw chunk")]
    assert _extract_crossfile_files(None) == []


def test_load_samples_reads_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "cce.jsonl"
    rec = {
        "task_id": "python/0",
        "language": "python",
        "prompt": "def add(a,b):\n    ",
        "groundtruth": "return a+b\n",
        "right_context": "",
        "crossfile_context": [{"path": "utils.py", "content": "def helper():\n    pass\n"}],
    }
    p.write_text(json.dumps(rec) + "\n")
    samples = load_samples(p)
    assert len(samples) == 1
    assert samples[0].task_id == "python/0"
    assert samples[0].crossfile_files == [("utils.py", "def helper():\n    pass\n")]

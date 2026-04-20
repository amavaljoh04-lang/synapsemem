"""Metrics used by the CrossCodeEval paper.

* **Exact Match (EM)** — 1.0 if the first predicted line matches the
  groundtruth line exactly (after stripping trailing whitespace), else
  0.0. This is the headline number in the paper.
* **Edit Similarity (ES)** — ``1 - edit_distance(pred, gt) / max(len(pred),
  len(gt), 1)``. Range [0, 1]. Reported alongside EM to catch near
  misses.
"""

from __future__ import annotations


def exact_match(prediction: str, groundtruth: str) -> float:
    pred = _first_line(prediction).rstrip()
    gt = _first_line(groundtruth).rstrip()
    return 1.0 if pred == gt else 0.0


def edit_similarity(prediction: str, groundtruth: str) -> float:
    pred = _first_line(prediction).rstrip()
    gt = _first_line(groundtruth).rstrip()
    if not pred and not gt:
        return 1.0
    dist = _levenshtein(pred, gt)
    return 1.0 - dist / max(len(pred), len(gt), 1)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]

"""Exact Match and Edit Similarity — the two metrics used by the
CrossCodeEval paper (Ding et al. 2023, NeurIPS) and re-used by
RepoBench. Both operate on single-line or multi-line completions.

We implement them locally (no external deps) so harness runs stay
hermetic and reproducible.
"""

from __future__ import annotations


def _normalise(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def exact_match(prediction: str, reference: str) -> bool:
    """True when the prediction equals the reference after trimming
    trailing whitespace on each line and stripping outer whitespace."""
    return _normalise(prediction) == _normalise(reference)


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
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def edit_similarity(prediction: str, reference: str) -> float:
    """1 - levenshtein / max_len, as a percentage 0..1.

    Matches the formula used by CrossCodeEval (``fuzzywuzzy.ratio`` / 100)
    up to tie-breaking on equal-length strings.
    """
    a = _normalise(prediction)
    b = _normalise(reference)
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / denom


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Reduce a list of per-sample ``{em, es}`` dicts to mean scores."""
    if not rows:
        return {"em": 0.0, "es": 0.0, "n": 0}
    em = sum(1 for r in rows if r["em"]) / len(rows)
    es = sum(r["es"] for r in rows) / len(rows)
    return {"em": em, "es": es, "n": len(rows)}

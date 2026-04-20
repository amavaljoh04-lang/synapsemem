"""Tests for the CrossCodeEval oracle/rg1 blob splitter.

The official release ships ``crossfile_context`` as a comment-annotated
blob. Without splitting it into real files, the AST extractor sees only
comments and SynapseMem's structured memory stays empty. These tests
lock the splitter shape so regressions there are caught before a
benchmark run.
"""

from __future__ import annotations

from app.bench.crosscodeeval import _split_cceval_blob

BLOB = """# Here are some relevant code fragments from other files of the repo:

# the below code fragment can be found in:
# alt_generator.py
# def helper():
#     return 42
#
# class Foo:
#     pass

# the below code fragment can be found in:
# utils.py
# def parse_config(path):
#     return {}

# the below code fragment can be found in:
# alt_generator.py
# def another_helper():
#     return helper() + 1
"""


def test_split_cceval_blob_groups_by_filename() -> None:
    out = dict(_split_cceval_blob(BLOB))
    assert set(out) == {"alt_generator.py", "utils.py"}


def test_split_cceval_blob_concatenates_repeated_files() -> None:
    out = dict(_split_cceval_blob(BLOB))
    assert "def helper()" in out["alt_generator.py"]
    assert "def another_helper()" in out["alt_generator.py"]
    assert "def parse_config(path)" in out["utils.py"]


def test_split_cceval_blob_strips_comment_prefix() -> None:
    out = dict(_split_cceval_blob(BLOB))
    for body in out.values():
        for line in body.splitlines():
            assert not line.startswith("# "), (
                f"leftover comment prefix in fragment body: {line!r}"
            )


def test_split_cceval_blob_empty_and_headerless() -> None:
    assert _split_cceval_blob("") == []
    assert _split_cceval_blob("# random\n# comment\n") == []

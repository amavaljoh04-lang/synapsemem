"""Unit tests for the LLM-response parser and project zip helper."""

from __future__ import annotations

import zipfile

from app.generator import (
    make_project_zip,
    parse_llm_response,
    project_workdir,
    syntax_check,
)


def test_parse_triple_equal_framing() -> None:
    text = """Some plan text.

=== FILE: app/main.py ===
```python
def main():
    return 1
```

=== FILE: app/utils.py ===
```
VALUE = 42
```
Trailing note.
"""
    out = parse_llm_response(text)
    paths = [f.path for f in out.files]
    assert paths == ["app/main.py", "app/utils.py"]
    assert "def main()" in out.files[0].content
    assert "VALUE = 42" in out.files[1].content
    assert "Some plan text" in out.prose


def test_parse_hash_file_framing_fallback() -> None:
    text = """# FILE: helpers.py
```python
def helper():
    pass
```
"""
    out = parse_llm_response(text)
    assert len(out.files) == 1
    assert out.files[0].path == "helpers.py"


def test_parse_rejects_absolute_and_traversal_paths() -> None:
    text = """=== FILE: /etc/passwd ===
```
nope
```
=== FILE: ../escape.py ===
```
nope
```
=== FILE: ok.py ===
```
ok = 1
```
"""
    out = parse_llm_response(text)
    assert [f.path for f in out.files] == ["ok.py"]


def test_syntax_check_detects_syntax_error() -> None:
    checks = syntax_check(
        [
            ("ok.py", "x = 1\n"),
            ("bad.py", "def x(:\n"),
            ("readme.md", "not python"),
        ]
    )
    assert checks[0].compile_ok is True
    assert checks[1].compile_ok is False
    assert "SyntaxError" in checks[1].compile_error
    assert checks[2].compile_ok is True


def test_make_project_zip_roundtrip(isolated_data_dir) -> None:
    # ``project_workdir`` creates the directory; we then drop a couple of
    # files and verify the zip contains them at the right paths.
    pid = "zipme"
    root = project_workdir(pid)
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("A = 1\n")
    (root / "README.md").write_text("hi\n")
    out = make_project_zip(pid)
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    assert names == ["README.md", "pkg/a.py"]

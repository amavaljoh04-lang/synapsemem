"""Unit tests for the Python AST extractor."""

from __future__ import annotations

import textwrap

from app.ast_extractor import extract_python


def test_extracts_class_function_variable() -> None:
    src = textwrap.dedent(
        '''
        """Module docstring."""
        from typing import List

        CONST = 42

        class Foo(Base):
            """Foo does things."""
            def method(self, x: int) -> str:
                return str(x)

        def free_fn(a, b=2):
            return a + b

        async def fetch(url: str) -> bytes:
            return b""
        '''
    )
    r = extract_python(src)
    assert r.parse_error == ""

    names = {s.name: s for s in r.symbols}
    assert set(names) == {"CONST", "Foo", "free_fn", "fetch"}

    assert names["Foo"].kind == "class"
    assert "class Foo(Base)" in names["Foo"].signature
    assert names["Foo"].docstring == "Foo does things."

    assert names["free_fn"].kind == "function"
    assert "def free_fn" in names["free_fn"].signature

    assert names["fetch"].kind == "async_function"
    assert "async def fetch" in names["fetch"].signature
    assert "-> bytes" in names["fetch"].signature

    assert names["CONST"].kind == "variable"


def test_extracts_imports_with_alias() -> None:
    src = textwrap.dedent(
        """
        import os
        import numpy as np
        from pathlib import Path
        from .utils import parse_config, dump
        from ..pkg.sub import thing
        """
    )
    r = extract_python(src)
    tuples = [(imp.module, imp.name, imp.alias) for imp in r.imports]
    assert ("os", "os", "") in tuples
    assert ("numpy", "numpy", "np") in tuples
    assert ("pathlib", "Path", "") in tuples
    assert (".utils", "parse_config", "") in tuples
    assert (".utils", "dump", "") in tuples
    assert ("..pkg.sub", "thing", "") in tuples


def test_syntax_error_is_captured_not_raised() -> None:
    r = extract_python("def broken(:\n")
    assert r.parse_error != ""
    assert r.symbols == []
    assert r.imports == []


def test_annotated_module_variable_captured() -> None:
    r = extract_python("MAX_RETRIES: int = 3\n")
    assert len(r.symbols) == 1
    assert r.symbols[0].name == "MAX_RETRIES"
    assert r.symbols[0].kind == "variable"

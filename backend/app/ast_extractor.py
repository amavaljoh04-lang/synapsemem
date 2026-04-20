"""Python AST → structured facts about a file.

We extract four things from a Python source file:

* **Top-level definitions** — classes, functions (sync + async), and
  module-level variable assignments. These become ``Symbol`` rows.
* **Imports** — both ``import x`` and ``from x import a, b``, preserving
  line numbers and aliases.
* **Docstrings and signatures** — compact one-line repr of each definition.
* **Module-relative resolution hints** — for ``from .utils import foo``,
  the module is recorded as a relative path so the caller can resolve it
  against the file's own location.

Anything we can't parse (e.g. file with a ``SyntaxError``) yields an empty
result plus a flag — we don't crash ingestion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ExtractedSymbol:
    name: str
    kind: str
    signature: str
    docstring: str
    line_start: int
    line_end: int


@dataclass
class ExtractedImport:
    module: str
    name: str
    alias: str
    line: int


@dataclass
class ExtractionResult:
    symbols: list[ExtractedSymbol] = field(default_factory=list)
    imports: list[ExtractedImport] = field(default_factory=list)
    parse_error: str = ""


def _signature_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Compact, human-readable signature — enough to tell call sites what to expect."""
    try:
        args_src = ast.unparse(node.args)
    except AttributeError:  # pragma: no cover - 3.8 fallback
        args_src = ", ".join(a.arg for a in node.args.args)
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:
            returns = ""
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{keyword} {node.name}({args_src}){returns}"


def _signature_from_class(node: ast.ClassDef) -> str:
    bases: list[str] = []
    for b in node.bases:
        try:
            bases.append(ast.unparse(b))
        except Exception:
            bases.append("<base>")
    suffix = f"({', '.join(bases)})" if bases else ""
    return f"class {node.name}{suffix}"


def _top_level_assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)
    else:
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
    return names


def _docstring_of(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) if hasattr(ast, "get_docstring") else None
    if not doc:
        return ""
    first_paragraph = doc.split("\n\n", 1)[0].strip()
    return first_paragraph[:500]


def extract_python(source: str) -> ExtractionResult:
    """Parse ``source`` and return an :class:`ExtractionResult`.

    On syntax error, return an empty result with ``parse_error`` populated
    — the caller can still record the file, just without structure.
    """
    result = ExtractionResult()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.parse_error = f"{exc.msg} (line {exc.lineno})"
        return result

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result.symbols.append(
                ExtractedSymbol(
                    name=node.name,
                    kind="class",
                    signature=_signature_from_class(node),
                    docstring=_docstring_of(node),
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
                )
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            kind = (
                "async_function"
                if isinstance(node, ast.AsyncFunctionDef)
                else "function"
            )
            result.symbols.append(
                ExtractedSymbol(
                    name=node.name,
                    kind=kind,
                    signature=_signature_from_function(node),
                    docstring=_docstring_of(node),
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
                )
            )
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            for name in _top_level_assignment_names(node):
                result.symbols.append(
                    ExtractedSymbol(
                        name=name,
                        kind="variable",
                        signature=name,
                        docstring="",
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
                    )
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result.imports.append(
                    ExtractedImport(
                        module=alias.name,
                        name=alias.name.split(".")[0],
                        alias=alias.asname or "",
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            # Relative imports keep their dots so the caller can resolve against the file path.
            module = ("." * (node.level or 0)) + (node.module or "")
            for alias in node.names:
                result.imports.append(
                    ExtractedImport(
                        module=module,
                        name=alias.name,
                        alias=alias.asname or "",
                        line=node.lineno,
                    )
                )

    return result

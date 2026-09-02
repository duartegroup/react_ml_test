"""
The ``rmlt`` command: run react_ml_test_public workflow scripts.

Installing this package (``pip install -e .``) provides a single ``rmlt``
executable, which dispatches to the workflow modules in
``react_ml_test_public.scripts``:

    rmlt list                          # list available scripts
    rmlt run-test-set-comparison       # run one (dashes == underscores)
    rmlt run-test-set-comparison ...   # anything after the name is passed
                                       # through to the script

The scripts read ``data/``, ``models/``, ``inputs/`` and write to ``out/``
relative to the current working directory, so run them from the repository
root (or any directory containing those folders).
"""

from __future__ import annotations

import ast
import importlib
import os
import runpy
import sys

_SCRIPTS_PKG = "react_ml_test_public.scripts"


def _scripts_dir() -> str:
    return os.path.dirname(importlib.import_module(_SCRIPTS_PKG).__file__)


def _script_names() -> list[str]:
    """Sorted module names of the runnable scripts (no __init__/private modules)."""
    names = []
    for entry in sorted(os.listdir(_scripts_dir())):
        if entry.startswith("_") or not entry.endswith(".py"):
            continue
        names.append(entry[:-3])
    return names


def _doc_summary(name: str) -> str:
    """First line of a script's module docstring, read without importing it."""
    try:
        with open(os.path.join(_scripts_dir(), name + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        doc = ast.get_docstring(tree)
    except (OSError, SyntaxError, ValueError):
        return ""
    return doc.strip().splitlines()[0].strip() if doc else ""


def _token(name: str) -> str:
    return name.replace("_", "-")


def _module_name(token: str) -> str | None:
    """Map a command token (dashes or underscores) back to a module name."""
    token = token.strip()
    if token.endswith(".py"):
        token = token[:-3]
    key = token.replace("-", "_")
    return key if key in _script_names() else None


def _print_list(stream=sys.stdout) -> None:
    print("Usage: rmlt <script> [args...]", file=stream)
    print("       rmlt list", file=stream)
    print(file=stream)
    print("Available scripts:", file=stream)
    names = _script_names()
    tokens = [_token(name) for name in names]
    width = max(len(token) for token in tokens) if tokens else 0
    for name, token in zip(names, tokens):
        summary = _doc_summary(name)
        print(f"  {token:<{width}}  {summary}" if summary else f"  {token}", file=stream)


def main(argv: list[str] | None = None) -> int | None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        _print_list()
        return 0

    if args[0] in ("list", "-l", "--list"):
        _print_list()
        return 0

    name = _module_name(args[0])
    if name is None:
        print(f"rmlt: unknown script '{args[0]}'", file=sys.stderr)
        print("Run 'rmlt list' to see the available scripts.", file=sys.stderr)
        return 2

    # runpy leaves sys.argv alone, so reshape it to look like a direct launch:
    # argv[0] stays (the rmlt command) and everything after the script name is
    # passed through untouched.
    sys.argv = [sys.argv[0] if sys.argv else "rmlt"] + args[1:]

    runpy.run_module(f"{_SCRIPTS_PKG}.{name}", run_name="__main__")
    return None

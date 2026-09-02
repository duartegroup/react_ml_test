#!/usr/bin/env python
"""Launcher for react_ml_test_public.scripts.<name>.

Preserves the historical invocation from the repository root:

    python scripts/<name>.py

(equivalent to `python -m react_ml_test_public.scripts.<name>`).
"""
import os
import runpy
import sys

# Make the package importable from a repo checkout even when it is not installed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MODULE = os.path.splitext(os.path.basename(__file__))[0]
runpy.run_module(f"react_ml_test_public.scripts.{_MODULE}", run_name="__main__")

#!/usr/bin/env python3
"""Shim. The implementation lives in src/nepal/reference.py so that it is
importable from the installed package and reachable as `nepal fetch-reference`,
which uses the interpreter the package was installed into.

Prefer:  nepal fetch-reference
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepal.reference import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

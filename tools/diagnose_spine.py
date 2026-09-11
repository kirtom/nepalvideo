#!/usr/bin/env python3
"""Shim. The implementation lives in src/nepal/diagnose.py.

Prefer:  nepal diagnose
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepal.diagnose import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

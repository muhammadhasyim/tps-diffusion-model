"""Test setup: make the ``boltz`` submodule importable when present (no separate pip install)."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_boltz_src = _root / "boltz" / "src"
if _boltz_src.is_dir():
    sys.path.insert(0, str(_boltz_src))

"""Boltz artifact locations (no heavy backend imports)."""

from __future__ import annotations

import os
from pathlib import Path


def default_boltz_cache_dir() -> Path:
    """Directory for Boltz checkpoints and ``mols/`` cache (same layout as ``~/.boltz``).

    Resolution order:

    1. ``BOLTZ_CACHE`` if set (non-empty).
    2. ``$SCRATCH/.boltz`` if ``SCRATCH`` is set (common on HPC).
    3. ``~/.boltz`` otherwise.

    Pass an explicit ``--cache`` in CLIs to override without environment variables.
    """
    env = (os.environ.get("BOLTZ_CACHE") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    scratch = (os.environ.get("SCRATCH") or "").strip()
    if scratch:
        return (Path(scratch).expanduser() / ".boltz").resolve()
    return Path.home() / ".boltz"

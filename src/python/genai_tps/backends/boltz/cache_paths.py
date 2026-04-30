"""Boltz artifact locations (no heavy backend imports)."""

from __future__ import annotations

import os
from pathlib import Path


def default_boltz_cache_dir() -> Path:
    """Directory for Boltz checkpoints and ``mols/`` cache (Boltz-style layout).

    Resolution order:

    1. ``BOLTZ_CACHE`` if set (non-empty).
    2. ``$SCRATCH/.boltz`` if ``SCRATCH`` is set (non-empty).

    There is **no** home-directory fallback. Set ``BOLTZ_CACHE`` or ``SCRATCH``,
    or pass ``--cache`` on CLIs that support it.

    Raises
    ------
    RuntimeError
        If neither ``BOLTZ_CACHE`` nor ``SCRATCH`` yields a directory.

    Returns
    -------
    pathlib.Path
        Absolute expanded path to the cache root.
    """
    env = (os.environ.get("BOLTZ_CACHE") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    scratch = (os.environ.get("SCRATCH") or "").strip()
    if scratch:
        return (Path(scratch).expanduser() / ".boltz").resolve()
    raise RuntimeError(
        "Boltz cache directory is undefined: set environment variable BOLTZ_CACHE "
        "or SCRATCH (no home-directory fallback), or pass --cache explicitly."
    )

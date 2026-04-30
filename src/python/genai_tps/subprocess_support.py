"""Environment helpers for spawning other repo ``scripts/*.py`` processes.

``sys.path`` edits in the parent interpreter are not inherited by child
processes.  Campaign drivers therefore pass ``PYTHONPATH`` including
``<repo>/src/python`` so children can ``import genai_tps`` even before (or
without) an editable ``pip install -e .`` in that environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root (directory that contains ``src/python/genai_tps``)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "src" / "python" / "genai_tps").is_dir():
            return p
    raise RuntimeError(
        f"Could not locate genai-tps repository root starting from {here}"
    )


def child_env_with_repo_src_python(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy *base* or ``os.environ``, then prepend ``<repo>/src/python`` to ``PYTHONPATH``."""
    env = os.environ.copy() if base is None else dict(base)
    src = str(repository_root() / "src" / "python")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + os.pathsep + prev if prev else src
    return env

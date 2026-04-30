"""Optional NVTX ranges for Nsight Systems timelines (CUDA + Python correlation).

Enable with environment variable ``GENAI_TPS_NVTX`` set to ``1``, ``true``, or
``yes`` (case-insensitive). When disabled or when the ``nvtx`` package is
missing, :func:`nvtx_range` is a no-op context manager with negligible overhead.

Install the marker library with the project extra ``profiling`` (``pip install
-e .[profiling]``) or ``conda install nvtx``.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_nvtx_mod: Any | None = False  # False = not tried yet; None = unavailable


def _env_nvtx_enabled() -> bool:
    v = os.environ.get("GENAI_TPS_NVTX", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _load_nvtx() -> Any | None:
    global _nvtx_mod
    if _nvtx_mod is not False:
        return _nvtx_mod
    try:
        import nvtx as mod  # type: ignore[import-untyped]

        _nvtx_mod = mod
        return mod
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning(
            "GENAI_TPS_NVTX is set but `nvtx` could not be imported (%s); "
            "NVTX ranges disabled for this process.",
            exc,
        )
        _nvtx_mod = None
        return None


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Enter an NVTX named range when profiling is enabled; otherwise no-op."""
    if not _env_nvtx_enabled():
        yield
        return
    mod = _load_nvtx()
    if mod is None:
        yield
        return
    annotate = getattr(mod, "annotate", None)
    if callable(annotate):
        with annotate(message):
            yield
        return
    if hasattr(mod, "push_range") and hasattr(mod, "pop_range"):
        mod.push_range(message)
        try:
            yield
        finally:
            mod.pop_range()
        return
    if hasattr(mod, "range_push") and hasattr(mod, "range_pop"):
        mod.range_push(message)
        try:
            yield
        finally:
            mod.range_pop()
        return
    logger.warning(
        "nvtx module has no annotate/push_range/range_push API; "
        "ignoring NVTX range %r.",
        message,
    )
    yield


__all__ = ["nvtx_range"]

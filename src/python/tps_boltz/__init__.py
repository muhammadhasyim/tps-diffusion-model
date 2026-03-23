"""Compatibility shim for the old ``tps_boltz`` package name. Use ``genai_tps.backends.boltz`` instead."""

from __future__ import annotations

import warnings
from typing import Any

warnings.warn(
    "The package name `tps_boltz` is deprecated; import from `genai_tps.backends.boltz` "
    "or `genai_tps` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BoltzDiffusionEngine",
    "BoltzSamplerCore",
    "BoltzSnapshot",
    "boltz_snapshot_descriptor",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from genai_tps.backends import boltz as _boltz_mod

        return getattr(_boltz_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

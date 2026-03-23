"""TPS integration for Boltz-2 reverse diffusion (OpenPathSampling + GPU core)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "BoltzDiffusionEngine",
    "BoltzSamplerCore",
    "BoltzSnapshot",
    "boltz_snapshot_descriptor",
]


def __getattr__(name: str) -> Any:
    if name == "BoltzDiffusionEngine":
        from genai_tps.backends.boltz.engine import BoltzDiffusionEngine

        return BoltzDiffusionEngine
    if name == "BoltzSamplerCore":
        from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore

        return BoltzSamplerCore
    if name == "BoltzSnapshot":
        from genai_tps.backends.boltz.snapshot import BoltzSnapshot

        return BoltzSnapshot
    if name == "boltz_snapshot_descriptor":
        from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor

        return boltz_snapshot_descriptor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor

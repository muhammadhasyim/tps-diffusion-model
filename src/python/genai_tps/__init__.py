"""genai-tps: path sampling for generative-model trajectories.

Vendored OpenPathSampling drives ensembles and moves; optional backends (e.g.
Boltz-2) implement :class:`openpathsampling.engines.DynamicsEngine`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "paths",
    "BoltzDiffusionEngine",
    "BoltzSamplerCore",
    "BoltzSnapshot",
    "boltz_snapshot_descriptor",
    "DynamicsEngine",
    "Trajectory",
    "FixedLengthTPSNetwork",
]


def __getattr__(name: str) -> Any:
    if name == "paths":
        import openpathsampling as paths

        return paths
    if name in (
        "BoltzDiffusionEngine",
        "BoltzSamplerCore",
        "BoltzSnapshot",
        "boltz_snapshot_descriptor",
    ):
        from genai_tps.backends import boltz as _boltz_backend

        return getattr(_boltz_backend, name)
    if name == "DynamicsEngine":
        from openpathsampling.engines import DynamicsEngine

        return DynamicsEngine
    if name == "Trajectory":
        from openpathsampling.engines.trajectory import Trajectory

        return Trajectory
    if name == "FixedLengthTPSNetwork":
        from openpathsampling.high_level.network import FixedLengthTPSNetwork

        return FixedLengthTPSNetwork
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
    from openpathsampling.engines import DynamicsEngine
    from openpathsampling.engines.trajectory import Trajectory
    from openpathsampling.high_level.network import FixedLengthTPSNetwork

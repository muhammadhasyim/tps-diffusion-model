"""genai-tps: path sampling for generative-model trajectories.

Vendored OpenPathSampling drives ensembles and moves; optional backends (e.g.
Boltz-2) implement :class:`openpathsampling.engines.DynamicsEngine`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openpathsampling as paths

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

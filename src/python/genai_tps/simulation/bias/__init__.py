"""Bias potentials for enhanced sampling (OPES, harmonic umbrella)."""

from __future__ import annotations

from typing import Protocol, Union, runtime_checkable

import numpy as np

# CV value type: scalar float for 1-D bias, np.ndarray shape (D,) for N-D.
CVValue = Union[float, np.ndarray]


@runtime_checkable
class EnhancedSamplingBias(Protocol):
    """Protocol for enhanced sampling bias objects used in the TPS loop."""

    def compute_acceptance_factor(self, cv_old: CVValue, cv_new: CVValue) -> float:
        ...

    def update(self, cv_accepted: CVValue, mc_step: int) -> None:
        ...


from genai_tps.simulation.bias.opes import Kernel, OPESBias  # noqa: E402
from genai_tps.simulation.bias.umbrella import HarmonicUmbrellaBias  # noqa: E402

__all__ = [
    "CVValue",
    "EnhancedSamplingBias",
    "HarmonicUmbrellaBias",
    "Kernel",
    "OPESBias",
]

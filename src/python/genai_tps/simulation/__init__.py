"""MD simulation primitives and biases for enhanced / reweighted dynamics.

Provides OpenMM-backed drivers, weighted dataset assembly, and bias engines
(shared with accelerated path methods). For Boltz diffusion TPS checkpoints
and rollout helpers see :mod:`genai_tps.evaluation.tps_runner`.

Enhanced sampling biases live in :mod:`genai_tps.simulation.bias`:

1. **Harmonic umbrella** — fixed restraint V = 0.5*k*(φ-φ₀)² combined across
   windows via MBAR (WHAM-compatible reduced potentials).
2. **OPES** — adaptive KDE bias (OPES_METAD-style; Invernizzi & Parrinello).

Both implement :class:`~genai_tps.simulation.bias.EnhancedSamplingBias`.
"""

from genai_tps.simulation.bias import (
    CVValue,
    EnhancedSamplingBias,
    HarmonicUmbrellaBias,
    Kernel,
    OPESBias,
)

__all__ = [
    "CVValue",
    "EnhancedSamplingBias",
    "HarmonicUmbrellaBias",
    "Kernel",
    "OPESBias",
]

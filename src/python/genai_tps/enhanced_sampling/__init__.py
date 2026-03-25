"""Enhanced sampling techniques for TPS diffusion model sampling.

Two independent techniques are provided:

1. **Exponential Tilting + MBAR** (REA-style): fixed bias lambda * CV, combine
   samples from multiple lambda values via MBAR to reconstruct the unbiased
   distribution.  Reference: Dorman et al., "Rare Event Analysis of Large
   Language Models", arxiv 2602.06791.

2. **OPES Adaptive Bias**: on-the-fly construction of a bias potential using
   kernel density estimation with compression.  Ported from PLUMED2's
   OPES_METAD (Invernizzi & Parrinello, JPCL 2020).

Both techniques share a common ``EnhancedSamplingBias`` protocol that hooks into
the TPS acceptance probability via ``trial.bias`` in the MC loop.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EnhancedSamplingBias(Protocol):
    """Protocol for enhanced sampling bias objects used in the TPS loop.

    Implementations must provide:

    * ``compute_acceptance_factor(cv_old, cv_new)`` -- multiplicative factor
      applied to ``trial.bias`` in the Metropolis-Hastings acceptance.
    * ``update(cv_accepted, mc_step)`` -- called after each accept/reject
      decision with the CV value of the current (accepted) path.  Used by
      adaptive methods (OPES) to evolve the bias; no-op for fixed methods
      (exponential tilting).
    """

    def compute_acceptance_factor(self, cv_old: float, cv_new: float) -> float:
        """Return multiplicative factor for trial.bias given old and new CV values."""
        ...

    def update(self, cv_accepted: float, mc_step: int) -> None:
        """Update internal state after accept/reject with the accepted path's CV."""
        ...


from genai_tps.enhanced_sampling.exponential_tilting import ExponentialTiltingBias  # noqa: E402
from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: E402

__all__ = [
    "EnhancedSamplingBias",
    "ExponentialTiltingBias",
    "OPESBias",
]

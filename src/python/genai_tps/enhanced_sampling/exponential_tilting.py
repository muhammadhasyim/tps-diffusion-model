"""Exponential tilting bias for enhanced sampling in TPS.

Targets the tilted distribution:

    p_lambda(path) ~ exp(-lambda * phi(path)) * p_model(path)

where phi is a collective variable (e.g. RMSD of the last frame).  The bias
enters the TPS acceptance as a multiplicative factor:

    factor = exp(-lambda * [phi(trial) - phi(current)])

Multiple runs at different lambda values can be combined via MBAR to
reconstruct the unbiased distribution p(phi).

Reference: Dorman et al., "Rare Event Analysis of Large Language Models",
arxiv 2602.06791 -- adapted from token-sequence TPS to protein structure TPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ExponentialTiltingBias:
    """Fixed exponential tilting bias for TPS enhanced sampling.

    Parameters
    ----------
    lambda_value : float
        Tilting strength.  Positive lambda biases toward lower CV values;
        negative lambda biases toward higher CV values.  lambda=0 recovers
        unbiased sampling.
    """

    lambda_value: float = 0.0
    _samples: list[dict] = field(default_factory=list, repr=False)

    def compute_acceptance_factor(self, cv_old: float, cv_new: float) -> float:
        """Return exp(-lambda * (cv_new - cv_old)).

        When lambda > 0, moves that decrease the CV are favored.
        When lambda < 0, moves that increase the CV are favored.
        When lambda = 0, returns 1.0 (no bias).
        """
        if self.lambda_value == 0.0:
            return 1.0
        delta = cv_new - cv_old
        exponent = -self.lambda_value * delta
        exponent = max(-700.0, min(700.0, exponent))
        return math.exp(exponent)

    def update(self, cv_accepted: float, mc_step: int) -> None:
        """Record the accepted CV value for later MBAR analysis."""
        self._samples.append({
            "mc_step": mc_step,
            "cv": cv_accepted,
            "lambda": self.lambda_value,
        })

    def set_lambda(self, new_lambda: float) -> None:
        """Change lambda for annealing schedules."""
        self.lambda_value = new_lambda

    @property
    def samples(self) -> list[dict]:
        """All recorded (mc_step, cv, lambda) tuples."""
        return self._samples

    def clear_samples(self) -> None:
        """Discard recorded samples (e.g. burn-in)."""
        self._samples.clear()

    def reduced_potential(self, cv: float) -> float:
        """Dimensionless reduced potential u = lambda * cv.

        Used to construct the u_kn matrix for MBAR.
        """
        return self.lambda_value * cv

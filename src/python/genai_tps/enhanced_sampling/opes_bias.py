"""OPES (On-the-fly Probability Enhanced Sampling) adaptive bias for TPS.

Ported from PLUMED2's OPES_METAD implementation (OPESmetad.cpp).

Builds a bias potential on-the-fly using kernel density estimation with
compression.  The bias flattens the CV distribution toward a well-tempered
target, enabling exploration of rare-event tails.

Key equations (1D, non-periodic CVs):

    P(s) = (1/W) * sum_i h_i * G(s; c_i, sigma_i)     [KDE estimate]
    V(s) = (1 - 1/gamma) * kT * log(P(s)/Z + epsilon)  [bias potential]

where W = sum(h_i) (convergence) or N_kernels_deposited (explore), Z is
the normalization, and gamma is the biasfactor controlling the well-tempered
target p_target(s) ~ [p(s)]^(1/gamma).

The bias enters TPS acceptance as:

    trial.bias *= exp(-(V(cv_new) - V(cv_old)) / kT)

References:
    Invernizzi & Parrinello, JPCL 11, 2731 (2020).
    Invernizzi & Parrinello, J. Chem. Theory Comput. 18, 3988 (2022).
    PLUMED2 source: plumed2/src/opes/OPESmetad.cpp
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_LOG_OVERFLOW_GUARD = 700.0


@dataclass
class Kernel:
    """A single Gaussian kernel in the OPES KDE.

    Attributes
    ----------
    center : float
        Location in CV space.
    sigma : float
        Width (standard deviation).
    height : float
        Unnormalized weight.
    """

    center: float
    sigma: float
    height: float


class OPESBias:
    """Adaptive OPES_METAD bias for 1D collective variables.

    Implements the convergence and explore modes of OPES_METAD from PLUMED2,
    restricted to a single non-periodic CV.

    Parameters
    ----------
    kbt : float
        Thermal energy kT in the same units as the bias.  For dimensionless
        TPS applications, set kbt=1.0 and interpret barrier/sigma in natural
        log units.
    barrier : float
        Estimated free-energy barrier in kT units.  Used to set the initial
        kernel width sigma_0 = sqrt(2 * barrier) (in CV units) and the
        default epsilon = exp(-barrier).
    biasfactor : float
        Well-tempering factor gamma.  Controls the target distribution:
        p_target(s) ~ [p(s)]^(1/gamma).  Larger gamma -> flatter target.
        Use float('inf') for a uniform target.
    epsilon : float or None
        Regularization in the log.  Default: exp(-barrier / kbt).
    kernel_cutoff : float
        Truncation radius in units of sigma.  Default: sqrt(2*6.5) ~ 3.6.
    compression_threshold : float
        Merge threshold in sigma-normalized distance.  Default: 1.0.
    pace : int
        Deposit a kernel every ``pace`` MC steps.  Default: 1.
    sigma_min : float
        Floor on the adaptive kernel width.  Default: 0.0 (no floor).
    fixed_sigma : float or None
        If set, use this fixed sigma instead of adaptive estimation.
    explore : bool
        If True, use explore mode (unweighted KDE of the biased distribution).
        If False (default), use convergence mode (reweighted KDE estimating
        the unbiased distribution).
    """

    def __init__(
        self,
        kbt: float = 1.0,
        barrier: float = 5.0,
        biasfactor: float = 10.0,
        epsilon: float | None = None,
        kernel_cutoff: float | None = None,
        compression_threshold: float = 1.0,
        pace: int = 1,
        sigma_min: float = 0.0,
        fixed_sigma: float | None = None,
        explore: bool = False,
    ):
        self.kbt = float(kbt)
        self.barrier = float(barrier)
        self.biasfactor = float(biasfactor)
        self.explore = explore

        if math.isinf(self.biasfactor):
            self.bias_prefactor = 1.0
        else:
            self.bias_prefactor = 1.0 - 1.0 / self.biasfactor

        if epsilon is not None:
            self.epsilon = float(epsilon)
        else:
            self.epsilon = math.exp(-self.barrier / self.kbt)

        if kernel_cutoff is not None:
            self.kernel_cutoff = float(kernel_cutoff)
        else:
            self.kernel_cutoff = math.sqrt(2.0 * 6.5)
        self.cutoff_value = math.exp(-0.5 * self.kernel_cutoff ** 2)

        self.compression_threshold = float(compression_threshold)
        self.pace = max(1, int(pace))
        self.sigma_min = float(sigma_min)
        self.fixed_sigma = float(fixed_sigma) if fixed_sigma is not None else None

        self.sigma_0 = math.sqrt(2.0 * self.barrier / self.kbt) if self.barrier > 0 else 0.5

        self.kernels: list[Kernel] = []
        self.sum_weights = 0.0
        self.sum_weights2 = 0.0
        self.counter = 0
        self.zed = 1.0
        self.kde_norm = 0.0

        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0

        self._n_merges = 0

    def _evaluate_single_kernel(self, kernel: Kernel, cv: float) -> float:
        """Evaluate a single truncated Gaussian kernel at cv."""
        d = (cv - kernel.center) / kernel.sigma
        if abs(d) >= self.kernel_cutoff:
            return 0.0
        return kernel.height * (math.exp(-0.5 * d * d) - self.cutoff_value)

    def _raw_kde(self, cv: float) -> float:
        """Unnormalized KDE sum at cv (sum of h_i * G(cv; c_i, sigma_i))."""
        total = 0.0
        for k in self.kernels:
            total += self._evaluate_single_kernel(k, cv)
        return total

    def kde_probability(self, cv: float) -> float:
        """Normalized OPES kernel mixture density P(s) = raw_kde(s) / kde_norm.

        This is the KDE estimate used inside :meth:`evaluate` (before division
        by ``zed`` and addition of ``epsilon``).  Useful for plotting the
        deposited kernel density.
        """
        if not self.kernels or self.kde_norm <= 0:
            return 0.0
        return self._raw_kde(cv) / self.kde_norm

    def evaluate(self, cv: float) -> float:
        """Compute the bias potential V(cv).

        Returns
        -------
        float
            The bias V(cv) = bias_prefactor * kT * log(P(cv)/Z + epsilon),
            where P = raw_kde / kde_norm.  Returns 0.0 before any kernels
            are deposited.
        """
        if not math.isfinite(cv):
            return 0.0
        if not self.kernels or self.kde_norm == 0:
            return 0.0
        prob = self._raw_kde(cv) / self.kde_norm
        return self.kbt * self.bias_prefactor * math.log(prob / self.zed + self.epsilon)

    def compute_acceptance_factor(self, cv_old: float, cv_new: float) -> float:
        """Return exp(-(V(cv_new) - V(cv_old)) / kT) for the Metropolis criterion."""
        if not self.kernels:
            return 1.0
        if not math.isfinite(cv_old) or not math.isfinite(cv_new):
            return 1.0
        v_old = self.evaluate(cv_old)
        v_new = self.evaluate(cv_new)
        exponent = -(v_new - v_old) / self.kbt
        exponent = max(-_LOG_OVERFLOW_GUARD, min(_LOG_OVERFLOW_GUARD, exponent))
        return math.exp(exponent)

    def update(self, cv_accepted: float, mc_step: int) -> None:
        """Deposit a kernel (every ``pace`` steps) and update the bias.

        Parameters
        ----------
        cv_accepted
            CV value of the currently accepted path.
        mc_step
            Current MC step number (1-indexed).
        """
        if not math.isfinite(cv_accepted):
            return
        self._welford_update(cv_accepted)

        if mc_step % self.pace != 0:
            return

        sigma = self._current_sigma()
        height = self._compute_height(cv_accepted)

        self.counter += 1
        self.sum_weights += height
        self.sum_weights2 += height * height
        self.kde_norm = self.counter if self.explore else self.sum_weights

        height_adjusted = height * (self.sigma_0 / sigma) if sigma > 0 else height
        self._add_kernel(cv_accepted, sigma, height_adjusted)
        self._update_zed()

    def _compute_height(self, cv: float) -> float:
        """Kernel height: exp(V/kT) in convergence mode, 1.0 in explore mode."""
        if self.explore:
            return 1.0
        if not self.kernels:
            return 1.0
        if not math.isfinite(cv):
            return 1.0
        current_bias = self.evaluate(cv)
        log_weight = current_bias / self.kbt
        log_weight = max(-_LOG_OVERFLOW_GUARD, min(_LOG_OVERFLOW_GUARD, log_weight))
        return math.exp(log_weight)

    def _current_sigma(self) -> float:
        """Current kernel width from adaptive estimation or fixed value."""
        if self.fixed_sigma is not None:
            return self.fixed_sigma
        if self._welford_n < 2:
            return self.sigma_0
        variance = self._welford_m2 / self._welford_n
        if self.explore:
            size = float(self.counter) if self.counter > 0 else 1.0
        else:
            neff = (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights2) if self.sum_weights2 > 0 else 1.0
            size = neff
        # Silverman's rule rescaling for 1D
        s_rescaling = size ** (-1.0 / 5.0) if size > 0 else 1.0
        sigma = math.sqrt(max(variance, 1e-12)) * s_rescaling
        return max(sigma, self.sigma_min) if self.sigma_min > 0 else max(sigma, 1e-8)

    def _welford_update(self, cv: float) -> None:
        """Welford online running mean and variance of CV values."""
        self._welford_n += 1
        delta = cv - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = cv - self._welford_mean
        self._welford_m2 += delta * delta2

    def _add_kernel(self, center: float, sigma: float, height: float) -> None:
        """Add a kernel, merging with an existing one if within threshold."""
        merge_idx = self._find_mergeable(center, sigma)
        if merge_idx is not None:
            self._merge_kernel(merge_idx, center, sigma, height)
        else:
            self.kernels.append(Kernel(center=center, sigma=sigma, height=height))

    def _find_mergeable(self, center: float, sigma: float) -> int | None:
        """Find the index of an existing kernel within compression threshold."""
        threshold2 = self.compression_threshold ** 2
        best_idx = None
        best_dist2 = float("inf")
        for i, k in enumerate(self.kernels):
            d = (center - k.center) / k.sigma
            dist2 = d * d
            if dist2 < threshold2 and dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = i
        return best_idx

    def _merge_kernel(
        self, idx: int, new_center: float, new_sigma: float, new_height: float
    ) -> None:
        """Merge a new kernel with existing kernel at index idx.

        Height-weighted average of centers and variances, following PLUMED2.
        """
        old = self.kernels[idx]
        total_h = old.height + new_height

        if total_h <= 0:
            return

        w_old = old.height / total_h
        w_new = new_height / total_h

        merged_center = w_old * old.center + w_new * new_center

        old_var = old.sigma ** 2
        new_var = new_sigma ** 2
        merged_var = (
            w_old * (old_var + (old.center - merged_center) ** 2)
            + w_new * (new_var + (new_center - merged_center) ** 2)
        )
        merged_sigma = math.sqrt(max(merged_var, 1e-16))

        self.kernels[idx] = Kernel(
            center=merged_center,
            sigma=merged_sigma,
            height=total_h,
        )
        self._n_merges += 1

    def _update_zed(self) -> None:
        """Update the normalization Z = mean of P(c_i) over kernel centers.

        Z = (1 / (kde_norm * N_kernels)) * sum_k sum_kk G(c_k; c_kk, sigma_kk) * h_kk

        This is the average probability density at kernel centers, matching
        PLUMED2's full O(N^2) computation (acceptable since compression keeps
        N_kernels small).
        """
        n = len(self.kernels)
        if n == 0 or self.kde_norm == 0:
            self.zed = 1.0
            return
        sum_uprob = 0.0
        for k in self.kernels:
            for kk in self.kernels:
                sum_uprob += self._evaluate_single_kernel(kk, k.center)
        self.zed = sum_uprob / self.kde_norm / n

    @property
    def n_kernels(self) -> int:
        return len(self.kernels)

    @property
    def n_effective(self) -> float:
        """Effective sample size from weights."""
        if self.sum_weights2 <= 0:
            return float(self.counter)
        return (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights2)

    @property
    def rct(self) -> float:
        """Reweighting factor c(t) = kT * log(sum_weights / counter)."""
        if self.counter <= 0:
            return 0.0
        return self.kbt * math.log(max(self.sum_weights, 1e-300) / self.counter)

    def bias_on_grid(
        self, cv_min: float, cv_max: float, n_points: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the bias on a uniform grid for plotting.

        Returns
        -------
        cv_grid : np.ndarray
            CV values.
        bias_grid : np.ndarray
            V(cv) at each grid point.
        """
        cv_grid = np.linspace(cv_min, cv_max, n_points)
        bias_grid = np.array([self.evaluate(float(s)) for s in cv_grid])
        return cv_grid, bias_grid

    def reweight_samples(
        self, cv_values: np.ndarray
    ) -> np.ndarray:
        """Compute reweighting factors to recover the unbiased distribution.

        For each sample with CV value s, the weight is:

            w(s) = exp(V(s) / kT) / sum_i exp(V(s_i) / kT)

        Parameters
        ----------
        cv_values
            Array of CV values from the biased TPS run.

        Returns
        -------
        np.ndarray
            Normalized weights (sum to 1).
        """
        log_weights = np.array([self.evaluate(float(s)) / self.kbt for s in cv_values])
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        return weights

    def save_state(self, path: Path | str) -> None:
        """Serialize the full OPES state to JSON for restart."""
        state = {
            "kbt": self.kbt,
            "barrier": self.barrier,
            "biasfactor": self.biasfactor,
            "epsilon": self.epsilon,
            "kernel_cutoff": self.kernel_cutoff,
            "compression_threshold": self.compression_threshold,
            "pace": self.pace,
            "sigma_min": self.sigma_min,
            "fixed_sigma": self.fixed_sigma,
            "explore": self.explore,
            "bias_prefactor": self.bias_prefactor,
            "sigma_0": self.sigma_0,
            "sum_weights": self.sum_weights,
            "sum_weights2": self.sum_weights2,
            "counter": self.counter,
            "zed": self.zed,
            "kde_norm": self.kde_norm,
            "welford_n": self._welford_n,
            "welford_mean": self._welford_mean,
            "welford_m2": self._welford_m2,
            "n_merges": self._n_merges,
            "kernels": [
                {"center": k.center, "sigma": k.sigma, "height": k.height}
                for k in self.kernels
            ],
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(cls, path: Path | str) -> "OPESBias":
        """Restore an OPESBias from a saved state file."""
        state = json.loads(Path(path).read_text())
        bias = cls(
            kbt=state["kbt"],
            barrier=state["barrier"],
            biasfactor=state["biasfactor"],
            epsilon=state["epsilon"],
            kernel_cutoff=state["kernel_cutoff"],
            compression_threshold=state["compression_threshold"],
            pace=state["pace"],
            sigma_min=state["sigma_min"],
            fixed_sigma=state.get("fixed_sigma"),
            explore=state.get("explore", False),
        )
        bias.sum_weights = state["sum_weights"]
        bias.sum_weights2 = state["sum_weights2"]
        bias.counter = state["counter"]
        bias.zed = state["zed"]
        bias.kde_norm = state["kde_norm"]
        bias._welford_n = state["welford_n"]
        bias._welford_mean = state["welford_mean"]
        bias._welford_m2 = state["welford_m2"]
        bias._n_merges = state.get("n_merges", 0)
        bias.kernels = [
            Kernel(center=k["center"], sigma=k["sigma"], height=k["height"])
            for k in state["kernels"]
        ]
        return bias

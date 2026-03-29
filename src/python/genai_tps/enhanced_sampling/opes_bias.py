"""OPES (On-the-fly Probability Enhanced Sampling) adaptive bias for TPS.

Ported from PLUMED2's OPES_METAD implementation (OPESmetad.cpp).

Builds a bias potential on-the-fly using kernel density estimation with
compression.  The bias flattens the CV distribution toward a well-tempered
target, enabling exploration of rare-event tails.

Key equations (N-D, non-periodic CVs):

    P(s) = (1/W) * sum_i h_i * G(s; c_i, Sigma_i)   [factorised Gaussian KDE]
    V(s) = (1 - 1/gamma) * kT * log(P(s)/Z + epsilon) [bias potential]

where G is a product of 1-D truncated Gaussians with per-dimension width
sigma_i (diagonal covariance), W = sum(h_i) (convergence) or N_kernels
(explore), Z is the normalisation, and gamma is the biasfactor.

The bias enters TPS acceptance as:

    trial.bias *= exp(-(V(cv_new) - V(cv_old)) / kT)

For D=1 the class is backward-compatible with the previous scalar API: scalar
float inputs are accepted everywhere and scalar floats are returned from
``evaluate`` / ``compute_acceptance_factor``.

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
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

_LOG_OVERFLOW_GUARD = 700.0

# Type alias for a CV value: scalar or 1-D array.
CVType = Union[float, np.ndarray]


def _to_array(v: CVType, ndim: int) -> np.ndarray:
    """Convert a scalar or array CV value to a float64 array of shape (ndim,)."""
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if arr.shape != (ndim,):
        raise ValueError(
            f"CV value has shape {arr.shape}; expected ({ndim},). "
            "Pass a scalar for 1-D bias or an array of length ndim."
        )
    return arr


def _all_finite(v: CVType) -> bool:
    """Return True iff all elements of v are finite."""
    return bool(np.all(np.isfinite(np.atleast_1d(np.asarray(v, dtype=np.float64)))))


@dataclass
class Kernel:
    """A single diagonal-Gaussian kernel in the OPES KDE.

    Attributes
    ----------
    center : np.ndarray
        Location in CV space, shape ``(D,)``.
    sigma : np.ndarray
        Per-dimension width (standard deviation), shape ``(D,)``.
    height : float
        Unnormalized weight.
    """

    center: np.ndarray
    sigma: np.ndarray
    height: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).ravel()
        self.sigma = np.asarray(self.sigma, dtype=np.float64).ravel()
        if self.center.shape != self.sigma.shape:
            raise ValueError("Kernel center and sigma must have the same length.")


class OPESBias:
    """Adaptive OPES_METAD bias for N-dimensional collective variables.

    Implements the convergence and explore modes of OPES_METAD from PLUMED2,
    supporting diagonal-covariance Gaussians in an arbitrary number of CV
    dimensions.  For ``ndim=1`` the API is fully backward-compatible with the
    previous scalar version.

    Parameters
    ----------
    ndim : int
        Number of CV dimensions.  ``ndim=1`` (default) retains the scalar API.
    kbt : float
        Thermal energy kT in the same units as the bias.
    barrier : float
        Estimated free-energy barrier in kT units.  Sets the initial kernel
        width ``sigma_0 = sqrt(2 * barrier / kbt)`` (broadcast to all dims).
    biasfactor : float
        Well-tempering factor gamma.
    epsilon : float or None
        Regularization in the log.  Default: exp(-barrier / kbt).
    kernel_cutoff : float
        Truncation radius in units of sigma.  Default: sqrt(2 * 6.5) ~ 3.6.
    compression_threshold : float
        Merge threshold in sigma-normalised distance.  Default: 1.0.
    pace : int
        Deposit a kernel every ``pace`` MC steps.  Default: 1.
    sigma_min : float or sequence
        Floor on the adaptive kernel width per dimension.  Default: 0.0.
    fixed_sigma : float or sequence or None
        If set, use this fixed sigma (per-dim or broadcast) instead of adaptive.
    explore : bool
        If True, use explore mode.  Default: convergence mode.
    """

    def __init__(
        self,
        ndim: int = 1,
        kbt: float = 1.0,
        barrier: float = 5.0,
        biasfactor: float = 10.0,
        epsilon: float | None = None,
        kernel_cutoff: float | None = None,
        compression_threshold: float = 1.0,
        pace: int = 1,
        sigma_min: float | np.ndarray = 0.0,
        fixed_sigma: float | np.ndarray | None = None,
        explore: bool = False,
    ):
        self.ndim = max(1, int(ndim))
        self.kbt = float(kbt)
        self.barrier = float(barrier)
        self.biasfactor = float(biasfactor)
        self.explore = explore

        if math.isinf(self.biasfactor):
            self.bias_prefactor = 1.0
        else:
            self.bias_prefactor = 1.0 - 1.0 / self.biasfactor

        self.epsilon = (
            float(epsilon)
            if epsilon is not None
            else math.exp(-self.barrier / self.kbt)
        )

        self.kernel_cutoff = (
            float(kernel_cutoff)
            if kernel_cutoff is not None
            else math.sqrt(2.0 * 6.5)
        )
        self.cutoff2 = self.kernel_cutoff ** 2
        self.cutoff_value = math.exp(-0.5 * self.kernel_cutoff ** 2)

        self.compression_threshold = float(compression_threshold)
        self.threshold2 = self.compression_threshold ** 2
        self.pace = max(1, int(pace))

        # Per-dimension sigma_min and sigma_0
        sigma_0_val = (
            math.sqrt(2.0 * self.barrier / self.kbt) if self.barrier > 0 else 0.5
        )
        self.sigma_0: np.ndarray = np.full(self.ndim, sigma_0_val, dtype=np.float64)

        # sigma_min: broadcast scalar or per-dim array
        sm = np.atleast_1d(np.asarray(sigma_min, dtype=np.float64))
        self.sigma_min: np.ndarray = (
            np.broadcast_to(sm, (self.ndim,)).copy()
            if sm.shape != (self.ndim,)
            else sm.copy()
        )

        # fixed_sigma: None or per-dim array
        if fixed_sigma is not None:
            fs = np.atleast_1d(np.asarray(fixed_sigma, dtype=np.float64))
            self.fixed_sigma: np.ndarray | None = (
                np.broadcast_to(fs, (self.ndim,)).copy()
                if fs.shape != (self.ndim,)
                else fs.copy()
            )
        else:
            self.fixed_sigma = None

        self.kernels: list[Kernel] = []
        self.sum_weights = 0.0
        self.sum_weights2 = 0.0
        self.counter = 0
        self.zed = 1.0
        self.kde_norm = 0.0

        # Per-dimension Welford accumulators
        self._welford_n = 0
        self._welford_mean: np.ndarray = np.zeros(self.ndim, dtype=np.float64)
        self._welford_m2: np.ndarray = np.zeros(self.ndim, dtype=np.float64)

        self._n_merges = 0

    # ------------------------------------------------------------------
    # Kernel evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_kernel(self, kernel: Kernel, cv: CVType) -> float:
        """Evaluate a single truncated diagonal-Gaussian kernel at cv.

        Parameters
        ----------
        kernel : Kernel
            A deposited Gaussian kernel.
        cv : float or np.ndarray
            Current CV value.  Scalars are accepted for 1-D biases.

        Returns
        -------
        float
            ``height * (exp(-0.5 * norm2) - cutoff_value)`` or 0 if truncated.

        Notes
        -----
        Mirrors PLUMED2 ``evaluateKernel`` (OPESmetad.cpp lines 1661-1671):
        the squared Mahalanobis distance is accumulated dimension-by-dimension
        and evaluation is short-circuited as soon as it exceeds the cutoff.
        """
        cv_arr = self._cv_array(cv)
        norm2 = 0.0
        for i in range(self.ndim):
            d = (cv_arr[i] - kernel.center[i]) / kernel.sigma[i]
            norm2 += d * d
            if norm2 >= self.cutoff2:
                return 0.0
        return kernel.height * (math.exp(-0.5 * norm2) - self.cutoff_value)

    def _raw_kde(self, cv: CVType) -> float:
        """Unnormalized KDE sum at cv."""
        total = 0.0
        for k in self.kernels:
            total += self._evaluate_single_kernel(k, cv)
        return total

    def _cv_array(self, cv: CVType) -> np.ndarray:
        """Convert scalar or array to float64 array of shape (ndim,)."""
        return _to_array(cv, self.ndim)

    def kde_probability(self, cv: CVType) -> float:
        """Normalized OPES kernel mixture density P(s) = raw_kde(s) / kde_norm.

        This is the KDE estimate used inside :meth:`evaluate` (before division
        by ``zed`` and addition of ``epsilon``).  Useful for plotting.
        """
        if not self.kernels or self.kde_norm <= 0:
            return 0.0
        return self._raw_kde(self._cv_array(cv)) / self.kde_norm

    def evaluate(self, cv: CVType) -> float:
        """Compute the bias potential V(cv).

        Parameters
        ----------
        cv : float or np.ndarray
            CV value.  Pass a scalar float for 1-D; ``np.ndarray`` of shape
            ``(D,)`` for multi-D.

        Returns
        -------
        float
            ``bias_prefactor * kT * log(P(cv)/Z + epsilon)``.
            Returns 0.0 before any kernels are deposited.
        """
        if not _all_finite(cv):
            return 0.0
        if not self.kernels or self.kde_norm == 0:
            return 0.0
        cv_arr = self._cv_array(cv)
        prob = self._raw_kde(cv_arr) / self.kde_norm
        return self.kbt * self.bias_prefactor * math.log(prob / self.zed + self.epsilon)

    def compute_acceptance_factor(self, cv_old: CVType, cv_new: CVType) -> float:
        """Return exp(-(V(cv_new) - V(cv_old)) / kT) for the Metropolis criterion."""
        if not self.kernels:
            return 1.0
        if not _all_finite(cv_old) or not _all_finite(cv_new):
            return 1.0
        v_old = self.evaluate(cv_old)
        v_new = self.evaluate(cv_new)
        exponent = -(v_new - v_old) / self.kbt
        exponent = max(-_LOG_OVERFLOW_GUARD, min(_LOG_OVERFLOW_GUARD, exponent))
        return math.exp(exponent)

    # ------------------------------------------------------------------
    # Adaptive sigma (Welford + Silverman)
    # ------------------------------------------------------------------

    def _welford_update(self, cv: np.ndarray) -> None:
        """Per-dimension Welford online mean and variance update.

        Notes
        -----
        Mirrors PLUMED2 OPESmetad.cpp lines 1006-1010:
        ``av_cv_[i] += diff_i/tau; av_M2_[i] += diff_i * diff_after_i``.
        Here we use the classic Welford formulation (equivalent for tau=n).
        """
        self._welford_n += 1
        delta = cv - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = cv - self._welford_mean
        self._welford_m2 += delta * delta2

    def _current_sigma(self) -> np.ndarray:
        """Current kernel width vector from adaptive estimation or fixed value.

        Returns
        -------
        np.ndarray
            Shape ``(D,)``.  Uses PLUMED2's D-dimensional Silverman rule
            (OPESmetad.cpp line 1094):

            .. math::

                s_{\\text{rescale}} =
                    \\left(\\text{size} \\cdot \\frac{D+2}{4}\\right)^{-1/(4+D)}

        """
        if self.fixed_sigma is not None:
            return self.fixed_sigma.copy()

        if self._welford_n < 2:
            return self.sigma_0.copy()

        variance = self._welford_m2 / self._welford_n  # shape (D,)

        if self.explore:
            size = float(self.counter) if self.counter > 0 else 1.0
        else:
            neff = (
                (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights2)
                if self.sum_weights2 > 0
                else 1.0
            )
            size = neff

        # D-dimensional Silverman rule from PLUMED2 (OPESmetad.cpp:1094)
        d = float(self.ndim)
        s_rescaling = (max(size, 1e-300) * (d + 2.0) / 4.0) ** (-1.0 / (4.0 + d))

        sigma = np.sqrt(np.maximum(variance, 1e-12)) * s_rescaling  # shape (D,)

        # Apply sigma_min floor per dimension
        has_floor = self.sigma_min > 0
        sigma = np.where(has_floor, np.maximum(sigma, self.sigma_min), np.maximum(sigma, 1e-8))
        return sigma

    # ------------------------------------------------------------------
    # Kernel management
    # ------------------------------------------------------------------

    def _find_mergeable(self, center: np.ndarray, sigma: np.ndarray) -> int | None:
        """Find the index of the closest mergeable kernel.

        Distance is the sum of squared per-dimension normalised deltas
        (PLUMED2 OPESmetad.cpp lines 1404-1411).  Returns ``None`` if no
        kernel is within ``compression_threshold^2``.
        """
        best_idx = None
        best_dist2 = self.threshold2
        for i, k in enumerate(self.kernels):
            norm2 = 0.0
            for dim in range(self.ndim):
                d = (center[dim] - k.center[dim]) / k.sigma[dim]
                norm2 += d * d
                if norm2 >= best_dist2:
                    break
            if norm2 < best_dist2:
                best_dist2 = norm2
                best_idx = i
        return best_idx

    def _merge_kernel(
        self,
        idx: int,
        new_center: np.ndarray,
        new_sigma: np.ndarray,
        new_height: float,
    ) -> None:
        """Merge a new kernel into the existing kernel at index idx.

        Per-dimension height-weighted center and variance, following
        PLUMED2 OPESmetad.cpp lines 1693-1712.
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
        merged_sigma = np.sqrt(np.maximum(merged_var, 1e-16))

        self.kernels[idx] = Kernel(
            center=merged_center,
            sigma=merged_sigma,
            height=total_h,
        )
        self._n_merges += 1

    def _add_kernel(
        self, center: np.ndarray, sigma: np.ndarray, height: float
    ) -> None:
        """Add a kernel, merging with an existing one if within threshold.

        The height is already pre-adjusted by ``prod(sigma_0 / sigma)``
        before this call (done in :meth:`update`).
        """
        merge_idx = self._find_mergeable(center, sigma)
        if merge_idx is not None:
            self._merge_kernel(merge_idx, center, sigma, height)
        else:
            self.kernels.append(Kernel(center=center, sigma=sigma, height=height))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, cv_accepted: CVType, mc_step: int) -> None:
        """Deposit a kernel (every ``pace`` steps) and update the bias.

        Parameters
        ----------
        cv_accepted : float or np.ndarray
            CV value of the currently accepted path.  Scalar for 1-D.
        mc_step : int
            Current MC step number (1-indexed).
        """
        if not _all_finite(cv_accepted):
            return
        cv = self._cv_array(cv_accepted)
        self._welford_update(cv)

        if mc_step % self.pace != 0:
            return

        sigma = self._current_sigma()
        height = self._compute_height(cv)

        self.counter += 1
        self.sum_weights += height
        self.sum_weights2 += height * height
        self.kde_norm = self.counter if self.explore else self.sum_weights

        # Height rescaled by product of sigma_0/sigma (PLUMED2 lines 1107-1108)
        ratio = self.sigma_0 / np.maximum(sigma, 1e-300)
        height_adjusted = height * float(np.prod(ratio))

        self._add_kernel(cv, sigma, height_adjusted)
        self._update_zed()

    def _compute_height(self, cv: np.ndarray) -> float:
        """Kernel height: exp(V/kT) in convergence mode, 1.0 in explore mode."""
        if self.explore:
            return 1.0
        if not self.kernels:
            return 1.0
        current_bias = self.evaluate(cv)
        log_weight = current_bias / self.kbt
        log_weight = max(-_LOG_OVERFLOW_GUARD, min(_LOG_OVERFLOW_GUARD, log_weight))
        return math.exp(log_weight)

    def _update_zed(self) -> None:
        """Update Z = mean of P(c_i) over kernel centers (O(N^2) but N stays small)."""
        n = len(self.kernels)
        if n == 0 or self.kde_norm == 0:
            self.zed = 1.0
            return
        sum_uprob = 0.0
        for k in self.kernels:
            for kk in self.kernels:
                sum_uprob += self._evaluate_single_kernel(kk, k.center)
        self.zed = sum_uprob / self.kde_norm / n

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Grid evaluation and reweighting
    # ------------------------------------------------------------------

    def bias_on_grid(
        self, cv_min: float, cv_max: float, n_points: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the bias on a uniform grid (1-D only).

        For multi-D biases (ndim > 1), this method raises ``NotImplementedError``.
        Use :meth:`evaluate` directly for multi-D.

        Returns
        -------
        cv_grid : np.ndarray
            CV values.
        bias_grid : np.ndarray
            V(cv) at each grid point.
        """
        if self.ndim != 1:
            raise NotImplementedError(
                "bias_on_grid is only supported for 1-D biases (ndim=1). "
                "For multi-D, call evaluate() directly."
            )
        cv_grid = np.linspace(cv_min, cv_max, n_points)
        bias_grid = np.array([self.evaluate(float(s)) for s in cv_grid])
        return cv_grid, bias_grid

    def reweight_samples(
        self, cv_values: np.ndarray
    ) -> np.ndarray:
        """Compute reweighting factors to recover the unbiased distribution.

        Parameters
        ----------
        cv_values : np.ndarray
            For 1-D: shape ``(N,)``.  For N-D: shape ``(N, D)``.

        Returns
        -------
        np.ndarray
            Normalized weights (sum to 1).
        """
        if self.ndim == 1:
            log_weights = np.array(
                [self.evaluate(float(s)) / self.kbt for s in cv_values]
            )
        else:
            log_weights = np.array(
                [self.evaluate(cv_values[i]) / self.kbt for i in range(len(cv_values))]
            )
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_state(
        self,
        path: Path | str,
        *,
        bias_cv: str | None = None,
        bias_cv_names: list[str] | None = None,
    ) -> None:
        """Serialize the full OPES state to JSON for restart.

        Optional ``bias_cv`` / ``bias_cv_names`` are stored for plotting sanity
        checks (must match ``cv_type`` / ``cv_names`` in ``cv_values.json``).
        """
        state = {
            "ndim": self.ndim,
            "kbt": self.kbt,
            "barrier": self.barrier,
            "biasfactor": self.biasfactor,
            "epsilon": self.epsilon,
            "kernel_cutoff": self.kernel_cutoff,
            "compression_threshold": self.compression_threshold,
            "pace": self.pace,
            "sigma_min": self.sigma_min.tolist(),
            "fixed_sigma": (
                self.fixed_sigma.tolist() if self.fixed_sigma is not None else None
            ),
            "explore": self.explore,
            "bias_prefactor": self.bias_prefactor,
            "sigma_0": self.sigma_0.tolist(),
            "sum_weights": self.sum_weights,
            "sum_weights2": self.sum_weights2,
            "counter": self.counter,
            "zed": self.zed,
            "kde_norm": self.kde_norm,
            "welford_n": self._welford_n,
            "welford_mean": self._welford_mean.tolist(),
            "welford_m2": self._welford_m2.tolist(),
            "n_merges": self._n_merges,
            "kernels": [
                {
                    "center": k.center.tolist(),
                    "sigma": k.sigma.tolist(),
                    "height": k.height,
                }
                for k in self.kernels
            ],
        }
        if bias_cv is not None:
            state["bias_cv"] = str(bias_cv)
        if bias_cv_names is not None:
            state["bias_cv_names"] = [str(x) for x in bias_cv_names]
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(cls, path: Path | str) -> "OPESBias":
        """Restore an OPESBias from a saved state file.

        Backward-compatible with old 1-D states that stored scalar
        ``center`` / ``sigma`` / ``welford_mean`` / ``welford_m2``
        instead of lists.
        """
        state = json.loads(Path(path).read_text())

        ndim = int(state.get("ndim", 1))

        # Backward compat: old states may store sigma_min / fixed_sigma / sigma_0
        # as scalars rather than lists.
        def _load_array(key: str, default: list) -> list:
            val = state.get(key, default)
            if isinstance(val, (int, float)):
                return [val] * ndim
            return val

        sigma_min_list = _load_array("sigma_min", [0.0] * ndim)
        fixed_sigma_list = state.get("fixed_sigma", None)
        if fixed_sigma_list is not None and isinstance(fixed_sigma_list, (int, float)):
            fixed_sigma_list = [fixed_sigma_list] * ndim

        bias = cls(
            ndim=ndim,
            kbt=state["kbt"],
            barrier=state["barrier"],
            biasfactor=state["biasfactor"],
            epsilon=state["epsilon"],
            kernel_cutoff=state["kernel_cutoff"],
            compression_threshold=state["compression_threshold"],
            pace=state["pace"],
            sigma_min=np.array(sigma_min_list, dtype=np.float64),
            fixed_sigma=(
                np.array(fixed_sigma_list, dtype=np.float64)
                if fixed_sigma_list is not None
                else None
            ),
            explore=state.get("explore", False),
        )
        bias.sum_weights = state["sum_weights"]
        bias.sum_weights2 = state["sum_weights2"]
        bias.counter = state["counter"]
        bias.zed = state["zed"]
        bias.kde_norm = state["kde_norm"]
        bias._welford_n = state["welford_n"]

        # Backward compat: welford_mean / welford_m2 may be scalars (old 1-D saves)
        wm = state["welford_mean"]
        wm2 = state["welford_m2"]
        bias._welford_mean = np.atleast_1d(np.asarray(wm, dtype=np.float64))
        bias._welford_m2 = np.atleast_1d(np.asarray(wm2, dtype=np.float64))

        bias._n_merges = state.get("n_merges", 0)

        # sigma_0 override from saved state
        s0 = _load_array("sigma_0", [math.sqrt(2.0 * state["barrier"]) for _ in range(ndim)])
        bias.sigma_0 = np.array(s0, dtype=np.float64)

        # Kernels: center / sigma may be scalar (old 1-D) or list (new)
        kernels = []
        for k in state["kernels"]:
            c = k["center"]
            s = k["sigma"]
            if isinstance(c, (int, float)):
                c = [c]
            if isinstance(s, (int, float)):
                s = [s]
            kernels.append(
                Kernel(
                    center=np.array(c, dtype=np.float64),
                    sigma=np.array(s, dtype=np.float64),
                    height=k["height"],
                )
            )
        bias.kernels = kernels
        bias.saved_bias_cv = state.get("bias_cv")
        bnames = state.get("bias_cv_names")
        bias.saved_bias_cv_names = (
            [str(x) for x in bnames] if isinstance(bnames, list) else None
        )
        return bias

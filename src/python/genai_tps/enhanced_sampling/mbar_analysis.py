"""MBAR reweighting for exponential-tilting enhanced sampling.

Combines samples from TPS runs at multiple tilting strengths (lambda values)
to reconstruct the unbiased distribution p(CV) using the Multistate Bennett
Acceptance Ratio (MBAR) estimator.

Reference: Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008).
           pymbar: https://pymbar.readthedocs.io/

Requires: ``pip install pymbar``
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TiltedSamples:
    """Samples collected from a single TPS run at a fixed lambda value.

    Parameters
    ----------
    lambda_value : float
        The tilting strength used during sampling.
    cv_values : list[float]
        CV values of accepted paths (one per MC step after burn-in).
    """

    lambda_value: float
    cv_values: list[float] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.cv_values)


@dataclass
class MBARResult:
    """Output of MBAR distribution reconstruction.

    Attributes
    ----------
    bin_centers : np.ndarray
        Centers of histogram bins.
    bin_edges : np.ndarray
        Edges of histogram bins (length = len(bin_centers) + 1).
    bin_probabilities : np.ndarray
        Estimated probability in each bin under the unbiased (lambda=0) distribution.
    bin_uncertainties : np.ndarray
        Uncertainty (standard error) of each bin probability.
    free_energies : np.ndarray
        Free energy differences f_k for each state k (relative to first state).
    free_energy_uncertainties : np.ndarray
        Uncertainties of free energy differences.
    lambda_values : np.ndarray
        The lambda values used (K states).
    n_samples_per_state : np.ndarray
        Number of samples from each state.
    """

    bin_centers: np.ndarray
    bin_edges: np.ndarray
    bin_probabilities: np.ndarray
    bin_uncertainties: np.ndarray
    free_energies: np.ndarray
    free_energy_uncertainties: np.ndarray
    lambda_values: np.ndarray
    n_samples_per_state: np.ndarray


class MBARDistributionEstimator:
    """Reconstruct the unbiased p(CV) from multiple tilted TPS simulations.

    Usage::

        estimator = MBARDistributionEstimator()
        estimator.add_samples(TiltedSamples(lambda_value=0.0, cv_values=[...]))
        estimator.add_samples(TiltedSamples(lambda_value=1.0, cv_values=[...]))
        estimator.add_samples(TiltedSamples(lambda_value=-1.0, cv_values=[...]))
        result = estimator.estimate(n_bins=50)

    Parameters
    ----------
    unbiased_lambda : float
        Lambda value of the target (unbiased) state.  Default 0.0.
    """

    def __init__(self, unbiased_lambda: float = 0.0):
        self.unbiased_lambda = unbiased_lambda
        self._state_samples: list[TiltedSamples] = []

    def add_samples(self, samples: TiltedSamples) -> None:
        """Add samples from one tilted TPS run."""
        if samples.n_samples == 0:
            logger.warning(
                "Skipping empty sample set for lambda=%.4g", samples.lambda_value
            )
            return
        self._state_samples.append(samples)

    def add_samples_from_bias(
        self,
        bias: "ExponentialTiltingBias",  # noqa: F821
        burn_in_fraction: float = 0.1,
    ) -> None:
        """Extract samples from an ExponentialTiltingBias object.

        Parameters
        ----------
        bias
            The tilting bias whose recorded samples to extract.
        burn_in_fraction
            Fraction of initial samples to discard as burn-in.
        """
        by_lambda: dict[float, list[float]] = {}
        for s in bias.samples:
            lam = s["lambda"]
            by_lambda.setdefault(lam, []).append(s["cv"])

        for lam, cvs in by_lambda.items():
            n_burn = int(len(cvs) * burn_in_fraction)
            self.add_samples(TiltedSamples(lambda_value=lam, cv_values=cvs[n_burn:]))

    @property
    def n_states(self) -> int:
        return len(self._state_samples)

    @property
    def n_total_samples(self) -> int:
        return sum(s.n_samples for s in self._state_samples)

    def _build_u_kn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the reduced potential matrix u_kn for pymbar.

        Returns
        -------
        u_kn : np.ndarray, shape (K, N_total)
            Reduced potentials.  u_kn[k, n] = lambda_k * cv_n.
        N_k : np.ndarray, shape (K,)
            Number of samples per state.
        all_cvs : np.ndarray, shape (N_total,)
            Concatenated CV values.
        """
        K = self.n_states
        lambda_vals = np.array([s.lambda_value for s in self._state_samples])
        N_k = np.array([s.n_samples for s in self._state_samples], dtype=np.int64)
        all_cvs = np.concatenate([np.array(s.cv_values) for s in self._state_samples])
        N_total = len(all_cvs)

        u_kn = np.zeros((K, N_total), dtype=np.float64)
        for k in range(K):
            u_kn[k, :] = lambda_vals[k] * all_cvs

        return u_kn, N_k, all_cvs

    def estimate(
        self,
        n_bins: int = 50,
        cv_range: tuple[float, float] | None = None,
    ) -> MBARResult:
        """Run MBAR and reconstruct the unbiased distribution.

        Parameters
        ----------
        n_bins
            Number of histogram bins for the CV distribution.
        cv_range
            (min, max) range for the histogram.  If None, uses the range of
            all observed CV values with a 10% margin.

        Returns
        -------
        MBARResult
            Contains bin probabilities, uncertainties, and free energies.
        """
        try:
            import pymbar
        except ImportError as e:
            raise ImportError(
                "pymbar is required for MBAR analysis. Install with: pip install pymbar"
            ) from e

        if self.n_states < 2:
            raise ValueError(
                f"MBAR requires at least 2 states; got {self.n_states}. "
                "Add samples from multiple lambda values."
            )

        u_kn, N_k, all_cvs = self._build_u_kn()
        logger.info(
            "Running MBAR: %d states, %d total samples, lambda range [%.4g, %.4g]",
            self.n_states,
            len(all_cvs),
            u_kn[:, 0].min() if len(all_cvs) > 0 else 0,
            u_kn[:, 0].max() if len(all_cvs) > 0 else 0,
        )

        mbar = pymbar.MBAR(u_kn, N_k)

        fe_result = mbar.compute_free_energy_differences()
        f_k = fe_result["Delta_f"][0]
        df_k = fe_result["dDelta_f"][0]

        if cv_range is None:
            margin = 0.1 * (all_cvs.max() - all_cvs.min() + 1e-12)
            cv_range = (all_cvs.min() - margin, all_cvs.max() + margin)

        bin_edges = np.linspace(cv_range[0], cv_range[1], n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bin_probs = np.zeros(n_bins, dtype=np.float64)
        bin_uncerts = np.zeros(n_bins, dtype=np.float64)

        target_idx = self._find_unbiased_state_index()

        for b in range(n_bins):
            indicator = ((all_cvs >= bin_edges[b]) & (all_cvs < bin_edges[b + 1])).astype(
                np.float64
            )
            if indicator.sum() == 0:
                continue
            try:
                result = mbar.compute_expectations(indicator)
                bin_probs[b] = result["mu"][target_idx]
                bin_uncerts[b] = result["sigma"][target_idx]
            except Exception:
                pass

        bin_widths = np.diff(bin_edges)
        total_prob = (bin_probs * bin_widths).sum()
        if total_prob > 0:
            bin_probs /= total_prob
            bin_uncerts /= total_prob

        lambda_vals = np.array([s.lambda_value for s in self._state_samples])

        return MBARResult(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            bin_probabilities=bin_probs,
            bin_uncertainties=bin_uncerts,
            free_energies=f_k,
            free_energy_uncertainties=df_k,
            lambda_values=lambda_vals,
            n_samples_per_state=N_k,
        )

    def _find_unbiased_state_index(self) -> int:
        """Index of the state closest to unbiased_lambda."""
        lambdas = [s.lambda_value for s in self._state_samples]
        dists = [abs(lam - self.unbiased_lambda) for lam in lambdas]
        return int(np.argmin(dists))

    def save_samples(self, path: Path | str) -> None:
        """Persist all state samples to JSON for reproducibility."""
        path = Path(path)
        data = []
        for s in self._state_samples:
            data.append({
                "lambda_value": s.lambda_value,
                "n_samples": s.n_samples,
                "cv_values": s.cv_values,
            })
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d states (%d total samples) to %s",
                     len(data), self.n_total_samples, path)

    @classmethod
    def load_samples(cls, path: Path | str) -> "MBARDistributionEstimator":
        """Load state samples from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        estimator = cls()
        for entry in data:
            estimator.add_samples(
                TiltedSamples(
                    lambda_value=entry["lambda_value"],
                    cv_values=entry["cv_values"],
                )
            )
        logger.info("Loaded %d states (%d total samples) from %s",
                     estimator.n_states, estimator.n_total_samples, path)
        return estimator

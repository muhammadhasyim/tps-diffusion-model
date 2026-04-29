"""MBAR reweighting for umbrella-window and multi-state enhanced sampling.

Combines histograms from biased runs (harmonic umbrella windows) into an
unbiased :math:`p(\\mathrm{CV})` estimate using pymbar MBAR.

Reference: Shirts & Chodera, J. Chem. Phys. 129, 124105 (2008).

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
class WindowSamples:
    """Samples from one umbrella window (harmonic restraint on phi).

    Reduced potential under ensemble *j* at configuration *phi_n*:

        u_j(phi_n) = 0.5 * kappa_j * (phi_n - center_j)^2 / kbt

    (OpenMM umbrella / PLUMED RESTRAINT harmonic form.)
    """

    center: float
    kappa: float
    kbt: float
    cv_values: list[float] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.cv_values)


@dataclass
class MBARResult:
    """Output of MBAR distribution reconstruction."""

    bin_centers: np.ndarray
    bin_edges: np.ndarray
    bin_probabilities: np.ndarray
    bin_uncertainties: np.ndarray
    free_energies: np.ndarray
    free_energy_uncertainties: np.ndarray
    window_centers: np.ndarray
    kappa_values: np.ndarray
    n_samples_per_state: np.ndarray


class MBARDistributionEstimator:
    """Reconstruct unbiased p(CV) from pooled umbrella-window data."""

    def __init__(self, unbiased_center: float | None = None) -> None:
        self.unbiased_center = unbiased_center
        self._state_samples: list[WindowSamples] = []

    def add_samples(self, samples: WindowSamples) -> None:
        if samples.n_samples == 0:
            logger.warning("Skipping empty window at center=%.4g", samples.center)
            return
        self._state_samples.append(samples)

    def add_samples_from_bias(
        self,
        bias: object,
        *,
        burn_in_fraction: float = 0.1,
    ) -> None:
        """Group recordings from :class:`~genai_tps.simulation.bias.umbrella.HarmonicUmbrellaBias`."""  # noqa: E501
        from genai_tps.simulation.bias.umbrella import HarmonicUmbrellaBias as HU

        if not isinstance(bias, HU):
            raise TypeError(
                "add_samples_from_bias expects HarmonicUmbrellaBias, "
                f"got {type(bias).__name__}",
            )

        by_key: dict[tuple[float, float, float], list[float]] = {}
        for s in bias.samples:
            key = (float(s["center"]), float(s["kappa"]), float(s["kbt"]))
            by_key.setdefault(key, []).append(float(s["cv"]))

        for (center, kappa, kbt), cvs in sorted(by_key.items(), key=lambda kv: kv[0][0]):
            n_burn = int(len(cvs) * burn_in_fraction)
            self.add_samples(
                WindowSamples(
                    center=center,
                    kappa=kappa,
                    kbt=kbt,
                    cv_values=cvs[n_burn:],
                ),
            )

    @property
    def n_states(self) -> int:
        return len(self._state_samples)

    @property
    def n_total_samples(self) -> int:
        return sum(s.n_samples for s in self._state_samples)

    def _build_u_kn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.n_states
        n_k = np.array([s.n_samples for s in self._state_samples], dtype=np.int64)
        all_cvs = np.concatenate(
            [np.asarray(s.cv_values, dtype=np.float64) for s in self._state_samples],
        )

        centers = np.array([s.center for s in self._state_samples], dtype=np.float64)
        kappas = np.array([s.kappa for s in self._state_samples], dtype=np.float64)
        kbts = np.array([s.kbt for s in self._state_samples], dtype=np.float64)

        u_kn = np.zeros((k, len(all_cvs)), dtype=np.float64)
        for j in range(k):
            diff = all_cvs - centers[j]
            u_kn[j, :] = 0.5 * kappas[j] * diff * diff / kbts[j]

        return u_kn, n_k, all_cvs

    def estimate(
        self,
        n_bins: int = 50,
        cv_range: tuple[float, float] | None = None,
    ) -> MBARResult:
        import pymbar  # noqa: PLC0415

        if self.n_states < 2:
            raise ValueError(
                f"MBAR requires at least 2 states; got {self.n_states}.",
            )

        u_kn, n_k_arr, all_cvs = self._build_u_kn()

        mbar = pymbar.MBAR(u_kn, n_k_arr)
        fe_result = mbar.compute_free_energy_differences()
        f_k = fe_result["Delta_f"][0]
        df_k = fe_result["dDelta_f"][0]

        if cv_range is None:
            margin = 0.1 * (float(all_cvs.max() - all_cvs.min()) + 1e-12)
            cv_range = (float(all_cvs.min()) - margin, float(all_cvs.max()) + margin)

        bin_edges = np.linspace(cv_range[0], cv_range[1], n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_probs = np.zeros(n_bins, dtype=np.float64)
        bin_uncerts = np.zeros(n_bins, dtype=np.float64)

        target_idx = self._find_reference_state_index()

        for bidx in range(n_bins):
            indicator = (
                (all_cvs >= bin_edges[bidx]) & (all_cvs < bin_edges[bidx + 1])
            ).astype(np.float64)
            if float(indicator.sum()) == 0:
                continue
            result = mbar.compute_expectations(indicator)
            bin_probs[bidx] = result["mu"][target_idx]
            bin_uncerts[bidx] = result["sigma"][target_idx]

        widths = np.diff(bin_edges)
        tot = float((bin_probs * widths).sum())
        if tot > 0:
            bin_probs /= tot
            bin_uncerts /= tot

        wc = np.array([s.center for s in self._state_samples], dtype=np.float64)
        kap = np.array([s.kappa for s in self._state_samples], dtype=np.float64)

        return MBARResult(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            bin_probabilities=bin_probs,
            bin_uncertainties=bin_uncerts,
            free_energies=f_k,
            free_energy_uncertainties=df_k,
            window_centers=wc,
            kappa_values=kap,
            n_samples_per_state=n_k_arr,
        )

    def _find_reference_state_index(self) -> int:
        if self.unbiased_center is None:
            return 0
        centers_list = np.array([s.center for s in self._state_samples], dtype=np.float64)
        return int(np.argmin(np.abs(centers_list - float(self.unbiased_center))))

    def save_samples(self, path: Path | str) -> None:
        path_p = Path(path)
        blob: list[dict[str, float | list[float]]] = []
        for s in self._state_samples:
            blob.append(
                {
                    "center": float(s.center),
                    "kappa": float(s.kappa),
                    "kbt": float(s.kbt),
                    "n_samples": float(s.n_samples),
                    "cv_values": list(s.cv_values),
                },
            )
        path_p.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        logger.info(
            "Saved %d umbrella windows (%d total samples) to %s",
            len(blob),
            self.n_total_samples,
            path_p,
        )

    @classmethod
    def load_samples(cls, path: Path | str) -> MBARDistributionEstimator:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        est = cls()
        for row in raw:
            est.add_samples(
                WindowSamples(
                    center=float(row["center"]),
                    kappa=float(row["kappa"]),
                    kbt=float(row["kbt"]),
                    cv_values=[float(y) for y in row["cv_values"]],
                ),
            )
        logger.info(
            "Loaded %d umbrella windows (%d total samples) from %s",
            est.n_states,
            est.n_total_samples,
            path,
        )
        return est

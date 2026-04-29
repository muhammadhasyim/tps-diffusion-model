"""Harmonic umbrella bias for enhanced sampling (OpenMM / PLUMED-style RESTRAINT).

Bias potential on a scalar collective variable φ:

    V(φ) = 0.5 * kappa * (φ - center)^2

This matches the usual umbrella window form (see OpenMM umbrella tutorial and
PLUMED RESTRAINT with KAPPA + AT).  The Metropolis acceptance factor is

    exp(-(V(φ_new) - V(φ_old)) / kBT)

aligned with :class:`~genai_tps.simulation.bias.opes.OPESBias`.

References
----------
- https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/umbrella_sampling.html
- https://plumed.org/doc-v2.7/user-doc/html/_r_e_s_t_r_a_i_n_t.html
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class HarmonicUmbrellaBias:
    """Fixed harmonic umbrella bias for TPS / path sampling.

    Parameters
    ----------
    center
        Restraint center φ₀ (same units as the CV).
    kappa
        Spring constant *k* (energy / CV²), e.g. kJ/mol/Å² for RMSD in Å.
    kbt
        Thermal energy *k*\\ :sub:`B` *T* in the same energy units as *kappa*.
    """

    center: float = 0.0
    kappa: float = 1.0
    kbt: float = 2.494  # ~300 K in kJ/mol
    _samples: list[dict] = field(default_factory=list, repr=False)

    def bias_potential(self, cv: float) -> float:
        """Return V(φ) = 0.5 * kappa * (φ - center)²."""
        d = float(cv) - float(self.center)
        return 0.5 * float(self.kappa) * d * d

    def compute_acceptance_factor(
        self, cv_old: float | np.ndarray, cv_new: float | np.ndarray
    ) -> float:
        """Return exp(-(V_new - V_old) / kBT). Supports scalar CV only."""
        o = self._scalar_cv(cv_old)
        n = self._scalar_cv(cv_new)
        v_old = self.bias_potential(o)
        v_new = self.bias_potential(n)
        exponent = -(v_new - v_old) / float(self.kbt)
        exponent = max(-700.0, min(700.0, exponent))
        return math.exp(exponent)

    def update(self, cv_accepted: float | np.ndarray, mc_step: int) -> None:
        """Record accepted CV for MBAR-compatible post-processing."""
        phi = self._scalar_cv(cv_accepted)
        self._samples.append(
            {
                "mc_step": mc_step,
                "cv": phi,
                "center": float(self.center),
                "kappa": float(self.kappa),
                "kbt": float(self.kbt),
            },
        )

    def set_center(self, new_center: float) -> None:
        """Move umbrella center (annealing schedules)."""
        self.center = float(new_center)

    def set_kappa(self, new_kappa: float) -> None:
        """Change spring constant mid-run if needed."""
        self.kappa = float(new_kappa)

    def set_kbt(self, new_kbt: float) -> None:
        self.kbt = float(new_kbt)

    @property
    def samples(self) -> list[dict]:
        """All recorded tuples for MBAR (each row has center/kappa/kbt at deposition)."""
        return self._samples

    def clear_samples(self) -> None:
        self._samples.clear()

    def _scalar_cv(self, v: Union[float, np.ndarray]) -> float:
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size != 1:
            raise ValueError(
                "HarmonicUmbrellaBias currently supports scalar CV values only.",
            )
        return float(arr[0])

    def reduced_potential(self, cv: float) -> float:
        """Dimensionless u = V(φ) / (kBT) for pymbar."""
        return self.bias_potential(cv) / float(self.kbt)

    def save_state(self, path: Path | str) -> None:
        """Persist centers/kappa/kbt lists for reproducibility."""
        state = {
            "center": self.center,
            "kappa": self.kappa,
            "kbt": self.kbt,
            "samples": self._samples,
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, path: Path | str) -> HarmonicUmbrellaBias:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bias = cls(
            center=float(data["center"]),
            kappa=float(data["kappa"]),
            kbt=float(data["kbt"]),
        )
        bias._samples = list(data.get("samples", []))
        return bias

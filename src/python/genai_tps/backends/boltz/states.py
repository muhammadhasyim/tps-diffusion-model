"""State volumes (A/B) for diffusion TPS."""

from __future__ import annotations

import openpathsampling as paths
from openpathsampling.collectivevariable import FunctionCV

from genai_tps.backends.boltz.collective_variables import make_sigma_cv


def state_volume_high_sigma(
    sigma_min: float,
    sigma_max: float = 1.0e6,
    cv_name: str = "sigma",
) -> paths.Volume:
    """State A: structures still at high noise (sigma in ``[sigma_min, sigma_max]``)."""
    cv = make_sigma_cv(cv_name)
    return paths.CVDefinedVolume(cv, sigma_min, sigma_max)


def state_volume_quality(
    cv: FunctionCV,
    plddt_min: float,
    plddt_max: float = 100.0,
) -> paths.Volume:
    r"""State B: quality filter on a scalar CV (e.g. pLDDT proxy or RMSD band)."""
    return paths.CVDefinedVolume(cv, plddt_min, plddt_max)


def state_volume_low_sigma(
    sigma_max: float,
    sigma_min: float = 0.0,
    cv_name: str = "sigma",
) -> paths.Volume:
    """State B (denoised): diffusion noise scale below ``sigma_max``.

    Uses the same :func:`~genai_tps.backends.boltz.collective_variables.make_sigma_cv`
    as :func:`state_volume_high_sigma`. For fixed-length diffusion TPS, pair with
    :func:`state_volume_high_sigma` so frame 0 is high-:math:`\\sigma` and the last
    frame is low-:math:`\\sigma`.
    """
    cv = make_sigma_cv(cv_name)
    return paths.CVDefinedVolume(cv, sigma_min, sigma_max)

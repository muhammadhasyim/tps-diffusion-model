"""Log path probability contributions for Boltz-2 diffusion trajectories."""

from __future__ import annotations

import math
from typing import Any

import torch


def log_det_jacobian_step(alpha: float, n_coords: int) -> torch.Tensor:
    """Per-step log |det J| for the affine map in :math:`\\varepsilon` (3M dims)."""
    d = 3 * n_coords
    return d * torch.log(torch.abs(torch.tensor(1.0 + alpha)) + 1e-30)


def log_gaussian_isotropic(eps: torch.Tensor, variance: float) -> torch.Tensor:
    """Log density of isotropic Gaussian for a tensor ``eps`` (any shape)."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    flat = eps.reshape(eps.shape[0], -1)
    d = flat.shape[1]
    quad = (flat**2).sum(dim=-1)
    return -0.5 * quad / variance - 0.5 * d * (math.log(2 * math.pi * variance))


def log_prior_initial(atom_coords: torch.Tensor, sigma0: float) -> torch.Tensor:
    """Log density of N(0, sigma0^2 I) for initial noise coordinates."""
    v = sigma0**2
    return log_gaussian_isotropic(atom_coords, v)


def compute_log_path_prob(
    eps_list: list[torch.Tensor],
    meta_list: list[dict[str, Any]],
    initial_coords: torch.Tensor | None = None,
    sigma0: float | None = None,
    include_jacobian: bool = True,
    n_atoms: int | None = None,
) -> torch.Tensor:
    """Sum log p over stochastic draws (noise terms) and optional Jacobian factors.

    Haar and translation densities are omitted (constant w.r.t. state; cancel in
    ratios). If ``initial_coords`` and ``sigma0`` are given, adds log rho(x_0).
    """
    device = eps_list[0].device if eps_list else torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)
    for eps, meta in zip(eps_list, meta_list):
        v = float(meta["noise_var"])
        if v <= 1e-30:
            continue
        total = total + log_gaussian_isotropic(eps, v).sum()
        if include_jacobian:
            t_hat = float(meta["t_hat"])
            sigma_t = float(meta["sigma_t"])
            step_scale = float(meta["step_scale"])
            alpha = step_scale * (sigma_t - t_hat) / t_hat
            m = n_atoms if n_atoms is not None else eps.shape[1]
            total = total + log_det_jacobian_step(alpha, m).to(device)
    if initial_coords is not None and sigma0 is not None:
        total = total + log_prior_initial(initial_coords, sigma0).sum()
    return total


def acceptance_ratio_fixed_length(
    log_p_new: float,
    log_p_old: float,
    reactive: bool,
) -> float:
    """Metropolis acceptance min(1, exp(log_p_old - log_p_new)) * H_AB."""
    if not reactive:
        return 0.0
    return min(1.0, math.exp(log_p_old - log_p_new))

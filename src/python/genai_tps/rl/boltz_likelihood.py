"""Boltz-specific importance weights for trajectory replay (DDPO-IS-style surrogate).

We treat the **denoiser velocity** :math:`v = (x_{\\mathrm{noisy}} - D_\\theta)/\\hat t`
as the mean of a Gaussian policy with fixed diagonal variance :math:`\\tau^2` on masked
atom coordinates. Given a rollout velocity :math:`v_{\\mathrm{old}}` (from the sampling
policy) and a replay velocity :math:`v_{\\mathrm{new}}` (from the current parameters),
use

.. math::

    \\log w = -\\frac{1}{2\\tau^2}\\|v_{\\mathrm{old}} - v_{\\mathrm{new}}\\|^2.

At :math:`v_{\\mathrm{new}} = v_{\\mathrm{old}}`, :math:`w = 1`. This is a **surrogate**
for the true induced density of :math:`x_{t+1}` given :math:`x_t`; it provides a
non-trivial importance ratio and gradients into the score network through
:math:`v_{\\mathrm{new}}`.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from genai_tps.backends.boltz.gpu_core import _kwargs_for_preconditioned_forward

__all__ = [
    "denoiser_velocity_importance_weight",
    "replay_denoiser",
    "forward_step_meta_tensor",
]


def forward_step_meta_tensor(core: Any, step_idx: int) -> dict[str, Any]:
    """Schedule metadata aligned with :func:`genai_tps.backends.boltz.path_probability.forward_step_meta`."""
    from genai_tps.backends.boltz.path_probability import forward_step_meta

    return forward_step_meta(core, step_idx)


def replay_denoiser(
    core: Any,
    x_noisy: torch.Tensor,
    step_idx: int,
) -> torch.Tensor:
    """Run ``preconditioned_network_forward`` at fixed ``x_noisy`` (trainable path)."""
    sch = core.schedule[step_idx]
    t_hat = sch.t_hat
    b = x_noisy.shape[0]
    sample_ids = torch.arange(b, device=x_noisy.device)
    kw = _kwargs_for_preconditioned_forward(dict(core.network_condition_kwargs))
    kw.pop("multiplicity", None)
    kwargs = dict(multiplicity=sample_ids.numel(), **kw)
    diffusion = core.diffusion
    if core.inference_dtype is not None and x_noisy.device.type == "cuda":
        with torch.autocast("cuda", dtype=core.inference_dtype):
            out = diffusion.preconditioned_network_forward(
                x_noisy, t_hat, network_condition_kwargs=kwargs
            )
        out = out.to(x_noisy.dtype)
    else:
        out = diffusion.preconditioned_network_forward(
            x_noisy, t_hat, network_condition_kwargs=kwargs
        )
    return out


def denoiser_velocity_importance_weight(
    x_noisy: torch.Tensor,
    denoised_old: torch.Tensor,
    denoised_new: torch.Tensor,
    t_hat: float | torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    tau_sq: float,
    log_clip: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(importance_weight, log_p_theta_minus_log_old, quad_diff)``.

    All tensors are batched ``(B, M, 3)``. ``atom_mask`` is ``(B, M)`` or ``(M,)`` bool/float.
    """
    if float(tau_sq) <= 0:
        raise ValueError("tau_sq must be positive")
    t_hat_t = torch.as_tensor(t_hat, device=x_noisy.device, dtype=x_noisy.dtype)
    v_old = (x_noisy.detach() - denoised_old) / t_hat_t
    v_new = (x_noisy - denoised_new) / t_hat_t
    m = atom_mask
    if m.dim() == 1:
        m = m.unsqueeze(0).expand(x_noisy.shape[0], -1)
    m3 = m.unsqueeze(-1).to(dtype=x_noisy.dtype)
    diff = (v_old - v_new) * m3
    # Sum over masked coords; count masked elements per batch for normalization
    quad = (diff**2).sum(dim=(1, 2))
    n_mask = (m3 > 0).to(diff.dtype).sum(dim=(1, 2)) * 3.0
    n_mask = torch.clamp(n_mask, min=1.0)
    # Average per-dimension quadratic (stabilizes scale with system size)
    quad_mean = quad / n_mask
    log_ratio = -0.5 * quad_mean / float(tau_sq)
    log_ratio = torch.clamp(log_ratio, -log_clip, log_clip)
    w = torch.exp(log_ratio)
    w = torch.nan_to_num(w, nan=0.0, posinf=math.exp(log_clip), neginf=0.0)
    return w, log_ratio, quad_mean

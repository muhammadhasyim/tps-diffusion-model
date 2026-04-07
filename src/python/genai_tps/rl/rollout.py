"""Collect Boltz diffusion trajectories for offline RL replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["BoltzRolloutStep", "rollout_forward_trajectory"]


@dataclass
class BoltzRolloutStep:
    """Tensors for one denoising step (CPU or GPU); detach before long storage."""

    x_prev: torch.Tensor
    eps: torch.Tensor
    random_r: torch.Tensor
    random_tr: torch.Tensor
    center_mean: torch.Tensor
    x_noisy: torch.Tensor
    denoised_old: torch.Tensor
    x_next: torch.Tensor
    step_idx: int
    t_hat: float
    noise_var: float
    sigma_t: float
    step_scale: float


def rollout_forward_trajectory(
    core: Any,
    x0: torch.Tensor | None = None,
    *,
    num_steps: int | None = None,
) -> list[BoltzRolloutStep]:
    """Run forward sampling under ``torch.inference_mode`` via :meth:`single_forward_step` internals.

    Uses :meth:`BoltzSamplerCore._single_forward_step_core` once per step (no double
    network evaluation).
    """
    if core._schedule is None:
        core.build_schedule()
    n = int(num_steps if num_steps is not None else core.num_sampling_steps)
    x = x0 if x0 is not None else core.sample_initial_noise()
    steps: list[BoltzRolloutStep] = []
    with torch.inference_mode():
        for step_idx in range(n):
            x_next, eps, rr, tr, meta, denoised, x_noisy = core._single_forward_step_core(
                x, step_idx, eps=None
            )
            steps.append(
                BoltzRolloutStep(
                    x_prev=x.detach(),
                    eps=eps.detach(),
                    random_r=rr.detach(),
                    random_tr=tr.detach(),
                    center_mean=meta["center_mean"].detach(),
                    x_noisy=x_noisy.detach(),
                    denoised_old=denoised.detach(),
                    x_next=x_next.detach(),
                    step_idx=step_idx,
                    t_hat=float(meta["t_hat"]),
                    noise_var=float(meta["noise_var"]),
                    sigma_t=float(meta["sigma_t"]),
                    step_scale=float(meta["step_scale"]),
                )
            )
            x = x_next
    return steps

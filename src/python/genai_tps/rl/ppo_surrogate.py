# Derived from RLDiff (MIT, University of Oxford, 2024).
# Original: RLDiff/utils/train_utils.py::compute_loss and reward normalization helpers.
# Repository: https://github.com/oxpig/RLDiff
# Paper: Broster et al., bioRxiv 2026, DOI 10.64898/2026.03.25.714128

from __future__ import annotations

import torch

__all__ = ["compute_ppo_loss", "normalize_rewards_per_trajectory"]


def normalize_rewards_per_trajectory(raw_rewards: list[float], eps: float = 1e-6) -> tuple[list[float], float, float]:
    """Z-score rewards for one trajectory (RLDiff ``_normalize_with_baseline`` pattern)."""
    if not raw_rewards:
        return [], 0.0, 1.0
    import numpy as np

    arr = np.asarray(raw_rewards, dtype=np.float64)
    mean = float(arr.mean())
    std = max(float(arr.std()), eps)
    z = ((arr - mean) / std).tolist()
    return z, mean, std


def compute_ppo_loss(
    importance_weight: torch.Tensor,
    reward: torch.Tensor,
    *,
    clip_range: float,
    step_idx: int,
    no_early_step_guidance: bool,
    alpha_step: int | None,
    early_step_cap: float = 0.03,
    early_step_end: int = 10,
) -> torch.Tensor:
    """PPO-style clipped surrogate loss on ``-min(unclipped, clipped)``.

    Parameters
    ----------
    importance_weight
        Shape ``(1,)`` or broadcastable scalar tensor (must allow gradients if training).
    reward
        Scalar or broadcastable advantage / return for this step.
    step_idx
        Diffusion step index (0 .. T-1) for early-step loss capping.
    """
    unclipped = importance_weight * reward
    clipped_weights = torch.clamp(importance_weight, 1.0 - clip_range, 1.0 + clip_range)
    clipped = clipped_weights * reward
    loss = -torch.minimum(unclipped, clipped)

    if not no_early_step_guidance:
        cap_boundary = alpha_step if alpha_step is not None else early_step_end
        if step_idx < cap_boundary:
            cap = torch.as_tensor(early_step_cap, device=loss.device, dtype=loss.dtype)
            loss_abs = loss.abs()
            scale = torch.clamp(cap / (loss_abs + 1e-12), max=1.0)
            loss = loss * scale

    return loss

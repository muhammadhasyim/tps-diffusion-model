"""Offline RL update from stored Boltz rollouts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from genai_tps.rl.boltz_likelihood import denoiser_velocity_importance_weight, replay_denoiser

from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
from genai_tps.rl.ppo_surrogate import compute_ppo_loss
from genai_tps.rl.rollout import BoltzRolloutStep

__all__ = ["fes_guided_trajectory_loss", "replay_trajectory_loss"]


def replay_trajectory_loss(
    core: Any,
    trajectory: Sequence[BoltzRolloutStep],
    per_step_reward: torch.Tensor,
    *,
    cfg: BoltzRLConfig,
) -> torch.Tensor:
    """Sum PPO surrogate losses over steps (same scalar reward broadcast to each step).

    ``per_step_reward`` shape ``(1,)`` or scalar; typically normalized terminal return
    shared across diffusion steps (RLDiff-style decay can be added later).
    """
    if not trajectory:
        return torch.zeros((), device=core.diffusion.device, dtype=torch.float32)
    device = core.diffusion.device
    atom_mask = core.atom_mask
    total = torch.zeros((), device=device, dtype=torch.float32)
    r = per_step_reward if per_step_reward.dim() > 0 else per_step_reward.unsqueeze(0)
    r = r.to(device=device, dtype=torch.float32)

    for st in trajectory:
        x_noisy_in = st.x_noisy.to(device=device, dtype=torch.float32)
        denoised_old = st.denoised_old.to(device=device, dtype=torch.float32)

        # Replay denoiser at fixed ``x_noisy`` (do not resample SE(3) aug — rollout aug differs each call).
        denoised_new = replay_denoiser(core, x_noisy_in, st.step_idx)
        w, _, _ = denoiser_velocity_importance_weight(
            x_noisy_in,
            denoised_old,
            denoised_new,
            st.t_hat,
            atom_mask,
            tau_sq=cfg.velocity_log_prob_tau_sq,
            log_clip=cfg.log_importance_clip,
        )
        w_s = w.mean() if w.dim() > 0 else w
        r_s = r.mean() if r.dim() > 0 else r
        step_loss = compute_ppo_loss(
            w_s,
            r_s,
            clip_range=cfg.clip_range,
            step_idx=st.step_idx,
            no_early_step_guidance=cfg.no_early_step_guidance,
            alpha_step=cfg.alpha_step,
            early_step_cap=cfg.early_step_cap,
            early_step_end=cfg.early_step_end,
        )
        total = total + step_loss
    return total


def fes_guided_trajectory_loss(
    core: Any,
    trajectory: Sequence[BoltzRolloutStep],
    cv_np: np.ndarray,
    teacher: Any,
    student_kde: Any,
    *,
    fes_cfg: FESTeacherConfig,
    rl_cfg: BoltzRLConfig,
) -> torch.Tensor:
    """PPO replay loss with terminal advantage ``clip(log p_target - log p_Boltz)``.

    The caller should update *student_kde* with *cv_np* **before** calling this
    function if the intended semantics match the training-loop recipe (KDE
    includes the current sample when evaluating ``log_density``).

    Parameters
    ----------
    core:
        :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore` instance.
    trajectory:
        Rollout steps from :func:`~genai_tps.rl.rollout.rollout_forward_trajectory`.
    cv_np:
        Collective-variable vector at the terminal frame, shape ``(D,)``.
    teacher:
        Object with ``log_p_target(cv) -> float`` (e.g. :class:`OpenMMTeacher`).
    student_kde:
        :class:`~genai_tps.rl.student_distribution.BoltzStudentKDE` instance.
    fes_cfg:
        Advantage clipping and related FES hyper-parameters.
    rl_cfg:
        Standard Boltz RL / PPO hyper-parameters.
    """
    if not trajectory:
        return torch.zeros((), device=core.diffusion.device, dtype=torch.float32)
    device = core.diffusion.device
    log_pt = float(teacher.log_p_target(cv_np))
    log_pb = float(student_kde.log_density(cv_np))
    adv = float(np.clip(log_pt - log_pb, -fes_cfg.advantage_clip, fes_cfg.advantage_clip))
    reward = torch.tensor([adv], device=device, dtype=torch.float32)
    return replay_trajectory_loss(core, trajectory, reward, cfg=rl_cfg)

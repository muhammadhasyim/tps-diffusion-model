"""Tests for RLDiff-style Boltz RL utilities (``genai_tps.rl``)."""

from __future__ import annotations

import pytest
import torch

from genai_tps.backends.boltz.path_probability import forward_step_meta
from genai_tps.rl.boltz_likelihood import denoiser_velocity_importance_weight, forward_step_meta_tensor
from genai_tps.rl.config import BoltzRLConfig
from genai_tps.rl.ppo_surrogate import compute_ppo_loss, normalize_rewards_per_trajectory


def test_normalize_rewards_single_preserves_signal():
    """One reward: avoid (r-mean)/eps == 0; return raw value with unit std."""
    z, mean, std = normalize_rewards_per_trajectory([0.73])
    assert z[0] == pytest.approx(0.73)
    assert mean == 0.0
    assert std == 1.0


def test_normalize_rewards_multi_zscore():
    z, mean, std = normalize_rewards_per_trajectory([0.0, 2.0])
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(1.0)
    assert z[0] == pytest.approx(-1.0)
    assert z[1] == pytest.approx(1.0)


def test_compute_ppo_loss_reference_scalar():
    """Hand-checked PPO surrogate (RLDiff ``compute_loss`` pattern)."""
    w = torch.tensor(1.5, dtype=torch.float32)
    r = torch.tensor(0.8, dtype=torch.float32)
    clip = 0.2
    unclipped = w * r
    clipped_w = torch.clamp(w, 1.0 - clip, 1.0 + clip)
    clipped = clipped_w * r
    expected = -torch.minimum(unclipped, clipped)
    out = compute_ppo_loss(
        w,
        r,
        clip_range=clip,
        step_idx=100,
        no_early_step_guidance=True,
        alpha_step=None,
    )
    assert torch.allclose(out, expected)


def test_forward_step_meta_tensor_matches_path_probability():
    """``forward_step_meta_tensor`` delegates to path_probability (schedule agreement)."""

    class _Sch:
        def __init__(self) -> None:
            self.sigma_tm = 80.0
            self.sigma_t = 70.0
            self.t_hat = 75.0
            self.noise_var = 4.0

    class _Diff:
        step_scale = 1.2

    class _Core:
        schedule = [_Sch()]
        diffusion = _Diff()

    core = _Core()
    d1 = forward_step_meta(core, 0)
    d2 = forward_step_meta_tensor(core, 0)
    assert d1.keys() == d2.keys()
    assert float(d1["t_hat"]) == float(d2["t_hat"])
    assert float(d1["noise_var"]) == float(d2["noise_var"])


def test_denoiser_velocity_importance_weight_gradient():
    """Gradients flow into ``denoised_new``; weight is 1.0 when new matches old."""
    x_noisy = torch.zeros(1, 5, 3, dtype=torch.float32)
    denoised_old = torch.ones(1, 5, 3, dtype=torch.float32)
    denoised_new = torch.ones(1, 5, 3, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.float32)
    w, log_ratio, _ = denoiser_velocity_importance_weight(
        x_noisy,
        denoised_old,
        denoised_new,
        t_hat=2.0,
        atom_mask=mask,
        tau_sq=0.5,
        log_clip=20.0,
    )
    assert w.shape == (1,)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5, rtol=1e-4)
    loss = w.mean()
    loss.backward()
    assert denoised_new.grad is not None

    denoised_new2 = torch.zeros(1, 5, 3, dtype=torch.float32, requires_grad=True)
    w2, _, _ = denoiser_velocity_importance_weight(
        x_noisy,
        denoised_old,
        denoised_new2,
        t_hat=2.0,
        atom_mask=mask,
        tau_sq=0.5,
        log_clip=20.0,
    )
    assert float(w2.mean()) < 1.0
    loss2 = w2.mean()
    loss2.backward()
    assert denoised_new2.grad is not None


def test_boltz_rl_config_frozen():
    cfg = BoltzRLConfig()
    with pytest.raises(Exception):
        cfg.learning_rate = 1.0  # type: ignore[misc]

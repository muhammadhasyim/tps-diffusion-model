"""Path probability and acceptance helper tests (no Boltz model)."""

import math

import pytest
import torch

from genai_tps.backends.boltz.path_probability import (
    acceptance_ratio_fixed_length,
    compute_log_path_prob,
    log_det_jacobian_step,
    log_gaussian_isotropic,
)


def test_log_gaussian_isotropic_matches_manual():
    torch.manual_seed(0)
    eps = torch.randn(2, 5, 3)
    v = 0.7
    got = log_gaussian_isotropic(eps, v).sum()
    flat = eps.reshape(eps.shape[0], -1)
    d = flat.shape[1]
    expect = (-0.5 * (flat**2).sum(dim=-1) / v - 0.5 * d * math.log(2 * math.pi * v)).sum()
    assert torch.allclose(got, expect)


def test_jacobian_logdet_scalar():
    n = 4
    alpha = 0.5
    d = 3 * n
    expect = d * math.log(abs(1.0 + alpha))
    got = float(log_det_jacobian_step(alpha, n))
    assert abs(got - expect) < 1e-5


def test_path_prob_linear_in_noise():
    eps1 = [torch.ones(1, 2, 3) * 0.1, torch.ones(1, 2, 3) * 0.2]
    meta = [
        {"noise_var": 1.0, "t_hat": 2.0, "sigma_t": 1.0, "step_scale": 1.5},
        {"noise_var": 2.0, "t_hat": 1.5, "sigma_t": 0.5, "step_scale": 1.5},
    ]
    lp = compute_log_path_prob(eps1, meta, include_jacobian=False)
    eps2 = [e * 2 for e in eps1]
    lp2 = compute_log_path_prob(eps2, meta, include_jacobian=False)
    assert lp2 < lp


def test_acceptance_ratio():
    acc = acceptance_ratio_fixed_length(-1.0, -2.0, True)
    assert acc == pytest.approx(min(1.0, math.exp(-1.0)))
    assert acceptance_ratio_fixed_length(0.0, 0.0, False) == 0.0

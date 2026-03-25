"""Path probability and acceptance helper tests (no Boltz model)."""

import math

import pytest
import torch

from genai_tps.backends.boltz.path_probability import (
    acceptance_ratio_fixed_length,
    compute_log_path_prob,
    log_det_jacobian_step,
    log_gaussian_isotropic,
    min_metropolis_acceptance_path,
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
    # log_p_old, log_p_new: min(1, exp(log_p_new - log_p_old)) = π_new/π_old capped
    acc = acceptance_ratio_fixed_length(-2.0, -1.0, True)
    assert acc == pytest.approx(min(1.0, math.exp(1.0)))
    acc2 = acceptance_ratio_fixed_length(-1.0, -2.0, True)
    assert acc2 == pytest.approx(min(1.0, math.exp(-1.0)))
    assert min_metropolis_acceptance_path(0.0, 0.0, reactive=False) == 0.0
    assert acceptance_ratio_fixed_length(0.0, 0.0, False) == 0.0


def test_min_metropolis_no_exp_overflow():
    """Large log-density improvement: min(1, exp(d)) is 1 without calling exp(d)."""
    assert min_metropolis_acceptance_path(-1e9, 0.0, reactive=True) == 1.0


# ---------------------------------------------------------------------------
# Consistency: compute_log_path_prob reconstructed from stored noise sums
# ---------------------------------------------------------------------------

def test_path_prob_consistency_with_manual_sum():
    """compute_log_path_prob returns the same value as manual eps log-density sum."""
    import math
    torch.manual_seed(99)
    eps_list = [
        torch.randn(1, 4, 3, dtype=torch.float64) * math.sqrt(2.5),
        torch.randn(1, 4, 3, dtype=torch.float64) * math.sqrt(1.2),
        torch.randn(1, 4, 3, dtype=torch.float64) * math.sqrt(3.0),
    ]
    meta_list = [
        {"noise_var": 2.5, "t_hat": 20.0, "sigma_t": 15.0, "step_scale": 1.5},
        {"noise_var": 1.2, "t_hat": 15.0, "sigma_t": 10.0, "step_scale": 1.5},
        {"noise_var": 3.0, "t_hat": 10.0, "sigma_t": 5.0, "step_scale": 1.5},
    ]
    computed = compute_log_path_prob(eps_list, meta_list, include_jacobian=False)

    # Manual sum: log N(eps; 0, v I) = -||eps||^2/(2v) - (d/2)*log(2*pi*v)
    expected = 0.0
    for eps, meta in zip(eps_list, meta_list):
        v = meta["noise_var"]
        flat = eps.reshape(1, -1)
        d = flat.shape[1]
        expected += float((-0.5 * (flat**2).sum(dim=-1) / v - 0.5 * d * math.log(2 * math.pi * v)).sum())

    assert abs(float(computed) - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {float(computed):.6f}"
    )


def test_path_prob_with_jacobian_increases_for_positive_alpha():
    """For alpha > 0, scalar Jacobian term is positive, increasing log path prob."""
    eps = [torch.ones(1, 2, 3) * 0.1]
    meta = [{"noise_var": 1.0, "t_hat": 2.0, "sigma_t": 3.0, "step_scale": 1.5}]
    lp_no_jac = compute_log_path_prob(eps, meta, include_jacobian=False, n_atoms=2)
    lp_with_jac = compute_log_path_prob(eps, meta, include_jacobian=True, n_atoms=2)
    # alpha = 1.5 * (3.0 - 2.0) / 2.0 = 0.75 > 0 => log|1+0.75| > 0
    assert float(lp_with_jac) > float(lp_no_jac)


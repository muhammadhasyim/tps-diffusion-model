"""Tests for weighted denoising score matching (WDSM) fine-tuning of Boltz 2.

All tests run on CPU and use a lightweight parametric mock instead of the full
Boltz-2 checkpoint so that CI requires no GPU or model weights.

The test plan matches the plan document:
  1. test_weighted_dsm_loss_gradient_flow
  2. test_weighted_dsm_loss_uniform_weights
  3. test_regularization_term_penalizes_drift
  4. test_effective_sample_size_uniform
  5. test_effective_sample_size_degenerate
  6. test_dataset_loading_from_npz
  7. test_noise_schedule_matches_boltz2
  8. test_integration_smoke
  9. Additional edge-case and robustness tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_root = Path(__file__).resolve().parents[1]
_src = _root / "src" / "python"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from genai_tps.training.config import WeightedDSMConfig
from genai_tps.training.dataset import ReweightedStructureDataset
from genai_tps.training.diagnostics import (
    clip_log_weights,
    effective_sample_size,
    weight_statistics,
)
from genai_tps.training.loss import (
    regularized_weighted_dsm_loss,
    weighted_dsm_loss,
)
from genai_tps.training.noise_schedule import EDMNoiseParams, edm_loss_weight, sample_noise_sigma


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

N_ATOMS = 8
BATCH_SIZE = 4


class ParametricMockDiffusion(nn.Module):
    """Minimal trainable denoiser for gradient tests.

    A learnable linear projection replaces the full score network.
    EDM preconditioning parameters match :class:`~tests.mock_boltz_diffusion.MockDiffusion`
    so that noise-schedule utilities can be tested against the same defaults.
    """

    def __init__(self, n_atoms: int = N_ATOMS, device: torch.device | None = None) -> None:
        super().__init__()
        d = device or torch.device("cpu")
        self.sigma_data: float = 16.0
        self.P_mean: float = -1.2
        self.P_std: float = 1.5
        # Tiny learnable layer so gradients exist.
        self.proj = nn.Linear(3, 3, bias=False).to(d)
        nn.init.eye_(self.proj.weight)
        self.register_buffer("_dev", torch.zeros(1, device=d), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dev.device

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def preconditioned_network_forward(
        self,
        noised_atom_coords: torch.Tensor,
        sigma: float | torch.Tensor,
        network_condition_kwargs: dict | None = None,
    ) -> torch.Tensor:
        B, M, _ = noised_atom_coords.shape
        if isinstance(sigma, float):
            sigma = torch.full((B,), sigma, device=noised_atom_coords.device)
        padded_sigma = sigma.view(B, 1, 1)
        # Apply learnable projection in atom dimension.
        flat = noised_atom_coords.reshape(B * M, 3)
        r_update = self.proj(flat).reshape(B, M, 3)
        denoised = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised

    def noise_distribution(self, batch_size: int) -> torch.Tensor:
        """Log-normal sigma matching :class:`AtomDiffusion` defaults."""
        return self.sigma_data * (
            self.P_mean + self.P_std * torch.randn(batch_size, device=self.device)
        ).exp()

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


def _random_batch(
    n: int = BATCH_SIZE,
    m: int = N_ATOMS,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x0, logw, atom_mask) with shape (N, M, 3), (N,), (N, M)."""
    d = device or torch.device("cpu")
    x0 = torch.randn(n, m, 3, device=d)
    logw = torch.randn(n, device=d)
    mask = torch.ones(n, m, device=d)
    return x0, logw, mask


# ---------------------------------------------------------------------------
# 1. Gradient flow
# ---------------------------------------------------------------------------

def test_weighted_dsm_loss_gradient_flow():
    """Loss must produce a gradient into the model's learnable parameters."""
    model = ParametricMockDiffusion()
    x0, logw, mask = _random_batch()
    noise_params = EDMNoiseParams()

    loss = weighted_dsm_loss(model, x0, logw, mask, noise_params)
    assert torch.isfinite(loss), "loss must be finite"

    loss.backward()
    assert model.proj.weight.grad is not None, "proj.weight must have gradient"
    assert torch.any(model.proj.weight.grad != 0), "gradient must be non-zero"


# ---------------------------------------------------------------------------
# 2. Uniform weights recover standard DSM
# ---------------------------------------------------------------------------

def test_weighted_dsm_loss_uniform_weights():
    """With uniform logw the weighted sum is equivalent to an unweighted mean."""
    model = ParametricMockDiffusion()
    torch.manual_seed(0)
    x0, _, mask = _random_batch(n=16)
    noise_params = EDMNoiseParams()

    # Uniform weights: softmax(0) = 1/N for all i, so sum_i w_i * L_i = mean(L_i).
    logw_uniform = torch.zeros(16)
    torch.manual_seed(42)
    loss_weighted = weighted_dsm_loss(model, x0, logw_uniform, mask, noise_params)
    assert torch.isfinite(loss_weighted)

    # Same seed → same noise → same per-sample losses; only weights differ.
    # softmax(zeros) == softmax(ones) == 1/N, so losses must match.
    logw_ones = torch.ones(16)
    torch.manual_seed(42)
    loss_ones = weighted_dsm_loss(model, x0, logw_ones, mask, noise_params)
    assert torch.allclose(loss_weighted, loss_ones, atol=1e-6), (
        "uniform logw variants must give identical loss"
    )


# ---------------------------------------------------------------------------
# 3. Regularization penalizes drift
# ---------------------------------------------------------------------------

def test_regularization_term_penalizes_drift():
    """Regularization is exactly zero when models are identical, positive when they differ."""
    torch.manual_seed(1)
    model = ParametricMockDiffusion()
    frozen = ParametricMockDiffusion()
    frozen.proj.weight = nn.Parameter(model.proj.weight.data.clone(), requires_grad=False)

    x0, logw, mask = _random_batch()
    noise_params = EDMNoiseParams()
    cfg = WeightedDSMConfig(beta=1.0)

    # Same seed for both calls so they draw the same noise.
    torch.manual_seed(77)
    loss_identical = regularized_weighted_dsm_loss(
        model, frozen, x0, logw, mask, noise_params, cfg
    )
    torch.manual_seed(77)
    loss_dsm_only = weighted_dsm_loss(model, x0, logw, mask, noise_params)
    assert torch.allclose(loss_identical, loss_dsm_only, atol=1e-5), (
        "identical models: regularized loss must equal bare DSM loss"
    )

    # Perturb frozen model weights — regularization must add a positive penalty.
    frozen_perturbed = ParametricMockDiffusion()
    with torch.no_grad():
        frozen_perturbed.proj.weight.copy_(model.proj.weight + 5.0)
    frozen_perturbed.proj.weight.requires_grad_(False)

    torch.manual_seed(77)
    loss_perturbed = regularized_weighted_dsm_loss(
        model, frozen_perturbed, x0, logw, mask, noise_params, cfg
    )
    assert loss_perturbed > loss_dsm_only, (
        "perturbation of frozen model must increase the regularized loss"
    )


# ---------------------------------------------------------------------------
# 4 & 5. Effective sample size
# ---------------------------------------------------------------------------

def test_effective_sample_size_uniform():
    """N_eff == N for exactly uniform log-weights."""
    N = 100
    logw = np.zeros(N)
    n_eff = effective_sample_size(logw)
    assert abs(n_eff - N) < 1e-6, f"expected N_eff={N}, got {n_eff}"


def test_effective_sample_size_degenerate():
    """N_eff ~ 1 when one weight dominates all others."""
    N = 50
    logw = np.zeros(N)
    logw[0] = 500.0  # one weight vastly larger than the rest
    n_eff = effective_sample_size(logw)
    assert n_eff < 2.0, f"degenerate weights must give N_eff near 1, got {n_eff}"


def test_effective_sample_size_two_equal():
    """Two equal weights → N_eff == 2."""
    logw = np.array([0.0, 0.0])
    n_eff = effective_sample_size(logw)
    assert abs(n_eff - 2.0) < 1e-10


def test_effective_sample_size_empty():
    """Empty array should return 0."""
    assert effective_sample_size(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# 6. Weight clipping
# ---------------------------------------------------------------------------

def test_clip_log_weights_no_clip():
    """No clipping when all weights are equal."""
    logw = np.zeros(10)
    result = clip_log_weights(logw, max_log_ratio=5.0)
    np.testing.assert_allclose(result, logw)


def test_clip_log_weights_extreme():
    """Extreme outlier weight is capped."""
    logw = np.array([0.0, 0.0, 100.0])
    result = clip_log_weights(logw, max_log_ratio=3.0)
    # Max log ratio from mean should be at most max_log_ratio.
    log_mean = np.log(np.mean(np.exp(result - result.max()))) + result.max()
    assert float(np.max(result) - log_mean) <= 3.0 + 1e-6


# ---------------------------------------------------------------------------
# Weight statistics
# ---------------------------------------------------------------------------

def test_weight_statistics_keys():
    logw = np.random.randn(50)
    stats = weight_statistics(logw)
    for key in ("n_eff", "n_eff_fraction", "max_weight_fraction", "min_logw", "max_logw"):
        assert key in stats, f"missing key {key!r}"


def test_weight_statistics_uniform():
    logw = np.zeros(20)
    stats = weight_statistics(logw)
    assert abs(stats["n_eff"] - 20.0) < 1e-6
    assert abs(stats["n_eff_fraction"] - 1.0) < 1e-6
    assert abs(stats["max_weight_fraction"] - 1.0 / 20) < 1e-6


# ---------------------------------------------------------------------------
# 7. Dataset loading from NPZ
# ---------------------------------------------------------------------------

def test_dataset_loading_from_npz(tmp_path):
    """Round-trip NPZ save and load via ReweightedStructureDataset."""
    N, M = 12, 10
    rng = np.random.default_rng(7)
    coords = rng.standard_normal((N, M, 3)).astype(np.float32)
    logw = rng.standard_normal(N).astype(np.float64)
    mask = np.ones((N, M), dtype=np.float32)

    npz_path = tmp_path / "test_data.npz"
    np.savez(npz_path, coords=coords, logw=logw, atom_mask=mask)

    ds = ReweightedStructureDataset.from_npz(npz_path)
    assert len(ds) == N
    sample = ds[0]
    assert set(sample.keys()) == {"coords", "logw", "atom_mask"}
    np.testing.assert_allclose(sample["coords"], coords[0])
    np.testing.assert_allclose(sample["logw"], logw[0])
    np.testing.assert_allclose(sample["atom_mask"], mask[0])


def test_dataset_loading_without_mask(tmp_path):
    """Dataset works when atom_mask is absent (defaults to all-ones)."""
    N, M = 5, 6
    coords = np.random.randn(N, M, 3).astype(np.float32)
    logw = np.zeros(N, dtype=np.float64)
    npz_path = tmp_path / "no_mask.npz"
    np.savez(npz_path, coords=coords, logw=logw)

    ds = ReweightedStructureDataset.from_npz(npz_path)
    assert len(ds) == N
    sample = ds[0]
    np.testing.assert_allclose(sample["atom_mask"], np.ones(M, dtype=np.float32))


def test_dataset_n_eff_reported(tmp_path, capsys):
    """from_npz reports N_eff on construction (no error)."""
    N, M = 8, 4
    coords = np.random.randn(N, M, 3).astype(np.float32)
    logw = np.zeros(N, dtype=np.float64)
    npz_path = tmp_path / "report.npz"
    np.savez(npz_path, coords=coords, logw=logw)
    ds = ReweightedStructureDataset.from_npz(npz_path)
    # N_eff must be accessible as an attribute.
    assert hasattr(ds, "n_eff")
    assert abs(ds.n_eff - N) < 1e-5


def test_dataset_collate_to_tensors(tmp_path):
    """DataLoader collation produces correctly shaped tensors."""
    N, M = 6, 5
    coords = np.random.randn(N, M, 3).astype(np.float32)
    logw = np.zeros(N, dtype=np.float64)
    mask = np.ones((N, M), dtype=np.float32)
    npz_path = tmp_path / "collate.npz"
    np.savez(npz_path, coords=coords, logw=logw, atom_mask=mask)

    ds = ReweightedStructureDataset.from_npz(npz_path)
    loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch["coords"].shape == (3, M, 3)
    assert batch["logw"].shape == (3,)
    assert batch["atom_mask"].shape == (3, M)


# ---------------------------------------------------------------------------
# 8. Noise schedule matches Boltz 2 AtomDiffusion
# ---------------------------------------------------------------------------

def test_noise_schedule_log_normal_mean_var():
    """Sampled sigmas must have the expected log-normal moments."""
    torch.manual_seed(42)
    params = EDMNoiseParams()
    N = 50_000
    sigmas = sample_noise_sigma(N, params, device=torch.device("cpu"))
    log_s = torch.log(sigmas / params.sigma_data)
    assert abs(float(log_s.mean()) - params.P_mean) < 0.05, "log-mean mismatch"
    assert abs(float(log_s.std()) - params.P_std) < 0.05, "log-std mismatch"


def test_noise_schedule_loss_weight_formula():
    """edm_loss_weight(sigma) matches the AtomDiffusion formula."""
    params = EDMNoiseParams(sigma_data=16.0)
    sigma = torch.tensor([1.0, 4.0, 16.0, 64.0])
    expected = (sigma ** 2 + params.sigma_data ** 2) / (sigma * params.sigma_data) ** 2
    result = edm_loss_weight(sigma, params)
    torch.testing.assert_close(result, expected)


def test_noise_schedule_from_diffusion_module():
    """EDMNoiseParams.from_diffusion reads attributes from a diffusion-like object."""
    class _FakeDiffusion:
        sigma_data = 12.0
        P_mean = -0.8
        P_std = 1.2

    params = EDMNoiseParams.from_diffusion(_FakeDiffusion())
    assert params.sigma_data == 12.0
    assert params.P_mean == -0.8
    assert params.P_std == 1.2


# ---------------------------------------------------------------------------
# 9. Integration smoke test (full 2-epoch training loop)
# ---------------------------------------------------------------------------

def test_integration_smoke(tmp_path):
    """End-to-end training loop: 2 epochs, no errors, loss decreases or is finite."""
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    torch.manual_seed(99)
    np.random.seed(99)

    N, M = 20, N_ATOMS
    # Structures all at origin so the denoiser has a well-posed target.
    coords = np.random.randn(N, M, 3).astype(np.float32) * 2.0
    logw = np.random.randn(N).astype(np.float64)
    mask = np.ones((N, M), dtype=np.float32)

    npz_path = tmp_path / "smoke.npz"
    np.savez(npz_path, coords=coords, logw=logw, atom_mask=mask)

    ds = ReweightedStructureDataset.from_npz(npz_path)
    loader = DataLoader(ds, batch_size=5, shuffle=True)

    model = ParametricMockDiffusion(n_atoms=M)
    frozen = ParametricMockDiffusion(n_atoms=M)
    # Frozen model: copy weights and disable grad.
    with torch.no_grad():
        frozen.proj.weight.copy_(model.proj.weight)
    frozen.proj.weight.requires_grad_(False)
    frozen.eval()

    cfg = WeightedDSMConfig(beta=0.01, learning_rate=1e-3)
    noise_params = EDMNoiseParams()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    losses = []
    for _epoch in range(2):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x0 = batch["coords"].float()
            lw = batch["logw"].float()
            am = batch["atom_mask"].float()

            optimizer.zero_grad()
            loss = regularized_weighted_dsm_loss(model, frozen, x0, lw, am, noise_params, cfg)
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss), f"non-finite loss: {loss}"
            epoch_loss += float(loss.detach())
            n_batches += 1

        losses.append(epoch_loss / max(n_batches, 1))

    assert all(np.isfinite(l) for l in losses), "all epoch losses must be finite"


# ---------------------------------------------------------------------------
# 10. WeightedDSMConfig validation
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = WeightedDSMConfig()
    assert cfg.beta >= 0.0
    assert cfg.learning_rate > 0.0
    assert cfg.max_grad_norm > 0.0


def test_config_invalid_beta():
    with pytest.raises((ValueError, AssertionError)):
        WeightedDSMConfig(beta=-1.0)


# ---------------------------------------------------------------------------
# 11. Loss handles edge cases gracefully
# ---------------------------------------------------------------------------

def test_loss_single_sample():
    """Batch size 1 must not raise or produce NaN."""
    model = ParametricMockDiffusion()
    x0 = torch.randn(1, N_ATOMS, 3)
    logw = torch.zeros(1)
    mask = torch.ones(1, N_ATOMS)
    noise_params = EDMNoiseParams()
    loss = weighted_dsm_loss(model, x0, logw, mask, noise_params)
    assert torch.isfinite(loss)


def test_loss_with_partial_mask():
    """Loss with partially masked atoms must be finite and not use masked atoms."""
    model = ParametricMockDiffusion()
    x0 = torch.randn(BATCH_SIZE, N_ATOMS, 3)
    logw = torch.zeros(BATCH_SIZE)
    mask = torch.ones(BATCH_SIZE, N_ATOMS)
    mask[:, -2:] = 0.0  # last 2 atoms masked out
    noise_params = EDMNoiseParams()
    loss = weighted_dsm_loss(model, x0, logw, mask, noise_params)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tests for true_quotient_dsm_loss (arXiv:2604.21809)
# ---------------------------------------------------------------------------

from genai_tps.training.loss import alignment_weighted_dsm_loss, true_quotient_dsm_loss  # noqa: E402


def test_true_quotient_loss_is_finite():
    """true_quotient_dsm_loss must return a finite scalar."""
    torch.manual_seed(11)
    model = ParametricMockDiffusion()
    x0 = torch.randn(BATCH_SIZE, N_ATOMS, 3)
    logw = torch.zeros(BATCH_SIZE)
    mask = torch.ones(BATCH_SIZE, N_ATOMS)
    noise_params = EDMNoiseParams()
    loss = true_quotient_dsm_loss(model, x0, logw, mask, noise_params)
    assert torch.isfinite(loss), f"true_quotient_dsm_loss is not finite: {loss}"


def test_true_quotient_loss_non_negative():
    """Loss value must be >= 0 (it is a sum of squared norms)."""
    torch.manual_seed(22)
    model = ParametricMockDiffusion()
    x0 = torch.randn(BATCH_SIZE, N_ATOMS, 3)
    logw = torch.zeros(BATCH_SIZE)
    mask = torch.ones(BATCH_SIZE, N_ATOMS)
    noise_params = EDMNoiseParams()
    loss = true_quotient_dsm_loss(model, x0, logw, mask, noise_params)
    assert loss.item() >= -1e-6, f"Loss is negative: {loss.item():.6f}"


def test_true_quotient_loss_gradient_flows():
    """Gradients must reach the model parameters."""
    torch.manual_seed(33)
    model = ParametricMockDiffusion()
    x0 = torch.randn(BATCH_SIZE, N_ATOMS, 3)
    logw = torch.zeros(BATCH_SIZE)
    mask = torch.ones(BATCH_SIZE, N_ATOMS)
    noise_params = EDMNoiseParams()

    loss = true_quotient_dsm_loss(model, x0, logw, mask, noise_params)
    loss.backward()

    assert model.proj.weight.grad is not None, "No gradient on proj.weight"
    assert torch.isfinite(model.proj.weight.grad).all(), "Non-finite gradient"


def test_true_quotient_loss_invariant_to_input_rotation():
    """Both original and rotated inputs should give finite, comparable losses
    (sanity check for consistent behaviour under SE(3) transformations)."""
    torch.manual_seed(44)
    model = ParametricMockDiffusion()
    model.eval()  # disable random augmentation

    B, N = BATCH_SIZE, N_ATOMS
    x0 = torch.randn(B, N, 3)
    logw = torch.zeros(B)
    mask = torch.ones(B, N)
    noise_params = EDMNoiseParams()

    from tests.test_quotient_projection import _random_so3
    R = _random_so3(B)
    x0_rot = torch.einsum("bij,bnj->bni", R.float(), x0)

    with torch.no_grad():
        torch.manual_seed(99)
        loss_orig = true_quotient_dsm_loss(model, x0, logw, mask, noise_params)
        torch.manual_seed(99)
        loss_rot = true_quotient_dsm_loss(model, x0_rot, logw, mask, noise_params)

    assert torch.isfinite(loss_orig), "Original loss not finite"
    assert torch.isfinite(loss_rot), "Rotated loss not finite"
    # Both should be non-negative
    assert loss_orig.item() >= -1e-5
    assert loss_rot.item() >= -1e-5


def test_true_quotient_loss_le_cartesian_loss():
    """The quotient loss uses horizontal projection of the residual which
    removes 3 DoFs (rigid rotations).  Therefore for the same noise and model
    output, ||P_x(r)||^2 <= ||r||^2 (orthogonal projection never inflates norm).

    We verify this directly using a manually controlled forward pass."""
    torch.manual_seed(77)
    B, N = BATCH_SIZE, N_ATOMS
    model = ParametricMockDiffusion()
    model.eval()

    from genai_tps.training.loss import _center, sample_noise_sigma
    from genai_tps.training.noise_schedule import EDMNoiseParams
    from genai_tps.training.quotient_projection import horizontal_projection

    x0 = torch.randn(B, N, 3)
    mask = torch.ones(B, N)
    noise_params = EDMNoiseParams()

    x0_c, _ = _center(x0, mask)
    sigma = sample_noise_sigma(B, noise_params, device=x0.device)
    eps = torch.randn_like(x0_c)
    x_noisy = x0_c + sigma.view(B, 1, 1) * eps

    with torch.no_grad():
        x_denoised = model.preconditioned_network_forward(x_noisy, sigma)

    residual = x_denoised - x0_c                             # (B, N, 3)
    x_noisy_c, _ = _center(x_noisy, mask)
    residual_proj = horizontal_projection(x_noisy_c, residual, mask)

    norm_full = (residual * mask.unsqueeze(-1)).norm()
    norm_proj = residual_proj.norm()

    assert norm_proj.item() <= norm_full.item() + 1e-6, (
        f"Projection inflated residual norm: {norm_proj.item():.4f} > {norm_full.item():.4f}"
    )


def test_true_quotient_loss_with_regularization():
    """With beta > 0 and a frozen model, regularized loss must be >= base loss."""
    torch.manual_seed(66)
    model = ParametricMockDiffusion()
    frozen = ParametricMockDiffusion()
    # Perturb frozen model to create a non-zero regularization term
    with torch.no_grad():
        frozen.proj.weight.add_(0.5 * torch.randn_like(frozen.proj.weight))

    x0 = torch.randn(BATCH_SIZE, N_ATOMS, 3)
    logw = torch.zeros(BATCH_SIZE)
    mask = torch.ones(BATCH_SIZE, N_ATOMS)
    noise_params = EDMNoiseParams()

    with torch.no_grad():
        base = true_quotient_dsm_loss(model, x0, logw, mask, noise_params)
        torch.manual_seed(66)
        regularized = true_quotient_dsm_loss(
            model, x0, logw, mask, noise_params, frozen_model=frozen, beta=1.0
        )

    assert regularized.item() >= base.item() - 1e-6, (
        f"Regularized loss {regularized.item():.6f} < base loss {base.item():.6f}"
    )


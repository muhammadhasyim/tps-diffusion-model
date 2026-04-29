"""Unit tests for genai_tps.training.quotient_projection.

Mathematical properties verified:
1. Vertical-kill: P_x annihilates any vertical vector (rigid rotation).
2. Idempotent: P_x(P_x(v)) == P_x(v).
3. Horizontal-is-fixed: P_x(h) == h for a vector already in H_x.
4. Dimension: projected standard Gaussian has effective rank 3N - 6
   (or 3N - 3 when only SO(3) is removed, ignoring translation which is
   handled by centering).
5. Mean curvature symmetry: h(x) is H_x-horizontal (P_x(h(x)) == h(x)).
6. Mean curvature shape and finite values.
7. horizontal_noise is identical to horizontal_projection on noise.
8. Batch consistency: per-sample results match single-sample calls.
9. Masking: padded atoms produce zero output.
"""

from __future__ import annotations

import math

import pytest
import torch

from genai_tps.training.quotient_projection import (
    horizontal_noise,
    horizontal_projection,
    mean_curvature_vector,
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cloud():
    """(B=2, N=8, 3) random centered point cloud with full mask."""
    B, N = 2, 8
    x = torch.randn(B, N, 3, dtype=torch.float64)
    # center each batch element
    x = x - x.mean(dim=1, keepdim=True)
    mask = torch.ones(B, N, dtype=torch.float64)
    return x, mask


@pytest.fixture
def padded_cloud():
    """(B=2, N=10, 3) cloud where last 3 atoms are padding."""
    B, N = 2, 10
    x = torch.randn(B, N, 3, dtype=torch.float64)
    mask = torch.ones(B, N, dtype=torch.float64)
    mask[:, -3:] = 0.0
    x[:, -3:] = 999.0  # should be ignored
    x_real = x.clone()
    x_real[:, -3:] = 0.0
    # center on real atoms only
    n_real = mask.sum(-1, keepdim=True).unsqueeze(-1)
    com = (x_real * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_real
    x = x - com * mask.unsqueeze(-1)
    x[:, -3:] = 999.0  # re-corrupt padding after centering
    return x, mask


def _random_so3(B: int, dtype=torch.float64) -> torch.Tensor:
    """Sample B uniform random SO(3) rotation matrices."""
    q = torch.randn(B, 4, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(B, 3, 3)
    return R


# ---------------------------------------------------------------------------
# 1. Vertical-kill: P_x removes angular momentum
# ---------------------------------------------------------------------------

def test_vertical_kill_exact(simple_cloud):
    """Rotating a point cloud by an infinitesimal rotation produces a vertical
    vector; P_x should annihilate it (up to floating-point tolerance)."""
    x, mask = simple_cloud
    B, N, _ = x.shape

    # Generate a vertical vector: v = omega × x  for random omega (B, 3)
    omega = torch.randn(B, 3, dtype=torch.float64)
    omega_exp = omega.unsqueeze(1).expand(B, N, 3)
    v_vertical = torch.linalg.cross(omega_exp, x, dim=-1)  # (B, N, 3)

    v_proj = horizontal_projection(x, v_vertical, mask)

    # Should be near-zero (within float64 numerical precision ~1e-7 due to K^{-1})
    assert v_proj.abs().max().item() < 1e-6, (
        f"Vertical vector not killed: max residual = {v_proj.abs().max().item():.2e}"
    )


def test_vertical_kill_batch_consistency(simple_cloud):
    """Vertical-kill works independently per batch element."""
    x, mask = simple_cloud
    B, N, _ = x.shape
    omega = torch.randn(B, 3, dtype=torch.float64)
    omega_exp = omega.unsqueeze(1).expand(B, N, 3)
    v_vert = torch.linalg.cross(omega_exp, x, dim=-1)
    v_proj = horizontal_projection(x, v_vert, mask)
    # Both batch elements should be zeroed
    for b in range(B):
        assert v_proj[b].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 2. Idempotent: P_x(P_x(v)) = P_x(v)
# ---------------------------------------------------------------------------

def test_idempotent(simple_cloud):
    """Applying P_x twice is the same as applying it once."""
    x, mask = simple_cloud
    v = torch.randn_like(x)

    pv = horizontal_projection(x, v, mask)
    ppv = horizontal_projection(x, pv, mask)

    residual = (ppv - pv).abs().max().item()
    assert residual < 1e-6, f"P_x not idempotent: max diff = {residual:.2e}"


# ---------------------------------------------------------------------------
# 3. Horizontal-is-fixed: P_x(h) == h when h is already horizontal
# ---------------------------------------------------------------------------

def test_horizontal_fixed_point(simple_cloud):
    """A vector already in H_x should be unchanged by projection."""
    x, mask = simple_cloud
    v = torch.randn_like(x)

    # Get a horizontal vector
    h = horizontal_projection(x, v, mask)
    # Project again
    h2 = horizontal_projection(x, h, mask)

    residual = (h2 - h).abs().max().item()
    assert residual < 1e-6, f"Horizontal vector not fixed: max diff = {residual:.2e}"


# ---------------------------------------------------------------------------
# 4. Effective dimension of projected noise
# ---------------------------------------------------------------------------

def test_effective_dimension_of_projected_noise():
    """Project many Gaussian noise samples; covariance should have rank 3N - 3
    (vertical subspace dim = 3 for SO(3); translation already removed by centering)."""
    torch.manual_seed(0)
    N = 6
    B_samples = 2000  # number of noise samples

    # Single fixed point cloud
    x_single = torch.randn(1, N, 3, dtype=torch.float64)
    x_single = x_single - x_single.mean(dim=1, keepdim=True)
    mask_single = torch.ones(1, N, dtype=torch.float64)

    # Generate and project many noise samples
    noise_list = []
    for _ in range(B_samples):
        eps = torch.randn(1, N, 3, dtype=torch.float64)
        p_eps = horizontal_projection(x_single, eps, mask_single)
        noise_list.append(p_eps.reshape(1, -1))  # flatten to (1, 3N)

    noise_mat = torch.cat(noise_list, dim=0)  # (B_samples, 3N)

    # Compute rank via SVD singular values
    _, S, _ = torch.linalg.svd(noise_mat - noise_mat.mean(dim=0, keepdim=True), full_matrices=False)
    # Count non-negligible singular values (relative threshold 1e-3 of max)
    rank = (S > S[0] * 1e-3).sum().item()

    expected_rank = 3 * N - 3  # remove 3 rotational DoFs; COM already fixed
    assert rank == expected_rank, (
        f"Expected rank {expected_rank} (3N - 3 = {3*N} - 3), got {rank}"
    )


# ---------------------------------------------------------------------------
# 5. Mean curvature is horizontal: P_x(h(x)) == h(x)
# ---------------------------------------------------------------------------

def test_mean_curvature_is_horizontal(simple_cloud):
    """The mean curvature vector h(x) must lie in the horizontal subspace."""
    x, mask = simple_cloud

    h = mean_curvature_vector(x, mask)
    ph = horizontal_projection(x, h, mask)

    residual = (ph - h).abs().max().item()
    assert residual < 1e-6, (
        f"Mean curvature not horizontal: max diff = {residual:.2e}"
    )


# ---------------------------------------------------------------------------
# 6. Mean curvature: shape, finite, non-trivial
# ---------------------------------------------------------------------------

def test_mean_curvature_shape(simple_cloud):
    x, mask = simple_cloud
    h = mean_curvature_vector(x, mask)
    assert h.shape == x.shape
    assert torch.isfinite(h).all(), "Mean curvature has non-finite values"


def test_mean_curvature_nontrivial(simple_cloud):
    """h(x) should be non-zero for generic point clouds."""
    x, mask = simple_cloud
    h = mean_curvature_vector(x, mask)
    assert h.abs().max().item() > 1e-10, "Mean curvature is suspiciously zero"


# ---------------------------------------------------------------------------
# 7. horizontal_noise == horizontal_projection
# ---------------------------------------------------------------------------

def test_horizontal_noise_matches_projection(simple_cloud):
    x, mask = simple_cloud
    noise = torch.randn_like(x)

    h1 = horizontal_projection(x, noise, mask)
    h2 = horizontal_noise(x, noise, mask)

    assert torch.allclose(h1, h2), "horizontal_noise differs from horizontal_projection"


# ---------------------------------------------------------------------------
# 8. Batch consistency
# ---------------------------------------------------------------------------

def test_batch_consistency():
    """Per-sample results from a batched call match individual calls."""
    B, N = 4, 12
    torch.manual_seed(7)
    x = torch.randn(B, N, 3, dtype=torch.float64)
    x = x - x.mean(dim=1, keepdim=True)
    mask = torch.ones(B, N, dtype=torch.float64)
    v = torch.randn(B, N, 3, dtype=torch.float64)

    v_batch = horizontal_projection(x, v, mask)

    for b in range(B):
        v_single = horizontal_projection(
            x[b:b+1], v[b:b+1], mask[b:b+1]
        )
        diff = (v_batch[b:b+1] - v_single).abs().max().item()
        assert diff < 1e-10, f"Batch b={b} inconsistent: {diff:.2e}"
# ---------------------------------------------------------------------------
# 9. Masking: padded atoms produce zero output
# ---------------------------------------------------------------------------

def test_padded_atoms_zeroed(padded_cloud):
    """Padded atoms (mask=0) must have zero contribution in the output."""
    x, mask = padded_cloud
    v = torch.randn_like(x)

    v_proj = horizontal_projection(x, v, mask)
    h = mean_curvature_vector(x, mask)

    # Padded atoms should produce zero output
    pad_idx = (mask == 0)
    assert v_proj[pad_idx].abs().max().item() < 1e-10
    assert h[pad_idx].abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
# 10. Projection reduces / preserves norm (P_x is an orthogonal projector)
# ---------------------------------------------------------------------------

def test_projection_reduces_norm(simple_cloud):
    """||P_x(v)|| <= ||v|| for all v (orthogonal projector on a subspace)."""
    x, mask = simple_cloud
    v = torch.randn_like(x)

    v_proj = horizontal_projection(x, v, mask)

    norm_v = (v * mask.unsqueeze(-1)).norm()
    norm_pv = v_proj.norm()

    assert norm_pv.item() <= norm_v.item() + 1e-9, (
        f"Projection inflated norm: ||Pv|| = {norm_pv:.4f} > ||v|| = {norm_v:.4f}"
    )


# ---------------------------------------------------------------------------
# 11. Equivariance under rotation (sanity)
# ---------------------------------------------------------------------------

def test_so3_equivariance_of_projection():
    """P_{Rx}(Rv) = R P_x(v) for any rotation R.

    This is a corollary of the construction: rotating the point cloud and
    the velocity by the same R should commute with projection.
    """
    torch.manual_seed(99)
    B, N = 2, 10
    x = torch.randn(B, N, 3, dtype=torch.float64)
    x = x - x.mean(dim=1, keepdim=True)
    mask = torch.ones(B, N, dtype=torch.float64)
    v = torch.randn(B, N, 3, dtype=torch.float64)

    R = _random_so3(B, dtype=torch.float64)  # (B, 3, 3)

    # Rotate point cloud and velocity
    Rx = torch.einsum("bij,bnj->bni", R, x)   # (B, N, 3)
    Rv = torch.einsum("bij,bnj->bni", R, v)   # (B, N, 3)

    # P_{Rx}(Rv)
    lhs = horizontal_projection(Rx, Rv, mask)
    # R P_x(v)
    pv = horizontal_projection(x, v, mask)
    rhs = torch.einsum("bij,bnj->bni", R, pv)

    residual = (lhs - rhs).abs().max().item()
    assert residual < 1e-6, f"SO(3) equivariance violated: max diff = {residual:.2e}"

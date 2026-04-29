"""Horizontal projection operators for quotient-space diffusion on R^{3N}/SE(3).

Implements the mathematical constructions from:
    Xu, Wang, Luo et al., "Quotient-Space Diffusion Models", ICLR 2026 Oral.
    arXiv:2604.21809, Theorems 3 & 4 (Eq. 11-12).

Background
----------
A point cloud x in R^{3N} with zero center-of-mass lives on the manifold
M = {x in R^{3N} | mean(x) = 0}.  The SO(3) action on M is x |-> R @ x.
Two configurations are equivalent if they differ by a rotation; the quotient
space Q = M / SO(3) is the "shape space" — it captures purely internal degrees
of freedom (bond lengths, angles, torsions) with rigid-body rotation factored out.

The tangent space at x decomposes into:
    - Vertical subspace V_x: infinitesimal rotations   (dim = 3)
    - Horizontal subspace H_x: pure deformations        (dim = 3N - 3)

The horizontal projection P_x: T_x M -> H_x removes the rotational component
from any velocity vector, leaving only the shape-changing component.

For a point cloud x = (x_1,...,x_N) in R^{3N} with atoms masked by `mask`:

    K = sum_n (||x_n||^2 I_3 - x_n x_n^T)      inertia tensor (3x3)
    L = sum_n x_n × v_n                           total angular momentum (3,)
    omega = K^{-1} L                              angular velocity (3,)
    P_x(v)_n = v_n - omega × x_n                 horizontal component

Mean curvature vector h(x) arises in the SDE sampler as a correction term
that compensates for the change in equivalence-class volume along the trajectory:

    h(x)_n = -(tr(K^{-1}) I - K^{-1}) x_n

Usage
-----
All functions operate on batched tensors (B, N, 3).  Padded atoms are excluded
by the `mask` argument (B, N), where 1 = real atom, 0 = padding.

The point cloud `x` must be centered (zero COM) before calling these functions.
Use `_center` from `genai_tps.training.loss` or equivalent.

References
----------
Xu Y., Wang Y., Luo S., et al. (2026). Quotient-Space Diffusion Models.
    ICLR 2026 Oral. arXiv:2604.21809, Sec. 3.2, Thm. 4.
"""

from __future__ import annotations

import torch

_INERTIA_EPS: float = 1e-6  # regularization for near-singular inertia tensors


def _inertia_tensor(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = _INERTIA_EPS,
) -> torch.Tensor:
    """Compute the 3x3 inertia-like tensor K for a masked point cloud.

    Parameters
    ----------
    x:
        Centered coordinates, shape (B, N, 3).  Must have zero COM for
        the quotient-space interpretation to be correct.
    mask:
        Binary atom mask, shape (B, N).  Padded atoms (0) are excluded.
    eps:
        Tikhonov regularization added to the diagonal of K to prevent
        singular inversions for degenerate point clouds (all collinear).

    Returns
    -------
    K:
        Inertia tensor (B, 3, 3).  Defined as
        K = sum_n mask_n * (||x_n||^2 I_3 - x_n x_n^T).
    """
    mask_3 = mask.unsqueeze(-1)           # (B, N, 1)
    x_m = x * mask_3                      # zero-out padding, (B, N, 3)

    # sum_n ||x_n||^2  -- scalar per batch element
    norm_sq_sum = (x_m * x_m).sum(dim=(-1, -2))   # (B,)

    # sum_n x_n x_n^T  -- (B, 3, 3)
    outer_sum = torch.einsum("bni,bnj->bij", x_m, x_m)

    I3 = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, 3, 3)
    K = norm_sq_sum.view(-1, 1, 1) * I3 - outer_sum   # (B, 3, 3)

    # regularize: K + eps * I
    K = K + eps * I3
    return K


def horizontal_projection(
    x: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = _INERTIA_EPS,
) -> torch.Tensor:
    """Project a velocity field onto the horizontal (shape-changing) subspace.

    Implements Equation 11 / Theorem 4 of arXiv:2604.21809:

        P_x(v)_n = v_n - omega × x_n,
        where  omega = K^{-1} (sum_n x_n × v_n)

    Vertical vectors (rigid rotations omega × x) are removed; the result
    has zero total angular momentum about the origin.

    Parameters
    ----------
    x:
        Centered point cloud, shape (B, N, 3).
    v:
        Velocity / update vectors to project, shape (B, N, 3).
    mask:
        Binary atom mask, shape (B, N).
    eps:
        Inertia-tensor regularization (see `_inertia_tensor`).

    Returns
    -------
    v_proj:
        Horizontally projected velocity, shape (B, N, 3).
        Padded atoms are zeroed.
    """
    mask_3 = mask.unsqueeze(-1)          # (B, N, 1)
    x_m = x * mask_3                     # (B, N, 3) -- masked
    v_m = v * mask_3                     # (B, N, 3) -- masked

    # Inertia tensor K (B, 3, 3)
    K = _inertia_tensor(x, mask, eps=eps)

    # Total angular momentum L = sum_n x_n × v_n  (B, 3)
    L = torch.linalg.cross(x_m, v_m, dim=-1).sum(dim=1)   # (B, 3)

    # Angular velocity omega = K^{-1} L  (B, 3)
    # Use linalg.solve for numerical stability over explicit inv
    omega = torch.linalg.solve(K, L.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # Per-atom correction: omega × x_n  (B, N, 3)
    omega_exp = omega.unsqueeze(1).expand_as(x_m)           # (B, N, 3)
    correction = torch.linalg.cross(omega_exp, x_m, dim=-1) # (B, N, 3)

    v_proj = (v - correction) * mask_3
    return v_proj


def mean_curvature_vector(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = _INERTIA_EPS,
) -> torch.Tensor:
    """Compute the mean curvature vector h(x) of the shape space Q at x.

    Implements Equation 12 / Theorem 4 of arXiv:2604.21809:

        h(x)_n = -(tr(K^{-1}) I_3 - K^{-1}) x_n
               = (K^{-1} - tr(K^{-1}) I_3) x_n

    This correction term arises in the SDE sampler (Equation 14) to account
    for the change of equivalence-class volume along the horizontal trajectory.
    It is NOT needed for the ODE sampler.

    Parameters
    ----------
    x:
        Centered point cloud, shape (B, N, 3).
    mask:
        Binary atom mask, shape (B, N).
    eps:
        Inertia-tensor regularization.

    Returns
    -------
    h:
        Mean curvature vector, shape (B, N, 3).  Padded atoms are zeroed.
    """
    mask_3 = mask.unsqueeze(-1)       # (B, N, 1)
    x_m = x * mask_3                  # (B, N, 3)

    K = _inertia_tensor(x, mask, eps=eps)

    # K^{-1} (B, 3, 3)
    K_inv = torch.linalg.inv(K)

    # tr(K^{-1}) (B,)
    tr_K_inv = K_inv.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B,)

    I3 = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
    # A = K^{-1} - tr(K^{-1}) I  (B, 3, 3)
    A = K_inv - tr_K_inv.view(-1, 1, 1) * I3

    # h_n = A @ x_n  (B, N, 3)
    h = torch.einsum("bij,bnj->bni", A, x_m)
    h = h * mask_3
    return h


def horizontal_noise(
    x: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = _INERTIA_EPS,
) -> torch.Tensor:
    """Project Gaussian noise onto the horizontal subspace at x.

    Convenience wrapper around :func:`horizontal_projection` for the
    stochastic (SDE) sampling step: noise is projected so that injected
    randomness only drives shape-space exploration, not rigid rotation.

    Parameters
    ----------
    x:
        Centered point cloud, shape (B, N, 3).
    noise:
        Standard Gaussian noise, shape (B, N, 3).
    mask:
        Binary atom mask, shape (B, N).
    eps:
        Inertia-tensor regularization.

    Returns
    -------
    noise_h:
        Noise projected onto the horizontal subspace, shape (B, N, 3).
    """
    return horizontal_projection(x, noise, mask, eps=eps)

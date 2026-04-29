"""Weighted denoising score matching loss for Boltz 2 fine-tuning.

Three SE(3) strategies are provided, ordered from most to least principled:

1. ``true_quotient_dsm_loss`` -- Quotient-space DSM (arXiv:2604.21809, ICLR 2026
   Oral).  Projects the denoising residual through the horizontal projection
   operator P_x before computing MSE, removing rigid-body DoFs from the loss.
   This is the **only variant that guarantees a valid sampler** (Table 1 of the
   paper).

2. ``alignment_weighted_dsm_loss`` (formerly ``quotient_weighted_dsm_loss``) --
   AF3-style alignment: aligns the ground truth toward the model prediction via
   Kabsch SVD.  Reduces learning difficulty but the paper proves this strategy
   has no compatible sampler (the learned model and standard ODE/SDE samplers
   are mathematically inconsistent).

3. ``weighted_dsm_loss`` / ``regularized_weighted_dsm_loss`` -- Plain Cartesian
   MSE without any SE(3) treatment.
"""

from __future__ import annotations

from typing import Any

import torch

from genai_tps.training.config import WeightedDSMConfig
from genai_tps.training.noise_schedule import EDMNoiseParams, edm_loss_weight, sample_noise_sigma
from genai_tps.training.quotient_projection import horizontal_projection


def _center(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove center of mass (translation-invariant frame).

    Returns centered coordinates and the COM that was subtracted.
    """
    n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
    mask_3 = mask.unsqueeze(-1)  # (B, M, 1)
    com = (x * mask_3).sum(dim=1, keepdim=True) / n.unsqueeze(-1)  # (B, 1, 3)
    return x - com * mask_3, com


def _kabsch_rotation(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute optimal rotation R such that R @ P ≈ Q (weighted by mask).

    Both P and Q should be centered (zero COM). Returns R as (B, 3, 3).
    Uses SVD-based Kabsch algorithm, differentiable through torch.linalg.svd.
    """
    mask_3 = mask.unsqueeze(-1)  # (B, M, 1)
    H = torch.bmm((P * mask_3).transpose(1, 2), Q * mask_3)  # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(H)
    d = torch.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))  # (B,)
    sign = torch.ones_like(d)
    sign[d < 0] = -1.0
    diag = torch.zeros_like(H)
    diag[:, 0, 0] = 1.0
    diag[:, 1, 1] = 1.0
    diag[:, 2, 2] = sign
    R = torch.bmm(Vh.transpose(1, 2), torch.bmm(diag, U.transpose(1, 2)))  # (B, 3, 3)
    return R


def _random_rotation(B: int, device: torch.device) -> torch.Tensor:
    """Sample uniform random SO(3) rotation matrices via random quaternions. (B, 3, 3)."""
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(B, 3, 3)
    return R


def _rigid_align_truth_to_pred(
    x_true: torch.Tensor, x_pred: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Align ground truth to prediction frame (Boltz-2 style, fully detached).

    Returns aligned_truth such that ||x_pred - aligned_truth||² measures
    internal-coordinate error. Gradients flow only through x_pred.
    """
    mask_3 = mask.unsqueeze(-1)
    n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)

    true_c = (x_true * mask_3).sum(dim=1, keepdim=True) / n
    pred_c = (x_pred * mask_3).sum(dim=1, keepdim=True) / n
    true_centered = (x_true - true_c) * mask_3
    pred_centered = (x_pred - pred_c) * mask_3

    H = torch.bmm(pred_centered.transpose(1, 2), true_centered)  # (B, 3, 3)
    H32 = H.float()
    U, S, Vh = torch.linalg.svd(H32)
    V = Vh.mH
    d = torch.det(torch.bmm(U, V.transpose(1, 2)))
    F_diag = torch.eye(3, device=x_true.device, dtype=H32.dtype).unsqueeze(0).expand(mask.shape[0], -1, -1).clone()
    F_diag[:, -1, -1] = d
    R = torch.bmm(U, torch.bmm(F_diag, V.transpose(1, 2))).to(x_true.dtype)

    aligned = torch.bmm(true_centered, R.transpose(1, 2)) + pred_c
    return aligned.detach()


def weighted_dsm_loss(
    model: Any,
    x0: torch.Tensor,
    logw: torch.Tensor,
    atom_mask: torch.Tensor,
    noise_params: EDMNoiseParams,
    *,
    network_condition_kwargs: dict | None = None,
) -> torch.Tensor:
    """Compute importance-weighted denoising score matching loss.

    Parameters
    ----------
    model:
        Diffusion model with ``preconditioned_network_forward(x_noisy, sigma, ...)``
        following Boltz 2 / EDM conventions.
    x0:
        (B, M, 3) ground-truth atom coordinates.
    logw:
        (B,) log importance weights from enhanced-sampling reweighting.
    atom_mask:
        (B, M) binary mask (1 = real atom, 0 = padding).
    noise_params:
        EDM noise distribution parameters.
    network_condition_kwargs:
        Optional kwargs passed through to the score network (s_trunk, s_inputs,
        feats, diffusion_conditioning, etc.). Omit for mock/test models.
    """
    B, M, _ = x0.shape
    device = x0.device

    sigma = sample_noise_sigma(B, noise_params, device=device)
    padded_sigma = sigma.view(B, 1, 1)

    eps = torch.randn_like(x0)
    x_noisy = x0 + padded_sigma * eps

    nck = network_condition_kwargs if network_condition_kwargs is not None else {}
    x_denoised = model.preconditioned_network_forward(x_noisy, sigma, network_condition_kwargs=nck)

    per_atom_mse = ((x_denoised - x0) ** 2).sum(dim=-1)  # (B, M)
    n_atoms = atom_mask.sum(dim=-1).clamp(min=1.0)  # (B,)
    per_sample_mse = (per_atom_mse * atom_mask).sum(dim=-1) / (3.0 * n_atoms)  # (B,)

    loss_w = edm_loss_weight(sigma, noise_params)  # (B,)
    per_sample_loss = per_sample_mse * loss_w  # (B,)

    w = torch.softmax(logw.float(), dim=0)  # (B,) normalized importance weights
    return (w * per_sample_loss).sum()


def regularized_weighted_dsm_loss(
    model: Any,
    frozen_model: Any,
    x0: torch.Tensor,
    logw: torch.Tensor,
    atom_mask: torch.Tensor,
    noise_params: EDMNoiseParams,
    cfg: WeightedDSMConfig,
    *,
    network_condition_kwargs: dict | None = None,
) -> torch.Tensor:
    """Weighted DSM loss plus regularization toward the frozen pretrained model.

    loss = weighted_dsm + beta * sum_i w_i * ||D_theta(x_i_noisy) - D_theta0(x_i_noisy)||^2

    The regularization uses the same noise realization as the DSM loss to avoid
    doubling the forward pass cost (shared x_noisy, sigma, eps).
    """
    B, M, _ = x0.shape
    device = x0.device

    sigma = sample_noise_sigma(B, noise_params, device=device)
    padded_sigma = sigma.view(B, 1, 1)

    eps = torch.randn_like(x0)
    x_noisy = x0 + padded_sigma * eps

    nck = network_condition_kwargs if network_condition_kwargs is not None else {}
    x_denoised = model.preconditioned_network_forward(x_noisy, sigma, network_condition_kwargs=nck)

    # --- DSM loss ---
    per_atom_mse = ((x_denoised - x0) ** 2).sum(dim=-1)
    n_atoms = atom_mask.sum(dim=-1).clamp(min=1.0)
    per_sample_mse = (per_atom_mse * atom_mask).sum(dim=-1) / (3.0 * n_atoms)
    loss_w = edm_loss_weight(sigma, noise_params)
    per_sample_loss = per_sample_mse * loss_w

    w = torch.softmax(logw.float(), dim=0)
    dsm_loss = (w * per_sample_loss).sum()

    # --- Regularization ---
    if cfg.beta <= 0.0:
        return dsm_loss

    with torch.no_grad():
        x_denoised_frozen = frozen_model.preconditioned_network_forward(
            x_noisy, sigma, network_condition_kwargs=nck
        )

    reg_per_atom = ((x_denoised - x_denoised_frozen) ** 2).sum(dim=-1)
    reg_per_sample = (reg_per_atom * atom_mask).sum(dim=-1) / (3.0 * n_atoms)
    reg_loss = cfg.beta * (w * reg_per_sample).sum()

    return dsm_loss + reg_loss


def alignment_weighted_dsm_loss(
    model: Any,
    x0: torch.Tensor,
    logw: torch.Tensor,
    atom_mask: torch.Tensor,
    noise_params: EDMNoiseParams,
    *,
    frozen_model: Any | None = None,
    beta: float = 0.0,
    network_condition_kwargs: dict | None = None,
) -> torch.Tensor:
    """SE(3)-invariant weighted DSM loss using AF3-style Kabsch alignment.

    .. warning::
        This strategy has **no compatible sampler**: the paper arXiv:2604.21809
        (Table 1) proves that AF3-alignment training and standard ODE/SDE
        sampling are mathematically inconsistent.  Retained for comparison and
        backward compatibility.  Use ``true_quotient_dsm_loss`` for principled
        training.

    Follows Boltz-2's approach to SE(3) invariance:

    1. **Centers** x₀ and applies **random SO(3) rotation** + small random
       translation (data augmentation makes the training distribution
       SE(3)-invariant).
    2. **Injects noise** in Cartesian space on the augmented coordinates.
    3. **Denoises** via the score network.
    4. **Aligns ground truth to prediction** (not prediction to truth) using
       Kabsch alignment, fully detached — gradients flow only through the
       network output.

    No Jacobian correction is applied (following Boltz-2).
    """
    B, M, _ = x0.shape
    device = x0.device

    # Step 1: Center + random SO(3) augmentation (Boltz-2 Algorithm 19)
    x0_c, _ = _center(x0, atom_mask)
    if model.training:
        R_aug = _random_rotation(B, device)
        x0_aug = torch.bmm(x0_c, R_aug.transpose(1, 2))
        x0_aug = x0_aug + torch.randn(B, 1, 3, device=device)  # s_trans=1.0
    else:
        x0_aug = x0_c

    # Step 2: Inject noise in Cartesian space
    sigma = sample_noise_sigma(B, noise_params, device=device)
    padded_sigma = sigma.view(B, 1, 1)
    eps = torch.randn_like(x0_aug) * atom_mask.unsqueeze(-1)
    x_noisy = x0_aug + padded_sigma * eps

    # Step 3: Denoise
    nck = network_condition_kwargs if network_condition_kwargs is not None else {}
    x_denoised = model.preconditioned_network_forward(x_noisy, sigma, network_condition_kwargs=nck)

    # Step 4: Align truth to prediction (Boltz-2 style, fully detached)
    aligned_truth = _rigid_align_truth_to_pred(x0_aug, x_denoised, atom_mask)

    # MSE: gradients flow only through x_denoised
    per_atom_mse = ((x_denoised - aligned_truth) ** 2).sum(dim=-1)  # (B, M)
    n_atoms = atom_mask.sum(dim=-1).clamp(min=1.0)  # (B,)
    per_sample_mse = (per_atom_mse * atom_mask).sum(dim=-1) / (3.0 * n_atoms)  # (B,)

    # EDM loss weight (no Jacobian correction)
    loss_w = edm_loss_weight(sigma, noise_params)  # (B,)
    per_sample_loss = per_sample_mse * loss_w

    # Importance weighting
    w = torch.softmax(logw.float(), dim=0)  # (B,)
    dsm_loss = (w * per_sample_loss).sum()

    # Optional regularization toward frozen model
    if beta > 0.0 and frozen_model is not None:
        with torch.no_grad():
            x_frozen = frozen_model.preconditioned_network_forward(
                x_noisy, sigma, network_condition_kwargs=nck
            )
        aligned_frozen = _rigid_align_truth_to_pred(x_frozen, x_denoised, atom_mask)
        reg_per_atom = ((x_denoised - aligned_frozen) ** 2).sum(dim=-1)
        reg_per_sample = (reg_per_atom * atom_mask).sum(dim=-1) / (3.0 * n_atoms)
        dsm_loss = dsm_loss + beta * (w * reg_per_sample).sum()

    return dsm_loss


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Deprecated alias -- ``quotient_weighted_dsm_loss`` was misnamed; it
#: implements AF3-style Kabsch alignment, not true quotient-space diffusion.
#: Use ``alignment_weighted_dsm_loss`` or ``true_quotient_dsm_loss`` instead.
quotient_weighted_dsm_loss = alignment_weighted_dsm_loss


# ---------------------------------------------------------------------------
# True quotient-space DSM loss (arXiv:2604.21809, ICLR 2026)
# ---------------------------------------------------------------------------

def true_quotient_dsm_loss(
    model: Any,
    x0: torch.Tensor,
    logw: torch.Tensor,
    atom_mask: torch.Tensor,
    noise_params: EDMNoiseParams,
    *,
    frozen_model: Any | None = None,
    beta: float = 0.0,
    network_condition_kwargs: dict | None = None,
) -> torch.Tensor:
    """Principled quotient-space weighted DSM loss (arXiv:2604.21809).

    Trains the diffusion model so that the training objective is compatible
    with standard ODE/SDE samplers — the **only** SE(3)-invariant strategy
    for which this compatibility holds (Table 1 of the paper).

    The key difference from ``alignment_weighted_dsm_loss``: instead of
    aligning the ground truth to the prediction via Kabsch SVD, we project
    the denoising **residual** through the horizontal projection operator
    P_x at the noisy point.  Only the shape-changing (horizontal) component
    of the error is penalized; the model is free to output any rigid rotation.

    Loss (Equation 13 of arXiv:2604.21809):
        L(θ) = E[w(t) * ||P_{x_noisy}(D_θ(x_noisy, t) - x0)||²]

    Parameters
    ----------
    model:
        Diffusion model with ``preconditioned_network_forward(x_noisy, sigma, ...)``.
    x0:
        (B, M, 3) ground-truth atom coordinates.
    logw:
        (B,) log importance weights from enhanced-sampling reweighting.
    atom_mask:
        (B, M) binary mask (1 = real atom, 0 = padding).
    noise_params:
        EDM noise distribution parameters.
    frozen_model:
        Optional frozen copy of the model for regularization.  When ``beta > 0``
        and this is provided, the loss includes a horizontal-projected distance
        between the live and frozen model outputs.
    beta:
        Regularization coefficient toward the frozen pretrained model.
        0.0 (default) disables regularization.
    network_condition_kwargs:
        Optional kwargs passed through to the score network.

    Notes
    -----
    The inertia tensor K is computed at the **noisy** point x_noisy (centered),
    which is the correct evaluation point for the horizontal projection in the
    diffusion loss (see Sec. 3.3 of arXiv:2604.21809).
    """
    B, M, _ = x0.shape
    device = x0.device

    # Step 1: Center + random SO(3) augmentation (Boltz-2 Algorithm 19)
    x0_c, _ = _center(x0, atom_mask)
    if model.training:
        R_aug = _random_rotation(B, device)
        x0_aug = torch.bmm(x0_c, R_aug.transpose(1, 2))
        x0_aug = x0_aug + torch.randn(B, 1, 3, device=device)  # s_trans=1.0
    else:
        x0_aug = x0_c

    # Step 2: Inject noise in Cartesian space
    sigma = sample_noise_sigma(B, noise_params, device=device)
    padded_sigma = sigma.view(B, 1, 1)
    eps = torch.randn_like(x0_aug) * atom_mask.unsqueeze(-1)
    x_noisy = x0_aug + padded_sigma * eps

    # Step 3: Denoise via score network
    nck = network_condition_kwargs if network_condition_kwargs is not None else {}
    x_denoised = model.preconditioned_network_forward(x_noisy, sigma, network_condition_kwargs=nck)

    # Step 4: Horizontal projection of residual at the noisy point
    # Center x_noisy for the projection (K requires zero COM)
    x_noisy_c, _ = _center(x_noisy, atom_mask)
    residual = x_denoised - x0_aug                                 # (B, M, 3)
    residual_proj = horizontal_projection(x_noisy_c, residual, atom_mask)

    # MSE on projected residual
    per_atom_mse = (residual_proj ** 2).sum(dim=-1)                # (B, M)
    n_atoms = atom_mask.sum(dim=-1).clamp(min=1.0)                 # (B,)
    per_sample_mse = (per_atom_mse * atom_mask).sum(dim=-1) / (3.0 * n_atoms)

    # EDM loss weight
    loss_w = edm_loss_weight(sigma, noise_params)
    per_sample_loss = per_sample_mse * loss_w

    # Importance weighting
    w = torch.softmax(logw.float(), dim=0)
    dsm_loss = (w * per_sample_loss).sum()

    # Optional regularization toward frozen model (projected)
    if beta > 0.0 and frozen_model is not None:
        with torch.no_grad():
            x_frozen = frozen_model.preconditioned_network_forward(
                x_noisy, sigma, network_condition_kwargs=nck
            )
        reg_residual = x_denoised - x_frozen
        reg_proj = horizontal_projection(x_noisy_c, reg_residual, atom_mask)
        reg_per_atom = (reg_proj ** 2).sum(dim=-1)
        reg_per_sample = (reg_per_atom * atom_mask).sum(dim=-1) / (3.0 * n_atoms)
        dsm_loss = dsm_loss + beta * (w * reg_per_sample).sum()

    return dsm_loss

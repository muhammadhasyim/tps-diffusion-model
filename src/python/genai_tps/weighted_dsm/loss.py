"""Weighted denoising score matching loss for Boltz 2 fine-tuning.

Includes both Cartesian-space and SE(3)-invariant quotient-space variants.
The quotient-space loss (``quotient_weighted_dsm_loss``) applies Kabsch
alignment before computing the denoising target, injects noise in a
translation-free frame, and applies a Jacobian correction for the SE(3)
orbit volume element.
"""

from __future__ import annotations

from typing import Any

import torch

from genai_tps.weighted_dsm.config import WeightedDSMConfig
from genai_tps.weighted_dsm.noise_schedule import EDMNoiseParams, edm_loss_weight, sample_noise_sigma


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


def quotient_weighted_dsm_loss(
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
    """SE(3)-invariant weighted denoising score matching loss (Boltz-2 style).

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

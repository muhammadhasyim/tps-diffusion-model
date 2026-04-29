"""Log path probability contributions for Boltz-2 diffusion trajectories.

Theory reference: docs/tps_diffusion_theory.tex

Jacobian note
-------------
The one-step map  eps_i -> x_{i+1}  has Jacobian:

    J_i = beta_i * I + mu_i * nabla_z f_theta |_{z = c_in(t_hat) * x_noisy}

with:
    beta_i = 1 + alpha_i * (1 - c_skip(t_hat))
    mu_i   = -alpha_i * c_out(t_hat) * c_in(t_hat)

The scalar approximation log|det J_i| = 3M * log|1 + alpha_i| treats
nabla_z f_theta as zero (i.e. D_theta(x) ~ x).  For fixed-length TPS with a
fixed schedule, alpha_i depends only on the step index (not the path), so
this term cancels exactly in acceptance ratios for both forward and backward
shooting.  The scalar approximation is therefore useful for cancellation and
diagnostics, but it is not a calibrated absolute neural-map determinant.

The exact Jacobian (compute_log_det_jacobian_exact) is available for
diagnostics.  It uses chunked forward-mode AD via torch.func.jvp +
torch.func.vmap and is O(M * n_chunks) forward passes per step.  Do not insert
exact, state-dependent Jacobians into backward-shooting acceptance ratios
without also deriving the matching proposal-ratio correction.

Density convention
------------------
The TPS implementation samples an extended random-variable state consisting of
coordinates plus per-step random draws (rotation, translation, churn noise).
In that parameterization the path density is a product of priors over the
draws and has no coordinate Jacobian.  A coordinate-density interpretation can
instead include ``-log|det J_i|`` for the change of variables from noise to
coordinates.  ``trajectory_log_path_prob`` and ``compute_log_path_prob`` return
a reduced diagnostic score: initial-noise prior, churn-noise priors, and
optional scalar Jacobian terms.  Haar factors are constant and translation
priors are intentionally omitted because the implemented MH moves draw
translation from the same prior in proposal and target ratios.

Quotient-space sampling note
-----------------------------
When ``BoltzSamplerCore.quotient_space_sampling=True`` the Euler update is
replaced by a horizontally projected step (arXiv:2604.21809, Eq. 13):

    x_{i+1} = x_i + step_scale * (sigma_{i+1} - t_hat_i) * P_{x_i}(v_theta)

The sampler applies the horizontal projection in a mask-centered shape frame
to remove rotational vertical motion.  Boltz's random translation is retained
as an auxiliary variable in the extended path state, so this implementation is
an SO(3)-quotient sampler with translation noise rather than a fully
CoM-free SE(3) coordinate sampler.

The scalar log|det J_i| term above remains a schedule-only diagnostic in this
mode.  No production code assumes it is a complete quotient-coordinate
Jacobian.

Forward shooting and global reshuffle remain valid because they sample from
the same forward quotient-space kernel used in the path weight.  Backward
shooting additionally requires quotient-aware fixed-point inversion of the
projected map; this is handled in ``BoltzSamplerCore._solve_x_noisy_from_output``.
Until the full backward Hastings correction is validated for production,
``BoltzDiffusionBackwardShootMover`` refuses quotient-space cores.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.snapshot import BoltzSnapshot

if TYPE_CHECKING:
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore


# ---------------------------------------------------------------------------
# EDM preconditioning scalars (Karras et al. 2022)
# Used by both the scalar Jacobian approximation and the exact computation.
# ---------------------------------------------------------------------------

def _c_skip(sigma: float, sigma_data: float) -> float:
    """Skip-connection coefficient: sigma_d^2 / (sigma^2 + sigma_d^2)."""
    return sigma_data**2 / (sigma**2 + sigma_data**2)


def _c_out(sigma: float, sigma_data: float) -> float:
    """Output scaling: sigma * sigma_d / sqrt(sigma^2 + sigma_d^2)."""
    return sigma * sigma_data / math.sqrt(sigma**2 + sigma_data**2)


def _c_in(sigma: float, sigma_data: float) -> float:
    """Input scaling: 1 / sqrt(sigma^2 + sigma_d^2)."""
    return 1.0 / math.sqrt(sigma**2 + sigma_data**2)


def _c_noise(sigma: float, sigma_data: float) -> float:
    """Noise conditioning: 0.25 * ln(sigma / sigma_d)."""
    return 0.25 * math.log(sigma / sigma_data)


# ---------------------------------------------------------------------------
# Exact Jacobian scalars (theory doc Section 4, Eq. beta/mu)
# ---------------------------------------------------------------------------

def _jacobian_scalars(
    alpha_i: float,
    t_hat: float,
    sigma_data: float,
) -> Tuple[float, float, float]:
    """Return (beta_i, mu_i, c_i) for the exact Jacobian decomposition.

    J_i = beta_i * I + mu_i * nabla_z f_theta
    log|det J_i| = 3M*log|beta_i| + log|det(I + c_i * nabla_z f_theta)|

    where c_i = mu_i / beta_i.

    The scalar Jacobian approximation sets nabla_z f_theta = 0 and recovers
    log|det J_i| ~ 3M * log|beta_i|. When c_skip ~ 1 (i.e. at low noise
    where sigma << sigma_data), beta_i ~ 1 + alpha_i, matching the scalar
    approximation log_det_jacobian_step(alpha, n_coords).
    """
    cs = _c_skip(t_hat, sigma_data)
    co = _c_out(t_hat, sigma_data)
    ci = _c_in(t_hat, sigma_data)
    beta = 1.0 + alpha_i * (1.0 - cs)
    mu = -alpha_i * co * ci
    c = mu / beta
    return beta, mu, c


# ---------------------------------------------------------------------------
# Chunked forward-mode Jacobian (theory doc Section 4.4)
# ---------------------------------------------------------------------------

def _chunked_jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    n: int,
    chunk_size: int = 64,
) -> Tensor:
    """Compute the full (n x n) Jacobian of f at x via chunked JVPs.

    Assembles the Jacobian column-by-column using forward-mode AD
    (torch.func.jvp wrapped in torch.func.vmap).
    Cost: ceil(n / chunk_size) batched forward passes through f.

    Args:
        f: Callable R^n -> R^n (flattened score model interface).
        x: Input tensor of shape (n,).
        n: Dimension (= 3*M for atom coordinates).
        chunk_size: Number of basis vectors per vmap batch (tune for
            memory/speed trade-off).

    Returns:
        Jacobian matrix of shape (n, n) on the same device as x.
    """
    device = x.device

    def jvp_col(v: Tensor) -> Tensor:
        _, tangent = torch.func.jvp(f, (x,), (v,))
        return tangent

    J_cols: List[Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = end - start
        basis = torch.zeros(chunk, n, device=device, dtype=x.dtype)
        idx = torch.arange(chunk, device=device)
        basis[idx, idx + start] = 1.0
        cols = torch.vmap(jvp_col)(basis)  # (chunk, n): each row is J[:, j]
        J_cols.append(cols)

    return torch.cat(J_cols, dim=0).T  # (n, n)


# ---------------------------------------------------------------------------
# Exact log|det J| via chunked forward-mode AD (theory doc Section 4.4)
# ---------------------------------------------------------------------------

def compute_log_det_jacobian_exact(
    score_model: nn.Module,
    x_noisy: Tensor,
    t_hat: float,
    network_kwargs: Dict,
    alpha_i: float,
    sigma_data: float,
    chunk_size: int = 64,
) -> Tensor:
    """Compute log|det J_i| exactly via chunked forward-mode AD.

    J_i = beta_i * I + mu_i * nabla_z f_theta |_{z = c_in(t_hat) * x_noisy}

    log|det J_i| computed via torch.linalg.slogdet on the assembled J matrix.

    NOTE: This is O(ceil(3M/chunk_size)) forward passes through score_model
    plus O((3M)^3) for slogdet.  For M=200 atoms at chunk_size=64, this is
    ~10 forward passes.  See theory doc Table 1 for cost estimates.

    NOTE ON SIGN: This returns log|det J_i| as a non-negative-aware signed
    log-det (i.e. the absolute value log-det from slogdet).  The path
    probability contribution is -log|det J_i| (change-of-variables from eps
    to x_{i+1}).  Callers are responsible for the sign.

    Args:
        score_model: Raw score model f_theta (NOT the preconditioned
            denoiser D_theta).  Must accept
            score_model(r_noisy=..., times=..., **network_kwargs).
        x_noisy: Atom coordinates (1, M, 3) or (M, 3) on device.
        t_hat: Effective noise level at this step.
        network_kwargs: Conditioning kwargs forwarded to score_model.
        alpha_i: Step coefficient s*(sigma_i - t_hat)/t_hat.
        sigma_data: Model hyperparameter sigma_data (default 16.0 for Boltz-2).
        chunk_size: JVP batch size.

    Returns:
        Scalar tensor: log|det J_i| (unsigned; subtract from log path prob).
    """
    if x_noisy.dim() == 3:
        x_noisy = x_noisy.squeeze(0)
    M = x_noisy.shape[0]
    n = 3 * M
    device = x_noisy.device
    dtype = x_noisy.dtype

    beta, mu, _ = _jacobian_scalars(alpha_i, t_hat, sigma_data)
    ci = _c_in(t_hat, sigma_data)
    cn_val = _c_noise(t_hat, sigma_data)
    times = torch.tensor([cn_val], device=device, dtype=dtype)

    z = (ci * x_noisy).reshape(n)

    def f_flat(z_vec: Tensor) -> Tensor:
        return score_model(
            r_noisy=z_vec.reshape(1, M, 3),
            times=times,
            **network_kwargs,
        ).reshape(n)

    J_f = _chunked_jacobian(f_flat, z, n, chunk_size)
    J = beta * torch.eye(n, device=device, dtype=dtype) + mu * J_f
    _, logabsdet = torch.linalg.slogdet(J)
    return logabsdet


def log_det_jacobian_step(alpha: float, n_coords: int) -> torch.Tensor:
    """Per-step log |det J| under the scalar Jacobian approximation (3M dims).

    Approximates nabla_z f_theta = 0, giving J_i ~ (1+alpha_i)*I and thus
    log|det J_i| ~ 3M * log|1+alpha_i|.

    This approximation is EXACT for acceptance ratios in fixed-length TPS
    because alpha_i depends only on the schedule step index (not the path),
    so these terms cancel identically between old and new paths.

    For absolute diagnostics of the coordinate map, use
    compute_log_det_jacobian_exact outside production MH ratios.
    """
    d = 3 * n_coords
    return d * torch.log(torch.abs(torch.tensor(1.0 + alpha)) + 1e-30)


def log_gaussian_isotropic(eps: torch.Tensor, variance: float) -> torch.Tensor:
    """Log density of isotropic Gaussian for a tensor ``eps`` (any shape)."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    flat = eps.reshape(eps.shape[0], -1)
    d = flat.shape[1]
    quad = (flat**2).sum(dim=-1)
    return -0.5 * quad / variance - 0.5 * d * (math.log(2 * math.pi * variance))


def log_prior_initial(atom_coords: torch.Tensor, sigma0: float) -> torch.Tensor:
    """Log density of N(0, sigma0^2 I) for initial noise coordinates."""
    v = sigma0**2
    return log_gaussian_isotropic(atom_coords, v)


def forward_step_meta(core: "BoltzSamplerCore", step_idx: int) -> dict[str, Any]:
    """Per-step metadata for :func:`compute_log_path_prob` (matches ``single_forward_step``)."""
    sch = core.schedule[step_idx]
    return {
        "sigma_tm": sch.sigma_tm,
        "sigma_t": sch.sigma_t,
        "t_hat": sch.t_hat,
        "noise_var": sch.noise_var,
        "step_scale": float(core.diffusion.step_scale),
    }


def trajectory_log_path_prob(
    traj: Trajectory,
    core: "BoltzSamplerCore",
    *,
    include_jacobian: bool = True,
    strict: bool = True,
) -> float | None:
    """Reduced log path score from stored frame noises (Boltz snapshots).

    Uses the same convention as :func:`compute_log_path_prob`: initial-noise
    prior, Gaussian churn-noise draws at each forward step, and optional scalar
    Jacobian diagnostics.  Haar factors and translation priors are omitted, so
    this is not a complete calibrated extended-state log density.

    Parameters
    ----------
    strict:
        When ``True`` (default), raises ``ValueError`` if the trajectory is too
        short, contains non-BoltzSnapshot frames, or is missing ``eps_used``.
        When ``False``, returns ``None`` for these cases (legacy behavior).

    If any lower frame was produced by backward re-noising, uses
    :func:`prefix_forward_transitions_log_prob_tensor` and recovered forward noises.
    """
    n = len(traj)
    if n < 2:
        if strict:
            raise ValueError(
                f"trajectory_log_path_prob: trajectory has {n} frames (need >= 2)."
            )
        return None
    snap0 = traj[0]
    if not isinstance(snap0, BoltzSnapshot):
        if strict:
            raise TypeError(
                f"trajectory_log_path_prob: first snapshot is {type(snap0).__name__}, "
                "not BoltzSnapshot. Path probability requires BoltzSnapshot frames."
            )
        return None
    use_mixed = any(
        isinstance(traj[i], BoltzSnapshot) and getattr(traj[i], "generated_by_backward", False)
        for i in range(n - 1)
    )
    if use_mixed:
        trans = prefix_forward_transitions_log_prob_tensor(
            traj, core, include_jacobian=include_jacobian
        )
        x0 = _tensor_batch(snap0, core)
        prior = log_prior_initial(x0, float(core.schedule[0].sigma_tm)).sum()
        return float((trans + prior).detach().cpu().item())

    eps_list: list[torch.Tensor] = []
    for i in range(n - 1):
        snap = traj[i + 1]
        if not isinstance(snap, BoltzSnapshot) or snap.eps_used is None:
            if strict:
                raise ValueError(
                    f"trajectory_log_path_prob: frame {i + 1} is not a BoltzSnapshot "
                    f"or has eps_used=None (type={type(snap).__name__})."
                )
            return None
        eps_list.append(snap.eps_used)
    x0 = snap0.tensor_coords
    if x0 is None:
        x0 = torch.as_tensor(snap0.coordinates, dtype=torch.float32, device=core.diffusion.device)
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
    meta_list = [forward_step_meta(core, k) for k in range(n - 1)]
    n_atoms = int(x0.shape[1])
    lp = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x0,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=include_jacobian,
        n_atoms=n_atoms,
    )
    return float(lp.detach().cpu().item())


def compute_log_path_prob(
    eps_list: list[torch.Tensor],
    meta_list: list[dict[str, Any]],
    initial_coords: torch.Tensor | None = None,
    sigma0: float | None = None,
    include_jacobian: bool = True,
    n_atoms: int | None = None,
) -> torch.Tensor:
    """Sum the reduced path score over noise draws and optional Jacobian terms.

    Uses the scalar Jacobian approximation (log_det_jacobian_step).  For
    fixed-length TPS the Jacobian terms cancel in acceptance ratios so
    ``include_jacobian=True`` has no effect on MCMC correctness; it is
    included here only as a diagnostic coordinate-density correction.  Haar
    densities are constant.  Translation densities are nonconstant in ``tau``
    but are omitted because every implemented MH proposal draws ``tau`` from
    the same prior, so these factors cancel in accepted-ratio calculations.
    If ``initial_coords`` and ``sigma0`` are given, adds log rho(x_0).
    """
    device = eps_list[0].device if eps_list else torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)
    for eps, meta in zip(eps_list, meta_list):
        v = float(meta["noise_var"])
        if v <= 1e-30:
            continue
        total = total + log_gaussian_isotropic(eps, v).sum()
        if include_jacobian:
            t_hat = float(meta["t_hat"])
            sigma_t = float(meta["sigma_t"])
            step_scale = float(meta["step_scale"])
            alpha = step_scale * (sigma_t - t_hat) / t_hat
            m = n_atoms if n_atoms is not None else eps.shape[1]
            total = total - log_det_jacobian_step(alpha, m).to(device)
    if initial_coords is not None and sigma0 is not None:
        total = total + log_prior_initial(initial_coords, sigma0).sum()
    return total


def min_metropolis_acceptance_path(
    log_p_old: float,
    log_p_new: float,
    *,
    reactive: bool,
) -> float:
    """Standard Metropolis acceptance :math:`\\min(1, \\pi_\\mathrm{new}/\\pi_\\mathrm{old})`.

    ``log_p_*`` are natural logs of the reduced path score in
    :func:`compute_log_path_prob` (noise likelihoods, optional scalar Jacobian
    terms, initial prior).
    OpenPathSampling’s shooting movers *also* apply proposal (selector / modifier)
    Hastings factors through ``sample.bias``; those are **not** included here—compare
    with ``metropolis_acceptance`` in the log for the OPS product including bias.

    If ``reactive`` is False (e.g. trial path not in the TPS ensemble), returns ``0.0``.

    Numerically stable: for :math:`d=\\log\\pi_\\mathrm{new}-\\log\\pi_\\mathrm{old}\\ge 0`,
    :math:`\\min(1,e^d)=1` without evaluating ``exp`` (avoids overflow when :math:`d` is large).
    """
    if not reactive:
        return 0.0
    d = log_p_new - log_p_old
    if d >= 0.0:
        return 1.0
    # d < 0 => exp(d) in (0, 1); no overflow; may underflow to 0.0 for very negative d
    return math.exp(d)


def acceptance_ratio_fixed_length(
    log_p_old: float,
    log_p_new: float,
    reactive: bool,
) -> float:
    """Alias of :func:`min_metropolis_acceptance_path` (argument order: old, new)."""
    return min_metropolis_acceptance_path(log_p_old, log_p_new, reactive=reactive)


def _tensor_batch(snap: BoltzSnapshot, core: "BoltzSamplerCore") -> torch.Tensor:
    tc = snap.tensor_coords
    if tc is not None:
        return tc
    c = torch.as_tensor(snap.coordinates, dtype=torch.float32, device=core.diffusion.device)
    return c.unsqueeze(0) if c.dim() == 2 else c


def prefix_forward_transitions_log_prob_tensor(
    traj: Trajectory,
    core: "BoltzSamplerCore",
    *,
    include_jacobian: bool = True,
) -> torch.Tensor:
    """Sum log :math:`p_{\\mathrm{fwd}}(x_{i+1}\\mid x_i)` (Gaussian + Jacobian) over prefix edges.

    For an edge whose lower frame was produced by backward re-noising, recovers the
    implied forward noise via :meth:`BoltzSamplerCore.recover_forward_noise` using
    ``R,\\tau`` stored on the **lower** (backward-generated) snapshot, because
    ``single_backward_step(x_{i+1}, i)`` samples fresh augmentation at step ``i`` and
    stores it on the resulting ``x_i`` frame.
    """
    n = len(traj)
    if n < 2:
        return torch.zeros((), device=core.diffusion.device, dtype=torch.float32)
    device = core.diffusion.device
    total = torch.zeros((), device=device, dtype=torch.float32)
    n_atoms = int(_tensor_batch(traj[0], core).shape[1])
    for i in range(n - 1):
        s0 = traj[i]
        s1 = traj[i + 1]
        if not isinstance(s0, BoltzSnapshot) or not isinstance(s1, BoltzSnapshot):
            raise TypeError("prefix forward log prob requires BoltzSnapshot frames")
        step_idx = int(s0.step_index)
        meta = forward_step_meta(core, step_idx)
        v = float(meta["noise_var"])
        if getattr(s0, "generated_by_backward", False):
            if s0.rotation_R is None or s0.translation_t is None:
                raise ValueError("forward recovery requires rotation_R and translation_t on lower (backward-generated) frame")
            eps, meta = core.recover_forward_noise(
                _tensor_batch(s0, core),
                _tensor_batch(s1, core),
                step_idx,
                s0.rotation_R,
                s0.translation_t,
                getattr(s0, "center_mean_before_step", None),
            )
        else:
            if s1.eps_used is None:
                raise ValueError("forward frame missing eps_used")
            eps = s1.eps_used
        if v > 1e-30:
            total = total + log_gaussian_isotropic(eps, v).sum()
        if include_jacobian:
            t_hat = float(meta["t_hat"])
            sigma_t = float(meta["sigma_t"])
            step_scale = float(meta["step_scale"])
            alpha = step_scale * (sigma_t - t_hat) / t_hat
            total = total - log_det_jacobian_step(alpha, n_atoms).to(device)
    return total


def prefix_backward_proposal_log_prob_tensor(
    traj: Trajectory,
    core: "BoltzSamplerCore",
) -> torch.Tensor:
    """Sum log backward-kernel Gaussian draws over prefix edges ``j+1 \\to j``.

    Uses stored ``eps_used`` on backward-generated lower frames; otherwise recovers
    implied backward noise using ``R,\\tau`` from the upper frame and the per-atom
    mean of the lower frame coordinates as the centering reference (consistent with
    how :meth:`BoltzSamplerCore.single_backward_step` centers before augmentation).
    """
    device = core.diffusion.device
    total = torch.zeros((), device=device, dtype=torch.float32)
    n = len(traj)
    for j in range(n - 1):
        s_lo, s_hi = traj[j], traj[j + 1]
        if not isinstance(s_lo, BoltzSnapshot) or not isinstance(s_hi, BoltzSnapshot):
            raise TypeError("prefix backward log prob requires BoltzSnapshot frames")
        v = float(core.schedule[j].noise_var)
        if v <= 1e-30:
            continue
        if s_lo.generated_by_backward and s_lo.eps_used is not None:
            eps = s_lo.eps_used
        else:
            if s_hi.rotation_R is None or s_hi.translation_t is None:
                raise ValueError("backward recovery requires R,tau on upper frame")
            x_lo = _tensor_batch(s_lo, core)
            if hasattr(core, "_masked_center_of_mass"):
                cm = core._masked_center_of_mass(x_lo)
            else:
                cm = x_lo.mean(dim=-2, keepdim=True)
            eps, _ = core.recover_backward_noise(
                x_lo,
                _tensor_batch(s_hi, core),
                j,
                s_hi.rotation_R,
                s_hi.translation_t,
                cm,
            )
        total = total + log_gaussian_isotropic(eps, v).sum()
    return total


def backward_shooting_metropolis_bias(
    old_traj: Trajectory,
    new_traj: Trajectory,
    shooting_index: int,
    core: "BoltzSamplerCore",
    *,
    include_jacobian: bool = True,
    exact_jacobian: bool = False,
) -> float:
    """Hastings factor for backward shooting under non-reversible diffusion dynamics.

    The Metropolis-Hastings ratio for backward shooting is

    .. math::

        r = \\frac{\\pi(X^{\\mathrm{new}})}{\\pi(X^{\\mathrm{old}})}
            \\cdot
            \\frac{q_{\\mathrm{bwd}}(X^{\\mathrm{old}}_{0:k})}{q_{\\mathrm{bwd}}(X^{\\mathrm{new}}_{0:k})}

    where ``pi`` is the path weight (forward Gaussian noise densities times initial prior)
    and ``q_bwd`` is the backward (re-noising) proposal density for the prefix 0..k.
    Suffix k..L is shared and cancels.  In log form:

    .. math::

        \\log r
        = \\underbrace{\\log\\pi_{\\mathrm{fwd}}(X^{\\mathrm{new}}_{0:k})
                     - \\log\\pi_{\\mathrm{fwd}}(X^{\\mathrm{old}}_{0:k})}_{\\text{path-weight ratio}}
        + \\underbrace{\\log q_{\\mathrm{bwd}}(X^{\\mathrm{old}}_{0:k})
                      - \\log q_{\\mathrm{bwd}}(X^{\\mathrm{new}}_{0:k})}_{\\text{Hastings correction}}
        + \\log\\rho(x^{\\mathrm{new}}_0) - \\log\\rho(x^{\\mathrm{old}}_0)

    Returns :math:`\\min(1, \\exp(\\log r))`, where :math:`k` is ``shooting_index``
    and prefixes include frames ``0..k``.
    """
    if exact_jacobian:
        raise NotImplementedError(
            "Exact Jacobians are not supported in backward_shooting_metropolis_bias. "
            "Use the scalar schedule-dependent Jacobian, which cancels for "
            "fixed-length TPS moves."
        )
    k = int(shooting_index)
    if k < 0 or k >= len(old_traj) or len(new_traj) != len(old_traj):
        return 1.0
    old_pre = Trajectory(list(old_traj[: k + 1]))
    new_pre = Trajectory(list(new_traj[: k + 1]))
    try:
        lf_old = prefix_forward_transitions_log_prob_tensor(
            old_pre, core, include_jacobian=include_jacobian
        )
        lf_new = prefix_forward_transitions_log_prob_tensor(
            new_pre, core, include_jacobian=include_jacobian
        )
        lb_old = prefix_backward_proposal_log_prob_tensor(old_pre, core)
        lb_new = prefix_backward_proposal_log_prob_tensor(new_pre, core)
    except (TypeError, ValueError, RuntimeError):
        return 1.0

    s0o, s0n = old_pre[0], new_pre[0]
    if not isinstance(s0o, BoltzSnapshot) or not isinstance(s0n, BoltzSnapshot):
        return 1.0
    sig0 = float(core.schedule[0].sigma_tm)
    lp0_old = log_prior_initial(_tensor_batch(s0o, core), sig0).sum()
    lp0_new = log_prior_initial(_tensor_batch(s0n, core), sig0).sum()

    log_r = (
        lf_new
        - lf_old
        + lb_old
        - lb_new
        + lp0_new
        - lp0_old
    )
    log_r = float(log_r.detach().cpu().item())
    log_r = min(log_r, 700.0)
    if log_r >= 0.0:
        return 1.0
    return math.exp(log_r)

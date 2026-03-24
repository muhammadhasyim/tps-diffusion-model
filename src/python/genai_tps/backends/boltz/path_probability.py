"""Log path probability contributions for Boltz-2 diffusion trajectories."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.snapshot import BoltzSnapshot

if TYPE_CHECKING:
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore


def log_det_jacobian_step(alpha: float, n_coords: int) -> torch.Tensor:
    """Per-step log |det J| for the affine map in :math:`\\varepsilon` (3M dims)."""
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
) -> float | None:
    """Log path density :math:`\\log \\pi(\\text{path})` from stored frame noises (Boltz snapshots).

    Uses the same factorization as :func:`compute_log_path_prob`: Gaussian draws at each
    forward step plus optional Jacobian terms and the initial noise prior. Returns
    ``None`` if frames are not :class:`BoltzSnapshot` or any step is missing ``eps_used``.
    If any lower frame was produced by backward re-noising, uses
    :func:`prefix_forward_transitions_log_prob_tensor` and recovered forward noises.
    """
    n = len(traj)
    if n < 2:
        return None
    snap0 = traj[0]
    if not isinstance(snap0, BoltzSnapshot):
        return None
    use_mixed = any(
        isinstance(traj[i], BoltzSnapshot) and getattr(traj[i], "generated_by_backward", False)
        for i in range(n - 1)
    )
    if use_mixed:
        try:
            trans = prefix_forward_transitions_log_prob_tensor(
                traj, core, include_jacobian=include_jacobian
            )
            x0 = _tensor_batch(snap0, core)
            prior = log_prior_initial(x0, float(core.schedule[0].sigma_tm)).sum()
            return float((trans + prior).detach().cpu().item())
        except (TypeError, ValueError, RuntimeError):
            return None

    eps_list: list[torch.Tensor] = []
    for i in range(n - 1):
        snap = traj[i + 1]
        if not isinstance(snap, BoltzSnapshot) or snap.eps_used is None:
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
    """Sum log p over stochastic draws (noise terms) and optional Jacobian factors.

    Haar and translation densities are omitted (constant w.r.t. state; cancel in
    ratios). If ``initial_coords`` and ``sigma0`` are given, adds log rho(x_0).
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
            total = total + log_det_jacobian_step(alpha, m).to(device)
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

    ``log_p_*`` are natural logs of the **factorized path density** in
    :func:`compute_log_path_prob` (noise likelihoods, Jacobian terms, initial prior).
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
            total = total + log_det_jacobian_step(alpha, n_atoms).to(device)
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
            cm = _tensor_batch(s_lo, core).mean(dim=-2, keepdim=True)
            eps, _ = core.recover_backward_noise(
                _tensor_batch(s_lo, core),
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

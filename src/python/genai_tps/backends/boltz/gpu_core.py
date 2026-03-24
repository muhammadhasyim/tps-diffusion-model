"""GPU-native Boltz-2 diffusion stepping (no OpenPathSampling imports)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F

from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.model.modules.utils import compute_random_augmentation

logger = logging.getLogger(__name__)

_NAN_FALLBACK_LOG_COUNT = 0
_NAN_FALLBACK_LOG_MAX = 5


def _nan_fallback(x_next: torch.Tensor, x_fallback: torch.Tensor, *, label: str) -> torch.Tensor:
    """Replace non-finite elements in *x_next* with *x_fallback*, log once.

    Prevents a single NaN frame from cascading through all subsequent
    diffusion steps, which would poison the entire trajectory irreversibly.
    """
    global _NAN_FALLBACK_LOG_COUNT
    bad = ~torch.isfinite(x_next)
    if not bad.any():
        return x_next
    _NAN_FALLBACK_LOG_COUNT += 1
    if _NAN_FALLBACK_LOG_COUNT <= _NAN_FALLBACK_LOG_MAX:
        n_bad = int(bad.sum().item())
        n_tot = int(x_next.numel())
        logger.warning(
            "%s: %d/%d non-finite values replaced with fallback coords "
            "(NaN-cascade guard #%d)",
            label, n_bad, n_tot, _NAN_FALLBACK_LOG_COUNT,
        )
    return torch.where(bad, x_fallback, x_next)


def _kwargs_for_preconditioned_forward(network_condition_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Prepare kwargs for :meth:`AtomDiffusion.preconditioned_network_forward`.

    ``build_network_condition_kwargs`` may include ``steering_args`` for
    :meth:`AtomDiffusion.sample`, but the score model's ``forward`` only accepts
    trunk/features/conditioning (not steering).
    """
    out = dict(network_condition_kwargs)
    out.pop("steering_args", None)
    return out


@dataclass(frozen=True)
class StepSchedule:
    """One sampling step: sigma_tm -> sigma_t with gamma on the upper index."""

    sigma_tm: float
    sigma_t: float
    gamma: float
    t_hat: float
    noise_var: float


class BoltzSamplerCore:
    """Wraps :class:`boltz.model.modules.diffusionv2.AtomDiffusion` for TPS.

    An explicit :meth:`build_schedule` length is preserved until :meth:`build_schedule`
    is called again; :meth:`sample_initial_noise` does not reset the step count to
    the checkpoint default.
    """

    def __init__(
        self,
        diffusion_module: Any,
        atom_mask: torch.Tensor,
        network_condition_kwargs: dict[str, Any],
        multiplicity: int = 1,
    ) -> None:
        self.diffusion = diffusion_module
        # Keep masks on the same device as the score network (required for rigid align / batching).
        self.atom_mask = atom_mask.to(device=self.diffusion.device)
        self.network_condition_kwargs = network_condition_kwargs
        self.multiplicity = multiplicity
        self._schedule: list[StepSchedule] | None = None
        self._num_sampling_steps: int | None = None

    def build_schedule(self, num_sampling_steps: int | None = None) -> None:
        n = num_sampling_steps if num_sampling_steps is not None else self.diffusion.num_sampling_steps
        self._num_sampling_steps = n
        sigmas = self.diffusion.sample_schedule(n)
        gammas = torch.where(sigmas > self.diffusion.gamma_min, self.diffusion.gamma_0, 0.0)
        triples = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        self._schedule = []
        eta = self.diffusion.noise_scale
        for sigma_tm, sigma_t, gamma in triples:
            s_tm = float(sigma_tm.item())
            s_t = float(sigma_t.item())
            g = float(gamma.item())
            t_hat = s_tm * (1.0 + g)
            noise_var = float((eta**2) * (t_hat**2 - s_tm**2))
            self._schedule.append(
                StepSchedule(
                    sigma_tm=s_tm,
                    sigma_t=s_t,
                    gamma=g,
                    t_hat=t_hat,
                    noise_var=noise_var,
                )
            )

    @property
    def num_sampling_steps(self) -> int:
        if self._num_sampling_steps is None:
            raise RuntimeError("Call build_schedule() first.")
        return self._num_sampling_steps

    @property
    def schedule(self) -> list[StepSchedule]:
        if self._schedule is None:
            raise RuntimeError("Call build_schedule() first.")
        return self._schedule

    def _shape(self) -> tuple[int, ...]:
        m = self.atom_mask.shape[1]
        return (self.multiplicity, m, 3)

    def _assert_coords_device(self, atom_coords: torch.Tensor) -> None:
        expected = self.diffusion.device
        if atom_coords.device != expected:
            raise ValueError(
                f"atom_coords must be on diffusion device {expected}, got {atom_coords.device}"
            )

    def sample_initial_noise(self) -> torch.Tensor:
        """Gaussian noise scaled by σ at schedule start (uses current schedule if set)."""
        if self._schedule is None:
            self.build_schedule()
        sigmas = self.diffusion.sample_schedule(self.num_sampling_steps)
        init_sigma = float(sigmas[0].item())
        shape = self._shape()
        return init_sigma * torch.randn(shape, device=self.diffusion.device, dtype=torch.float32)

    def _solve_x_noisy_from_output(
        self,
        x_out: torch.Tensor,
        step_idx: int,
        n_fixed_point: int = 4,
    ) -> torch.Tensor:
        """Recover x_noisy from the step output x_out via rescaled fixed-point iteration.

        Solves the implicit equation produced by the Boltz denoising update:

            x_out = (1 + α) x_noisy − α D(x_noisy, t̂)

        where α = step_scale · (σ_t − t̂) / t̂.  The naive iteration
        x ← (x_out + α D(x)) / (1 + α) diverges at low noise levels because
        the denoiser's skip connection D(x) ≈ c_skip · x makes the Jacobian
        |α c_skip / (1 + α)| > 1 when step_scale > 1.

        This method factors out the skip connection:

            (1 + α(1 − c_skip)) x = x_out + α(D(x) − c_skip x)

        giving the rescaled iteration

            x ← (x_out + α(x̂ − c_skip x)) / γ_eff

        with γ_eff = 1 + α(1 − c_skip).  The Jacobian is now
        α(D' − c_skip)/γ_eff ≈ α c_out c_in F'(·)/γ_eff, which is bounded
        and small at all noise levels.

        When ``alignment_reverse_diff`` is enabled, the rigid alignment is
        applied inside the loop to match the forward sampling kernel.
        """
        sch = self.schedule[step_idx]
        t_hat = sch.t_hat

        step_scale = self.diffusion.step_scale
        sigma_t = sch.sigma_t
        alpha = step_scale * (sigma_t - t_hat) / t_hat

        sigma_data = self.diffusion.sigma_data
        c_skip = sigma_data ** 2 / (t_hat ** 2 + sigma_data ** 2)
        gamma_eff = 1.0 + alpha * (1.0 - c_skip)
        if abs(gamma_eff) < 1e-8:
            gamma_eff = 1e-8 if gamma_eff >= 0 else -1e-8

        b = x_out.shape[0]
        x_noisy = x_out
        for _ in range(n_fixed_point):
            kw = _kwargs_for_preconditioned_forward(dict(self.network_condition_kwargs))
            kw.pop("multiplicity", None)
            x_hat = self.diffusion.preconditioned_network_forward(
                x_noisy,
                t_hat,
                network_condition_kwargs=dict(multiplicity=b, **kw),
            )
            if self.diffusion.alignment_reverse_diff:
                x_noisy = weighted_rigid_align(
                    x_noisy.float(),
                    x_hat.float(),
                    self.atom_mask.float(),
                    self.atom_mask.float(),
                ).to(x_hat)
            x_noisy = (x_out + alpha * (x_hat - c_skip * x_noisy)) / gamma_eff
        return x_noisy

    @torch.inference_mode()
    def recover_forward_noise(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        step_idx: int,
        random_r: torch.Tensor,
        random_tr: torch.Tensor,
        *,
        n_fixed_point: int = 4,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Recover forward-step noise :math:`\\varepsilon` given consecutive frames and augmentation.

        Inverts the deterministic Boltz update (fixed-point in :math:`x^{\\mathrm{noisy}}`)
        using the same iteration as :meth:`single_backward_step`, then
        :math:`\\varepsilon = x^{\\mathrm{noisy}} - \\tilde{x}` with centered/augmented
        :math:`\\tilde{x}` from ``x_prev``. Used for log :math:`p_{\\mathrm{fwd}}` on
        prefixes produced by backward (re-noising) shooting.

        Parameters
        ----------
        x_prev, x_next
            Batched coordinates ``(B, M, 3)`` before and after the forward step
            (same ``step_idx`` as in :meth:`single_forward_step`).
        random_r, random_tr
            SE(3) augmentation used when scoring this transition (stored on the
            later-frame snapshot).
        n_fixed_point
            Fixed-point iterations for the implicit :math:`x^{\\mathrm{noisy}}` solve.
        """
        self._assert_coords_device(x_prev)
        self._assert_coords_device(x_next)
        if self._schedule is None:
            self.build_schedule()
        sch = self.schedule[step_idx]
        b = x_prev.shape[0]
        center_mean = x_prev.mean(dim=-2, keepdim=True)
        x_aug = torch.einsum("bmd,bds->bms", x_prev - center_mean, random_r) + random_tr

        x_noisy = self._solve_x_noisy_from_output(x_next, step_idx, n_fixed_point)

        eps = x_noisy - x_aug
        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sch.sigma_t,
            "t_hat": sch.t_hat,
            "noise_var": sch.noise_var,
            "step_scale": float(self.diffusion.step_scale),
        }
        return eps, meta

    @torch.inference_mode()
    def recover_backward_noise(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        step_idx: int,
        random_r: torch.Tensor,
        random_tr: torch.Tensor,
        center_mean_before: torch.Tensor,
        *,
        n_fixed_point: int = 4,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Recover backward-kernel noise :math:`\\varepsilon` given consecutive frames.

        Inverts the deterministic part of :meth:`single_backward_step` (fixed-point
        solve for :math:`x^{\\mathrm{noisy}}` from ``x_next``), then
        :math:`\\varepsilon = x^{\\mathrm{noisy}} - \\tilde{x}` with
        :math:`\\tilde{x} = R(x_{\\mathrm{prev}}-\\bar{x})+\\tau`.
        """
        self._assert_coords_device(x_prev)
        self._assert_coords_device(x_next)
        if self._schedule is None:
            self.build_schedule()
        sch = self.schedule[step_idx]
        b = x_prev.shape[0]
        cm = center_mean_before.to(device=x_prev.device, dtype=x_prev.dtype)
        y = x_prev - cm
        x_tilde = torch.einsum("bmd,bds->bms", y, random_r) + random_tr

        x_noisy = self._solve_x_noisy_from_output(x_next, step_idx, n_fixed_point)

        eps = x_noisy - x_tilde
        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sch.sigma_t,
            "t_hat": sch.t_hat,
            "noise_var": sch.noise_var,
            "step_scale": float(self.diffusion.step_scale),
        }
        return eps, meta

    @torch.inference_mode()
    def single_forward_step(
        self,
        atom_coords: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """One Boltz-2 reverse step (denoise toward lower sigma).

        Parameters
        ----------
        atom_coords
            (B, M, 3) coordinates before this step (B = multiplicity).
        step_idx
            Index into the sampling loop ``0 .. num_sampling_steps - 1``.
        """
        self._assert_coords_device(atom_coords)
        if self._schedule is None:
            self.build_schedule()
        sch = self.schedule[step_idx]
        b = atom_coords.shape[0]
        random_r, random_tr = compute_random_augmentation(
            b, device=atom_coords.device, dtype=atom_coords.dtype
        )
        center_mean = atom_coords.mean(dim=-2, keepdim=True)
        x = atom_coords - center_mean
        x = torch.einsum("bmd,bds->bms", x, random_r) + random_tr

        noise_var = sch.noise_var
        eps = sqrt(noise_var) * torch.randn(x.shape, device=x.device, dtype=x.dtype)
        x_noisy = x + eps

        t_hat = sch.t_hat
        atom_coords_denoised = torch.zeros_like(x_noisy)
        sample_ids = torch.arange(b, device=x_noisy.device)
        kw = _kwargs_for_preconditioned_forward(dict(self.network_condition_kwargs))
        kw.pop("multiplicity", None)
        kwargs = dict(multiplicity=sample_ids.numel(), **kw)
        atom_coords_denoised = self.diffusion.preconditioned_network_forward(
            x_noisy, t_hat, network_condition_kwargs=kwargs
        )

        if self.diffusion.alignment_reverse_diff:
            x_noisy = weighted_rigid_align(
                x_noisy.float(),
                atom_coords_denoised.float(),
                self.atom_mask.float(),
                self.atom_mask.float(),
            )
            x_noisy = x_noisy.to(atom_coords_denoised)

        step_scale = self.diffusion.step_scale
        sigma_t = sch.sigma_t
        denoised_over_sigma = (x_noisy - atom_coords_denoised) / t_hat
        x_next = x_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma

        x_next = _nan_fallback(x_next, atom_coords, label=f"forward step {step_idx}")

        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sigma_t,
            "t_hat": t_hat,
            "noise_var": noise_var,
            "step_scale": float(step_scale),
            "center_mean": center_mean,
        }
        return x_next, eps, random_r, random_tr, meta

    @torch.inference_mode()
    def single_backward_step(
        self,
        atom_coords: torch.Tensor,
        step_idx: int,
        n_fixed_point: int = 4,
        center_mean_before: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Approximate inverse step (re-noising toward higher sigma).

        Uses fixed-point iteration to recover :math:`x^{\\mathrm{noisy}}` from
        :math:`x_{i+1}` under the Boltz update, then samples a fresh
        :math:`\\varepsilon` and inverts SE(3) to obtain coordinates before the
        forward sub-step. This is a valid proposal kernel for path moves; the
        explicit Metropolis correction uses :mod:`genai_tps.backends.boltz.path_probability`.
        """
        self._assert_coords_device(atom_coords)
        if self._schedule is None:
            self.build_schedule()
        if step_idx < 0 or step_idx >= len(self.schedule):
            raise ValueError("step_idx out of range")
        sch = self.schedule[step_idx]
        b = atom_coords.shape[0]
        random_r, random_tr = compute_random_augmentation(
            b, device=atom_coords.device, dtype=atom_coords.dtype
        )

        step_scale = self.diffusion.step_scale
        t_hat = sch.t_hat
        sigma_t = sch.sigma_t
        alpha = step_scale * (sigma_t - t_hat) / t_hat

        x_noisy = self._solve_x_noisy_from_output(atom_coords, step_idx, n_fixed_point)

        noise_var = sch.noise_var
        eps = sqrt(noise_var) * torch.randn(x_noisy.shape, device=x_noisy.device, dtype=x_noisy.dtype)
        x_tilde = x_noisy - eps
        y = torch.einsum("bds,bmd->bms", random_r.transpose(-1, -2), x_tilde - random_tr)
        if center_mean_before is None:
            center_mean_before = torch.zeros(
                (b, 1, 3), device=atom_coords.device, dtype=atom_coords.dtype
            )
        else:
            center_mean_before = center_mean_before.to(
                device=atom_coords.device, dtype=atom_coords.dtype
            )
        x_prev = y + center_mean_before

        x_prev = _nan_fallback(x_prev, atom_coords, label=f"backward step {step_idx}")

        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sigma_t,
            "t_hat": t_hat,
            "noise_var": noise_var,
            "step_scale": float(step_scale),
            "alpha": float(alpha),
        }
        return x_prev, eps, random_r, random_tr, meta

    @torch.inference_mode()
    def generate_segment(
        self,
        atom_coords: torch.Tensor,
        start_step: int,
        end_step: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
        """Apply forward steps ``start_step .. end_step - 1`` inclusive."""
        traj = [atom_coords]
        eps_list: list[torch.Tensor] = []
        r_list: list[torch.Tensor] = []
        t_list: list[torch.Tensor] = []
        meta_list: list[dict[str, Any]] = []
        x = atom_coords
        for s in range(start_step, end_step):
            x, eps, rr, tr, meta = self.single_forward_step(x, s)
            traj.append(x)
            eps_list.append(eps)
            r_list.append(rr)
            t_list.append(tr)
            meta_list.append(meta)
        return traj, eps_list, r_list, t_list, meta_list

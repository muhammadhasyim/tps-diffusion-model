"""GPU-native Boltz-2 diffusion stepping (no OpenPathSampling imports)."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F

from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.model.modules.utils import compute_random_augmentation


@dataclass(frozen=True)
class StepSchedule:
    """One sampling step: sigma_tm -> sigma_t with gamma on the upper index."""

    sigma_tm: float
    sigma_t: float
    gamma: float
    t_hat: float
    noise_var: float


class BoltzSamplerCore:
    """Wraps :class:`boltz.model.modules.diffusionv2.AtomDiffusion` for TPS."""

    def __init__(
        self,
        diffusion_module: Any,
        atom_mask: torch.Tensor,
        network_condition_kwargs: dict[str, Any],
        multiplicity: int = 1,
    ) -> None:
        self.diffusion = diffusion_module
        self.atom_mask = atom_mask
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

    def sample_initial_noise(self) -> torch.Tensor:
        self.build_schedule()
        sigmas = self.diffusion.sample_schedule(self.num_sampling_steps)
        init_sigma = float(sigmas[0].item())
        shape = self._shape()
        return init_sigma * torch.randn(shape, device=self.diffusion.device, dtype=torch.float32)

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
        kw = dict(self.network_condition_kwargs)
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

        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sigma_t,
            "t_hat": t_hat,
            "noise_var": noise_var,
            "step_scale": float(step_scale),
            "center_mean": center_mean,
        }
        return x_next, eps, random_r, random_tr, meta

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

        x_out = atom_coords
        x_noisy = x_out
        for _ in range(n_fixed_point):
            kw_b = dict(self.network_condition_kwargs)
            kw_b.pop("multiplicity", None)
            x_hat = self.diffusion.preconditioned_network_forward(
                x_noisy,
                t_hat,
                network_condition_kwargs=dict(multiplicity=b, **kw_b),
            )
            x_noisy = (x_out + alpha * x_hat) / (1.0 + alpha)

        noise_var = sch.noise_var
        eps = sqrt(noise_var) * torch.randn(x_noisy.shape, device=x_noisy.device, dtype=x_noisy.dtype)
        x_tilde = x_noisy - eps
        y = torch.einsum("bds,bmd->bms", random_r.transpose(-1, -2), x_tilde - random_tr)
        if center_mean_before is None:
            center_mean_before = torch.zeros(
                (b, 1, 3), device=atom_coords.device, dtype=atom_coords.dtype
            )
        x_prev = y + center_mean_before

        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sigma_t,
            "t_hat": t_hat,
            "noise_var": noise_var,
            "step_scale": float(step_scale),
            "alpha": float(alpha),
        }
        return x_prev, eps, random_r, random_tr, meta

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

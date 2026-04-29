"""GPU-native Boltz-2 diffusion stepping (no OpenPathSampling imports)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F

from boltz.model.loss.diffusionv2 import weighted_rigid_align
from boltz.model.modules.utils import compute_random_augmentation

from genai_tps.training.quotient_projection import horizontal_projection

logger = logging.getLogger(__name__)

_NAN_FALLBACK_LOG_COUNT = 0
_NAN_FALLBACK_LOG_MAX = 5


def _nan_fallback_enabled() -> bool:
    """Check env var at call time so tests can toggle it dynamically."""
    return os.environ.get("GENAI_TPS_NAN_FALLBACK", "0").lower() in ("1", "true", "yes")


def _nan_fallback(
    x_next: torch.Tensor,
    x_fallback: torch.Tensor,
    *,
    label: str,
    allow_fallback: bool | None = None,
) -> torch.Tensor:
    """Check for non-finite elements in *x_next*.

    By default (strict mode), raises ``RuntimeError`` when NaN/Inf values
    are detected.  Set ``allow_fallback=True`` or the environment variable
    ``GENAI_TPS_NAN_FALLBACK=1`` to replace non-finite values with
    *x_fallback* instead (useful for exploratory runs).
    """
    global _NAN_FALLBACK_LOG_COUNT
    bad = ~torch.isfinite(x_next)
    if not bad.any():
        return x_next

    do_fallback = allow_fallback if allow_fallback is not None else _nan_fallback_enabled()
    n_bad = int(bad.sum().item())
    n_tot = int(x_next.numel())

    if not do_fallback:
        raise RuntimeError(
            f"{label}: {n_bad}/{n_tot} non-finite values detected in diffusion step. "
            "This indicates numerical instability in the score network or schedule. "
            "Set GENAI_TPS_NAN_FALLBACK=1 to replace with fallback coords (exploratory mode)."
        )

    _NAN_FALLBACK_LOG_COUNT += 1
    if _NAN_FALLBACK_LOG_COUNT <= _NAN_FALLBACK_LOG_MAX:
        logger.warning(
            "%s: %d/%d non-finite values replaced with fallback coords "
            "(NaN-cascade guard #%d)",
            label, n_bad, n_tot, _NAN_FALLBACK_LOG_COUNT,
        )
    return torch.where(bad, x_fallback, x_next)


def _cuda_h_include_search_dirs() -> list[str]:
    """Ordered directories that may contain ``cuda.h`` (Triton JIT helper build)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(path: str) -> None:
        p = os.path.normpath(os.path.expanduser(path))
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)

    for key in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(key)
        if v:
            add(os.path.join(v, "include"))
    nvcc = shutil.which("nvcc")
    if nvcc:
        root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        add(os.path.join(root, "include"))
    add("/usr/local/cuda/include")
    add("/usr/lib/cuda/include")
    prefix = sys.prefix
    add(os.path.join(prefix, "targets/x86_64-linux/include"))
    add(os.path.join(prefix, "include"))
    return ordered


def _first_dir_with_cuda_h() -> str | None:
    for d in _cuda_h_include_search_dirs():
        if os.path.isfile(os.path.join(d, "cuda.h")):
            return d
    return None


def _prepend_cpath_for_cuda_toolkit(include_dir: str) -> None:
    """Prepend *include_dir* to CPATH / C_INCLUDE_PATH for subprocess gcc (Triton)."""
    for var in ("CPATH", "C_INCLUDE_PATH"):
        prev = os.environ.get(var, "")
        if prev:
            os.environ[var] = include_dir + os.pathsep + prev
        else:
            os.environ[var] = include_dir


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

    ``quotient_space_sampling=True`` applies the horizontal projection in a
    mask-centered coordinate frame, removing rotational vertical motion.  The
    Boltz random translation is kept as an auxiliary stochastic draw in the
    extended path state rather than projected out as a CoM-free SE(3)
    coordinate.
    """

    def __init__(
        self,
        diffusion_module: Any,
        atom_mask: torch.Tensor,
        network_condition_kwargs: dict[str, Any],
        multiplicity: int = 1,
        compile_model: bool = False,
        n_fixed_point: int = 4,
        inference_dtype: "torch.dtype | None" = None,
        quotient_space_sampling: bool = False,
    ) -> None:
        self.compile_model = compile_model
        self.n_fixed_point = n_fixed_point
        self.inference_dtype = inference_dtype
        self.quotient_space_sampling = quotient_space_sampling
        self.diffusion = diffusion_module
        if compile_model:
            # torch.compile → inductor → Triton compiles driver.c with #include "cuda.h".
            # Conda PyTorch envs often lack toolkit headers under sys.prefix; system CUDA is
            # typically /usr/local/cuda/include — expose it to gcc via CPATH.
            cuda_inc = _first_dir_with_cuda_h()
            if cuda_inc is not None:
                _prepend_cpath_for_cuda_toolkit(cuda_inc)
                logger.debug("torch.compile: prepended CPATH for CUDA headers: %s", cuda_inc)
            elif torch.cuda.is_available():
                logger.warning(
                    "compile_model=True but cuda.h was not found in CUDA_HOME, nvcc parent, "
                    "/usr/local/cuda/include, or conda targets/*/include. "
                    "torch.compile may fail when Triton builds GPU helpers; "
                    "install the CUDA toolkit dev headers or set CUDA_HOME."
                )
            self.diffusion.preconditioned_network_forward = torch.compile(
                self.diffusion.preconditioned_network_forward,
                mode="reduce-overhead",
            )
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

    def _masked_center(self, atom_coords: torch.Tensor) -> torch.Tensor:
        """Center real atoms with the mask-normalized COM and zero padded atoms."""
        mask = self.atom_mask.to(device=atom_coords.device, dtype=atom_coords.dtype)
        mask_3 = mask.unsqueeze(-1)
        center = self._masked_center_of_mass(atom_coords)
        return (atom_coords - center) * mask_3

    def _masked_center_of_mass(self, atom_coords: torch.Tensor) -> torch.Tensor:
        """Mask-normalized center of mass over real atoms."""
        mask = self.atom_mask.to(device=atom_coords.device, dtype=atom_coords.dtype)
        mask_3 = mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        return (atom_coords * mask_3).sum(dim=1, keepdim=True) / denom

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
        n_fixed_point: int | None = None,
    ) -> torch.Tensor:
        """Recover x_noisy from the step output x_out via rescaled fixed-point iteration.

        Solves the implicit equation produced by the Boltz denoising update:

            x_out = (1 + α) x_noisy − α D(x_noisy, t̂)

        In quotient-space mode the denoiser term is replaced by its horizontal
        projection, matching the forward sampler:

            x_out = (1 + α) x_noisy − α P_x(D(x_noisy, t̂))

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
        applied inside the loop to match the forward sampling kernel.  The
        quotient-space sampler has its own projected branch and does not use
        the alignment branch.
        """
        n_fp = n_fixed_point if n_fixed_point is not None else self.n_fixed_point
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
        for _ in range(n_fp):
            kw = _kwargs_for_preconditioned_forward(dict(self.network_condition_kwargs))
            kw.pop("multiplicity", None)
            if self.inference_dtype is not None and x_noisy.device.type == "cuda":
                with torch.autocast("cuda", dtype=self.inference_dtype):
                    x_hat = self.diffusion.preconditioned_network_forward(
                        x_noisy,
                        t_hat,
                        network_condition_kwargs=dict(multiplicity=b, **kw),
                    )
                x_hat = x_hat.to(x_noisy.dtype)
            else:
                x_hat = self.diffusion.preconditioned_network_forward(
                    x_noisy,
                    t_hat,
                    network_condition_kwargs=dict(multiplicity=b, **kw),
                )
            if self.diffusion.alignment_reverse_diff and not self.quotient_space_sampling:
                x_noisy = weighted_rigid_align(
                    x_noisy.float(),
                    x_hat.float(),
                    self.atom_mask.float(),
                    self.atom_mask.float(),
                ).to(x_hat)
            if self.quotient_space_sampling:
                x_noisy_c = self._masked_center(x_noisy)
                x_hat = horizontal_projection(
                    x_noisy_c.float(),
                    x_hat.float(),
                    self.atom_mask.float(),
                ).to(x_hat.dtype)
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
        center_mean_before: torch.Tensor | None = None,
        *,
        n_fixed_point: int | None = None,
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
            lower, backward-generated frame ``s0`` when recovering noise for a
            backward-shot prefix).
        center_mean_before
            The centroid that was subtracted from ``x_prev`` before augmentation
            during the step that produced ``x_next``.  Must be the same value that
            was passed as ``center_mean_before`` to :meth:`single_backward_step`
            when ``x_prev`` was generated.  For backward-generated frames this is
            the zeros tensor stored on the snapshot as ``center_mean_before_step``.
            If ``None``, zeros are used (consistent with ``single_backward_step``
            default).
        n_fixed_point
            Fixed-point iterations for the implicit :math:`x^{\\mathrm{noisy}}` solve.
        """
        self._assert_coords_device(x_prev)
        self._assert_coords_device(x_next)
        if self._schedule is None:
            self.build_schedule()
        sch = self.schedule[step_idx]
        if center_mean_before is None:
            cm = torch.zeros((x_prev.shape[0], 1, 3), device=x_prev.device, dtype=x_prev.dtype)
        else:
            cm = center_mean_before.to(device=x_prev.device, dtype=x_prev.dtype)
        x_aug = torch.einsum("bmd,bds->bms", x_prev - cm, random_r) + random_tr

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
        n_fixed_point: int | None = None,
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
        n_fp = n_fixed_point if n_fixed_point is not None else self.n_fixed_point
        sch = self.schedule[step_idx]
        b = x_prev.shape[0]
        cm = center_mean_before.to(device=x_prev.device, dtype=x_prev.dtype)
        y = x_prev - cm
        x_tilde = torch.einsum("bmd,bds->bms", y, random_r) + random_tr

        x_noisy = self._solve_x_noisy_from_output(x_next, step_idx, n_fp)

        eps = x_noisy - x_tilde
        meta = {
            "sigma_tm": sch.sigma_tm,
            "sigma_t": sch.sigma_t,
            "t_hat": sch.t_hat,
            "noise_var": sch.noise_var,
            "step_scale": float(self.diffusion.step_scale),
        }
        return eps, meta

    def _single_forward_step_core(
        self,
        atom_coords: torch.Tensor,
        step_idx: int,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
        """Shared forward-step implementation (with or without autograd).

        When ``eps`` is ``None``, samples Gaussian noise with variance ``noise_var``.
        Otherwise uses the provided noise tensor (trajectory replay for RL).

        Returns
        -------
        x_next, eps, random_r, random_tr, meta, atom_coords_denoised, x_noisy
            ``x_noisy`` is the noisy state fed into the denoiser after any
            ``alignment_reverse_diff``. ``atom_coords_denoised`` is the network
            output before the EDM update to ``x_next``.
        """
        self._assert_coords_device(atom_coords)
        if self._schedule is None:
            self.build_schedule()
        sch = self.schedule[step_idx]
        b = atom_coords.shape[0]
        random_r, random_tr = compute_random_augmentation(
            b, device=atom_coords.device, dtype=atom_coords.dtype
        )
        center_mean = self._masked_center_of_mass(atom_coords)
        x = self._masked_center(atom_coords)
        x = torch.einsum("bmd,bds->bms", x, random_r) + random_tr

        noise_var = sch.noise_var
        if eps is None:
            eps = sqrt(noise_var) * torch.randn(x.shape, device=x.device, dtype=x.dtype)
        else:
            if eps.shape != x.shape:
                raise ValueError(f"eps shape {eps.shape} != expected {x.shape}")
        x_noisy = x + eps

        t_hat = sch.t_hat
        sample_ids = torch.arange(b, device=x_noisy.device)
        kw = _kwargs_for_preconditioned_forward(dict(self.network_condition_kwargs))
        kw.pop("multiplicity", None)
        kwargs = dict(multiplicity=sample_ids.numel(), **kw)
        if self.inference_dtype is not None and x_noisy.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.inference_dtype):
                atom_coords_denoised = self.diffusion.preconditioned_network_forward(
                    x_noisy, t_hat, network_condition_kwargs=kwargs
                )
            atom_coords_denoised = atom_coords_denoised.to(x_noisy.dtype)
        else:
            atom_coords_denoised = self.diffusion.preconditioned_network_forward(
                x_noisy, t_hat, network_condition_kwargs=kwargs
            )

        if self.quotient_space_sampling:
            # Quotient-space ODE sampler (arXiv:2604.21809, Eq. 13):
            # Project the velocity (denoised_over_sigma) through P_x to
            # remove rotational vertical components in the centered shape
            # frame. Boltz's random translation is retained as auxiliary noise
            # in the extended path state.
            step_scale = self.diffusion.step_scale
            sigma_t = sch.sigma_t
            denoised_over_sigma = (x_noisy - atom_coords_denoised) / t_hat
            # The inertia tensor K requires a zero-COM anchor over real atoms.
            x_noisy_c = self._masked_center(x_noisy)
            v_proj = horizontal_projection(
                x_noisy_c.float(), denoised_over_sigma.float(), self.atom_mask.float()
            ).to(x_noisy.dtype)
            x_next = x_noisy + step_scale * (sigma_t - t_hat) * v_proj
        elif self.diffusion.alignment_reverse_diff:
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
        else:
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
        return x_next, eps, random_r, random_tr, meta, atom_coords_denoised, x_noisy

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
        x_next, eps, random_r, random_tr, meta, _, _ = self._single_forward_step_core(
            atom_coords, step_idx, eps=None
        )
        return x_next, eps, random_r, random_tr, meta

    def single_forward_step_trainable(
        self,
        atom_coords: torch.Tensor,
        step_idx: int,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
        """Forward step with autograd through the score network (RL fine-tuning).

        Callers should run rollout under ``torch.no_grad`` and replay with stored
        ``eps`` and detached ``x_prev`` as appropriate.
        """
        return self._single_forward_step_core(atom_coords, step_idx, eps=eps)

    @torch.inference_mode()
    def single_backward_step(
        self,
        atom_coords: torch.Tensor,
        step_idx: int,
        n_fixed_point: int | None = None,
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
        n_fp = n_fixed_point if n_fixed_point is not None else self.n_fixed_point
        sch = self.schedule[step_idx]
        b = atom_coords.shape[0]
        random_r, random_tr = compute_random_augmentation(
            b, device=atom_coords.device, dtype=atom_coords.dtype
        )

        step_scale = self.diffusion.step_scale
        t_hat = sch.t_hat
        sigma_t = sch.sigma_t
        alpha = step_scale * (sigma_t - t_hat) / t_hat

        x_noisy = self._solve_x_noisy_from_output(atom_coords, step_idx, n_fp)

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

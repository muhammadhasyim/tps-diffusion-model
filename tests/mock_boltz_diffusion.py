"""Minimal :class:`boltz.model.modules.diffusionv2.AtomDiffusion` stand-in for unit tests."""

from __future__ import annotations

import torch


class MockDiffusion(torch.nn.Module):
    """Identity denoiser in preconditioned space (no checkpoint)."""

    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        d = device or torch.device("cpu")
        self.num_sampling_steps = 3
        self.sigma_min = 0.01
        self.sigma_max = 10.0
        self.sigma_data = 16.0
        self.rho = 7.0
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.step_scale_random = None
        self.alignment_reverse_diff = False
        self.register_buffer("zero", torch.tensor(0.0, device=d), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.zero.device

    def sample_schedule(self, num_sampling_steps=None):
        n = num_sampling_steps if num_sampling_steps is not None else self.num_sampling_steps
        steps = torch.arange(n, dtype=torch.float32, device=self.device)
        inv_rho = 1.0 / self.rho
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (n - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def preconditioned_network_forward(self, noised_atom_coords, sigma, network_condition_kwargs=None):
        return noised_atom_coords

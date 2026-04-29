"""EDM noise distribution and loss weight for Boltz 2 diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EDMNoiseParams:
    """Parameters matching AtomDiffusion's log-normal noise distribution.

    Defaults are Boltz 2's values: sigma_data=16.0, P_mean=-1.2, P_std=1.5.
    """

    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5

    @classmethod
    def from_diffusion(cls, diffusion: Any) -> EDMNoiseParams:
        """Extract noise params from an AtomDiffusion-like object."""
        return cls(
            sigma_data=float(diffusion.sigma_data),
            P_mean=float(diffusion.P_mean),
            P_std=float(diffusion.P_std),
        )


def sample_noise_sigma(
    batch_size: int,
    params: EDMNoiseParams,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample sigma ~ sigma_data * exp(P_mean + P_std * N(0,1))."""
    z = torch.randn(batch_size, device=device)
    return params.sigma_data * (params.P_mean + params.P_std * z).exp()


def edm_loss_weight(sigma: torch.Tensor, params: EDMNoiseParams) -> torch.Tensor:
    """EDM loss weight: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
    sd = params.sigma_data
    return (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

"""Tests for multi-protein WDSM training internals."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from genai_tps.training.config import WeightedDSMConfig
from genai_tps.training.multi_protein_trainer import (
    _coords_from_boltz_batch,
    _multi_protein_batch_loss,
)
from genai_tps.training.noise_schedule import EDMNoiseParams


class TinyDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.sigma_data = 16.0
        self.P_mean = -1.2
        self.P_std = 1.2

    def preconditioned_network_forward(self, x_noisy, sigma, network_condition_kwargs):
        assert network_condition_kwargs["multiplicity"] == x_noisy.shape[0]
        return self.scale * x_noisy


def test_coords_from_boltz_batch_squeezes_single_conformer_axis():
    coords = torch.zeros(2, 1, 4, 3)

    out = _coords_from_boltz_batch({"coords": coords})

    assert out.shape == (2, 4, 3)


def test_multi_protein_batch_loss_backprops_through_diffusion(monkeypatch):
    diffusion = TinyDiffusion()
    model = SimpleNamespace(structure_module=diffusion)
    batch = {
        "coords": torch.zeros(2, 1, 4, 3),
        "logw": torch.zeros(2),
        "atom_pad_mask": torch.ones(2, 4),
    }

    def fake_trunk(_model, feats, *, recycling_steps):
        return feats["atom_pad_mask"], {
            "s_trunk": torch.zeros(2, 1, 1),
            "s_inputs": torch.zeros(2, 1, 1),
            "feats": feats,
            "diffusion_conditioning": {},
        }

    monkeypatch.setattr(
        "genai_tps.training.multi_protein_trainer.compute_boltz2_trunk_to_network_kwargs",
        fake_trunk,
    )

    loss = _multi_protein_batch_loss(
        model=model,
        frozen_diffusion=TinyDiffusion(),
        batch=batch,
        cfg=WeightedDSMConfig(beta=0.0),
        loss_type="cartesian",
        noise_params=EDMNoiseParams(sigma_data=16.0, P_mean=-1.2, P_std=1.2),
        recycling_steps=1,
        training_mode=True,
    )
    loss.backward()

    assert diffusion.scale.grad is not None
    assert torch.isfinite(loss)

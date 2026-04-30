"""Tests for :mod:`genai_tps.backends.boltz.utils`."""

from __future__ import annotations

import torch

from genai_tps.backends.boltz.utils import build_network_condition_kwargs


def test_build_network_condition_kwargs_merges_bundle():
    steer = {
        "fk_steering": False,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
    }
    bundle = {
        "s_trunk": torch.zeros(1),
        "s_inputs": torch.ones(1),
        "feats": {"k": torch.tensor(2.0)},
        "diffusion_conditioning": {"q": torch.tensor(3.0)},
        "steering_args": steer,
    }
    out = build_network_condition_kwargs(bundle)
    assert set(out.keys()) == {
        "s_trunk",
        "s_inputs",
        "feats",
        "diffusion_conditioning",
        "steering_args",
    }
    assert torch.equal(out["s_trunk"], bundle["s_trunk"])
    assert out["steering_args"] is steer

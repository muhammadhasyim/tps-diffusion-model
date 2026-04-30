"""Boltz-2 loading helpers and conditioning bundles for :class:`BoltzSamplerCore`."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def default_steering_args() -> dict[str, Any]:
    """Steering config with potentials disabled (suitable for path sampling)."""
    from boltz.main import BoltzSteeringParams  # noqa: PLC0415 — defer heavy boltz/lightning import

    d = asdict(BoltzSteeringParams())
    d["fk_steering"] = False
    d["physical_guidance_update"] = False
    d["contact_guidance_update"] = False
    return d


def load_conditioning_bundle(path: str | Path) -> dict[str, Any]:
    """Load a pickled dict with keys ``s_trunk``, ``s_inputs``, ``feats``, ``diffusion_conditioning``."""
    p = Path(path)
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("Conditioning bundle must be a dict.")
    required = ("feats", "diffusion_conditioning", "s_trunk", "s_inputs")
    for k in required:
        if k not in obj:
            raise KeyError(f"Bundle missing key {k!r}")
    return obj


def build_network_condition_kwargs(bundle: dict[str, Any]) -> dict[str, Any]:
    """Merge bundle tensors with default steering for ``AtomDiffusion.sample``."""
    # Do not use ``.get(k, default())`` — the default is evaluated eagerly in Python.
    steering = bundle["steering_args"] if "steering_args" in bundle else default_steering_args()
    out = {
        "s_trunk": bundle["s_trunk"],
        "s_inputs": bundle["s_inputs"],
        "feats": bundle["feats"],
        "diffusion_conditioning": bundle["diffusion_conditioning"],
        "steering_args": steering,
    }
    return out


def load_boltz2_module(checkpoint: str | Path, device: str | torch.device) -> Any:
    """Load a :class:`boltz.model.models.boltz2.Boltz2` Lightning module from checkpoint."""
    from boltz.model.models.boltz2 import Boltz2

    return Boltz2.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        strict=False,
    )

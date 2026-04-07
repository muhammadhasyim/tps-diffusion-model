"""Terminal rewards from GPU-native PoseBusters-style checks (no ``posebusters`` package)."""

from __future__ import annotations

from typing import Any

import torch

from genai_tps.analysis.posebusters_gpu import GPUPoseBustersEvaluator, pass_fraction_from_gpu_row
from genai_tps.backends.boltz.snapshot import BoltzSnapshot

__all__ = ["gpu_pass_fraction_reward_from_coords", "snapshot_from_coords_for_reward"]


def snapshot_from_coords_for_reward(
    coords_batched: torch.Tensor,
    *,
    step_index: int,
    sigma: float | None,
) -> BoltzSnapshot:
    """Minimal :class:`BoltzSnapshot` for :meth:`GPUPoseBustersEvaluator.evaluate_snapshot`."""
    c = coords_batched.detach()
    if c.dim() == 2:
        c = c.unsqueeze(0)
    return BoltzSnapshot.from_gpu_batch(
        c,
        step_index=step_index,
        sigma=sigma,
        defer_numpy_coords=True,
    )


def gpu_pass_fraction_reward_from_coords(
    evaluator: GPUPoseBustersEvaluator,
    coords_batched: torch.Tensor,
    *,
    step_index: int = 0,
    sigma: float | None = None,
) -> float:
    """Mean pass rate over :mod:`genai_tps.analysis.posebusters_gpu` boolean checks."""
    snap = snapshot_from_coords_for_reward(
        coords_batched, step_index=step_index, sigma=sigma
    )
    row = evaluator.evaluate_snapshot(snap)
    return float(pass_fraction_from_gpu_row(row))

"""GPU tensor pool and batched trajectory helpers (OPS orchestration)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot

import numpy as np


@dataclass
class GPUTensorPool:
    """Keeps references to GPU tensors until checkpoints materialize to CPU."""

    slots: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    def register(self, key: str, tensor: torch.Tensor) -> None:
        self.slots.append((key, tensor))

    def clear(self) -> None:
        self.slots.clear()


def snapshot_from_gpu(
    coords_gpu: torch.Tensor,
    step_index: int,
    eps: torch.Tensor | None,
    rotation_R: torch.Tensor | None,
    translation_t: torch.Tensor | None,
    sigma: float | None,
    center_mean_before_step: torch.Tensor | None = None,
) -> BoltzSnapshot:
    """Build :class:`BoltzSnapshot` via :meth:`BoltzSnapshot.from_gpu_batch` (GPU-native)."""
    return BoltzSnapshot.from_gpu_batch(
        coords_gpu,
        step_index=step_index,
        sigma=sigma,
        eps_used=eps,
        rotation_R=rotation_R,
        translation_t=translation_t,
        center_mean_before_step=center_mean_before_step,
    )


def batched_forward_segments(
    core: BoltzSamplerCore,
    starts: torch.Tensor,
    start_steps: list[int],
    segment_lengths: list[int],
) -> list[tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]]:
    """Run several forward segments in parallel (shared batch dimension).

    Parameters
    ----------
    starts
        (B, M, 3) batch of starting coordinates.
    start_steps
        Length B list of step indices where each segment begins.
    segment_lengths
        Length B list of how many forward steps to take per segment.

    Returns
    -------
    One list entry per batch element with (coords_traj, eps_list, meta_list).
    """
    b = starts.shape[0]
    if len(start_steps) != b or len(segment_lengths) != b:
        raise ValueError("Batch dimension mismatch")
    results: list[tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]] = []
    for bi in range(b):
        x = starts[bi : bi + 1]
        s0 = start_steps[bi]
        ln = segment_lengths[bi]
        traj, eps_l, _, _, meta_l = core.generate_segment(x, s0, s0 + ln)
        results.append((traj, eps_l, meta_l))
    return results

"""OpenPathSampling snapshot with lazy GPU tensor storage."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from openpathsampling.engines import SnapshotDescriptor
from openpathsampling.engines import features as feats
from openpathsampling.engines.snapshot import SnapshotFactory

_BoltzSnapshotBase = SnapshotFactory(
    name="_BoltzSnapshotBase",
    features=[feats.coordinates],
    description="Coordinates-only snapshot base for Boltz TPS.",
)


class BoltzSnapshot(_BoltzSnapshotBase):
    """Snapshot storing optional GPU tensors for diffusion extended state."""

    def __init__(
        self,
        coordinates: np.ndarray,
        topology: Any = None,
        *,
        tensor_coords_gpu: torch.Tensor | None = None,
        step_index: int = 0,
        sigma: float | None = None,
        eps_used: torch.Tensor | None = None,
        rotation_R: torch.Tensor | None = None,
        translation_t: torch.Tensor | None = None,
        center_mean_before_step: torch.Tensor | None = None,
    ) -> None:
        super().__init__(coordinates=coordinates)
        self.topology = topology
        self._tensor_coords_gpu = tensor_coords_gpu
        self.step_index = int(step_index)
        self.sigma = sigma
        self.eps_used = eps_used
        self.rotation_R = rotation_R
        self.translation_t = translation_t
        self.center_mean_before_step = center_mean_before_step

    def lazy_numpy_coordinates(self) -> np.ndarray:
        """Materialize OPS coordinates from GPU if needed."""
        if self._tensor_coords_gpu is not None:
            return self._tensor_coords_gpu.detach().cpu().numpy()
        return np.asarray(self.coordinates, dtype=np.float32)

    def clear_cache(self) -> None:
        """Optional hook for OPS engines that clear snapshot caches."""
        pass

    def create_reversed(self) -> BoltzSnapshot:
        """Partner snapshot for OPS backward integration (same extended state)."""
        coords = np.asarray(self.coordinates, dtype=np.float32)
        rev = BoltzSnapshot(
            coords,
            topology=self.topology,
            tensor_coords_gpu=self._tensor_coords_gpu,
            step_index=self.step_index,
            sigma=self.sigma,
            eps_used=self.eps_used,
            rotation_R=self.rotation_R,
            translation_t=self.translation_t,
            center_mean_before_step=self.center_mean_before_step,
        )
        rev._reversed = self
        self._reversed = rev
        return rev


def boltz_snapshot_descriptor(n_atoms: int, n_spatial: int = 3) -> SnapshotDescriptor:
    return SnapshotDescriptor.construct(
        snapshot_class=BoltzSnapshot,
        snapshot_dimensions={"n_atoms": n_atoms, "n_spatial": n_spatial},
    )

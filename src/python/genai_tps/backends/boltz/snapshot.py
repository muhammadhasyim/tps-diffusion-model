"""OpenPathSampling snapshots with GPU-native coordinates for Boltz dynamics.

**Contract:** Trajectories produced by :class:`BoltzSamplerCore` and
:class:`BoltzDiffusionEngine` carry a **torch.Tensor** of atom coordinates on
the **same device as the diffusion module** (typically CUDA). Each
:class:`BoltzSnapshot` from :meth:`BoltzSnapshot.from_gpu_batch` **clones** that
tensor so frames do not alias the engine’s live buffer. The NumPy ``coordinates``
array exists for OpenPathSampling storage and serialization only; dynamics and
path-probability code use :attr:`tensor_coords_gpu` (or :meth:`BoltzSnapshot.from_gpu_batch`).
"""

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

# OPS stores plain ``coordinates`` on the generated base; we override with a property,
# so assignments must land in this backing attribute (not ``super().coordinates``).
_BOLT_COORDS = "_bolt_coords_numpy"


class BoltzSnapshot(_BoltzSnapshotBase):
    """Snapshot with GPU coordinate tensor for Boltz diffusion (OPS-compatible)."""

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
        generated_by_backward: bool = False,
        defer_numpy_coords: bool = False,
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
        self.generated_by_backward = bool(generated_by_backward)
        self._defer_numpy_coords = bool(defer_numpy_coords) and tensor_coords_gpu is not None
        self._numpy_coords_cache: np.ndarray | None = None

    @classmethod
    def from_gpu_batch(
        cls,
        coords_gpu: torch.Tensor,
        *,
        step_index: int = 0,
        sigma: float | None = None,
        eps_used: torch.Tensor | None = None,
        rotation_R: torch.Tensor | None = None,
        translation_t: torch.Tensor | None = None,
        center_mean_before_step: torch.Tensor | None = None,
        topology: Any = None,
        generated_by_backward: bool = False,
        defer_numpy_coords: bool = True,
    ) -> BoltzSnapshot:
        """Build a snapshot from **batched** GPU coordinates (canonical runtime constructor).

        Parameters
        ----------
        coords_gpu
            Shape ``(batch, n_atoms, 3)``. Must live on the target accelerator. A
            **detach().clone()** is stored so each snapshot does not alias the engine’s
            live coordinate buffer (otherwise trajectories and NPZ checkpoints can
            see every frame overwritten by the latest step).
        defer_numpy_coords
            If True (default), avoid a GPU→CPU copy until :attr:`coordinates` is read.
        """
        if not isinstance(coords_gpu, torch.Tensor):
            raise TypeError("coords_gpu must be a torch.Tensor")
        if coords_gpu.dim() != 3:
            raise ValueError("coords_gpu must have shape (batch, n_atoms, 3)")
        m = int(coords_gpu.shape[1])
        placeholder = np.zeros((m, 3), dtype=np.float32)
        stored = coords_gpu.detach().clone()
        return cls(
            placeholder,
            topology=topology,
            tensor_coords_gpu=stored,
            step_index=step_index,
            sigma=sigma,
            eps_used=eps_used,
            rotation_R=rotation_R,
            translation_t=translation_t,
            center_mean_before_step=center_mean_before_step,
            generated_by_backward=generated_by_backward,
            defer_numpy_coords=defer_numpy_coords,
        )

    @property
    def coordinates(self) -> np.ndarray:  # type: ignore[override]
        if self._numpy_coords_cache is not None:
            return self._numpy_coords_cache
        if self._defer_numpy_coords and self._tensor_coords_gpu is not None:
            self._numpy_coords_cache = self._tensor_coords_gpu.detach().cpu().numpy()[0].astype(np.float32)
            self._defer_numpy_coords = False
            return self._numpy_coords_cache
        return object.__getattribute__(self, _BOLT_COORDS)

    @coordinates.setter
    def coordinates(self, value: np.ndarray) -> None:
        self._numpy_coords_cache = None
        self._defer_numpy_coords = False
        object.__setattr__(self, _BOLT_COORDS, np.asarray(value, dtype=np.float32))

    @property
    def tensor_coords(self) -> torch.Tensor | None:
        """Alias for the stored GPU coordinate tensor (batch, n_atoms, 3)."""
        return self._tensor_coords_gpu

    @property
    def has_gpu_coords(self) -> bool:
        return self._tensor_coords_gpu is not None

    @property
    def device(self) -> torch.device:
        """Device of the coordinate tensor; requires :attr:`tensor_coords_gpu`."""
        if self._tensor_coords_gpu is None:
            raise RuntimeError(
                "BoltzSnapshot has no tensor_coords_gpu; device is only defined for GPU-backed snapshots"
            )
        return self._tensor_coords_gpu.device

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
            tensor_coords_gpu=(
                self._tensor_coords_gpu.detach().clone()
                if self._tensor_coords_gpu is not None
                else None
            ),
            step_index=self.step_index,
            sigma=self.sigma,
            eps_used=self.eps_used,
            rotation_R=self.rotation_R,
            translation_t=self.translation_t,
            center_mean_before_step=self.center_mean_before_step,
            generated_by_backward=self.generated_by_backward,
            defer_numpy_coords=False,
        )
        rev._numpy_coords_cache = self._numpy_coords_cache
        rev._defer_numpy_coords = False
        rev._reversed = self
        self._reversed = rev
        return rev


def boltz_snapshot_descriptor(n_atoms: int, n_spatial: int = 3) -> SnapshotDescriptor:
    return SnapshotDescriptor.construct(
        snapshot_class=BoltzSnapshot,
        snapshot_dimensions={"n_atoms": n_atoms, "n_spatial": n_spatial},
    )


def _sync_tensor_device_for_readback(tc: torch.Tensor) -> None:
    """Ensure accelerator work is complete before copying coordinates to host memory."""
    if tc.is_cuda:
        torch.cuda.synchronize()
    elif tc.device.type == "mps" and hasattr(torch, "mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.synchronize()


def snapshot_frame_numpy_copy(snap: BoltzSnapshot) -> np.ndarray:
    """Return one ``(n_atoms, 3)`` float32 frame, copied so NPZ/I/O does not alias GPU buffers.

    Call before ``np.savez`` / disk writes when the trajectory may reuse the same
    underlying tensor across steps. Uses ``detach().clone()`` before host copy for
    snapshots that may still alias shared engine storage.
    """
    tc = snap.tensor_coords
    if tc is not None:
        _sync_tensor_device_for_readback(tc)
        arr = tc.detach().clone().cpu().float().contiguous().numpy()
        return np.asarray(arr[0], dtype=np.float32, order="C").copy()
    c = np.asarray(snap.coordinates, dtype=np.float32)
    if c.ndim == 3:
        return c[0].copy()
    return c.copy()


def snapshot_frame_tensor_view(snap: BoltzSnapshot) -> torch.Tensor:
    """Return one ``(n_atoms, 3)`` frame as a torch tensor without host readback.

    This accessor is intended for tensor-native CVs that can operate directly on
    the stored device tensor and therefore avoid the explicit synchronize +
    GPU->CPU copy used by :func:`snapshot_frame_numpy_copy`.
    """
    tc = snap.tensor_coords
    if tc is not None:
        if tc.dim() == 3:
            return tc[0]
        return tc
    c = torch.as_tensor(np.asarray(snap.coordinates, dtype=np.float32))
    if c.dim() == 3:
        return c[0]
    return c

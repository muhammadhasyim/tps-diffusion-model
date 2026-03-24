"""OpenPathSampling :class:`DynamicsEngine` wrapping :class:`BoltzSamplerCore`."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import openpathsampling as paths
from openpathsampling.engines import DynamicsEngine
from openpathsampling.engines.dynamics_engine import EngineMaxLengthError
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot


class BoltzDiffusionEngine(DynamicsEngine):
    """Diffusion engine: forward (+1) denoises; backward (-1) re-noises."""

    base_snapshot_type = BoltzSnapshot

    _default_options: dict[str, Any] = {
        "n_frames_max": 10_000,
    }

    def __init__(
        self,
        core: BoltzSamplerCore,
        descriptor: Any,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options=options, descriptor=descriptor)
        self.core = core
        self._current_coords: torch.Tensor | None = None
        self._current_step_index: int = 0
        self._integration_direction: int = 1
        self._last_eps: torch.Tensor | None = None
        self._last_R: torch.Tensor | None = None
        self._last_tau: torch.Tensor | None = None
        self._last_sigma: float | None = None
        self._last_meta: dict[str, Any] | None = None
        self._last_center_mean: torch.Tensor | None = None
        self._snapshot_generated_backward: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["core"] = self.core
        return d

    @property
    def current_snapshot(self) -> BoltzSnapshot | None:
        if self._current_coords is None:
            return None
        return BoltzSnapshot.from_gpu_batch(
            self._current_coords,
            step_index=self._current_step_index,
            sigma=self._last_sigma,
            eps_used=self._last_eps,
            rotation_R=self._last_R,
            translation_t=self._last_tau,
            center_mean_before_step=self._last_center_mean,
            generated_by_backward=self._snapshot_generated_backward,
        )

    @current_snapshot.setter
    def current_snapshot(self, snap: BoltzSnapshot) -> None:
        self.check_snapshot_type(snap)
        if snap._tensor_coords_gpu is not None:
            # Never alias trajectory-stored tensors: integration may use the buffer in-place
            # or share views; mutating would corrupt frames still referenced by OPS trajectories.
            self._current_coords = snap._tensor_coords_gpu.detach().clone()
        else:
            c = torch.as_tensor(snap.coordinates, dtype=torch.float32, device=self.core.diffusion.device)
            self._current_coords = c.unsqueeze(0).clone()
        self._current_step_index = snap.step_index
        self._last_eps = snap.eps_used
        self._last_R = snap.rotation_R
        self._last_tau = snap.translation_t
        self._last_sigma = snap.sigma
        self._last_center_mean = getattr(snap, "center_mean_before_step", None)

    def iter_generate(self, initial, running=None, direction=+1, intervals=10, max_length=0):
        self._integration_direction = int(direction)
        try:
            yield from super().iter_generate(
                initial, running=running, direction=direction, intervals=intervals, max_length=max_length
            )
        finally:
            self._integration_direction = 1

    def generate_next_frame(self) -> BoltzSnapshot:
        if self._current_coords is None:
            raise RuntimeError("current_snapshot must be set before integration")
        if self._integration_direction > 0:
            return self._forward_step()
        return self._backward_step()

    def _forward_step(self) -> BoltzSnapshot:
        k = self._current_step_index
        if k >= self.core.num_sampling_steps:
            raise EngineMaxLengthError(
                "Reached end of diffusion schedule (forward).",
                Trajectory([]),
            )
        xn, eps, rr, tr, meta = self.core.single_forward_step(self._current_coords, k)
        self._current_coords = xn
        self._current_step_index = k + 1
        self._last_eps = eps
        self._last_R = rr
        self._last_tau = tr
        self._last_sigma = float(meta["sigma_t"])
        self._last_meta = meta
        self._last_center_mean = meta.get("center_mean")
        self._snapshot_generated_backward = False
        return self.current_snapshot

    def _backward_step(self) -> BoltzSnapshot:
        k = self._current_step_index
        if k <= 0:
            raise EngineMaxLengthError(
                "Cannot step backward before diffusion start.",
                Trajectory([]),
            )
        step_idx = k - 1
        xn, eps, rr, tr, meta = self.core.single_backward_step(
            self._current_coords, step_idx, center_mean_before=self._last_center_mean
        )
        self._current_coords = xn
        self._current_step_index = k - 1
        self._last_eps = eps
        self._last_R = rr
        self._last_tau = tr
        self._last_sigma = float(meta["sigma_tm"])
        self._last_meta = meta
        self._last_center_mean = None
        self._snapshot_generated_backward = True
        return self.current_snapshot

    def generate_n_frames(self, n_frames: int = 1) -> paths.Trajectory:
        """Generate ``n_frames`` without Python round-trips between steps."""
        self.start()
        frames: list[BoltzSnapshot] = []
        for _ in range(n_frames):
            frames.append(self.generate_next_frame())
        traj = Trajectory(frames)
        self.stop(traj)
        return traj

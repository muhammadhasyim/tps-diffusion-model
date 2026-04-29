"""Generative-model TPS orchestration helpers (Boltz diffusion, not OpenMM MD).

Checkpoint serialization (NPZ/json) and ``initial_trajectory`` for Boltz denoising
runs were shared between ``scripts/run_opes_tps.py`` and
``scripts/run_cofolding_tps_demo.py``; they live in this evaluation subpackage /
OpenPathSampling layer, distinct from ``genai_tps.simulation`` (reference MD).

"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, snapshot_frame_numpy_copy
def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (tmp + replace) for crash-safe incremental outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_tps_checkpoint_npz(traj: Trajectory, out_npz: Path, *, mc_step: int) -> bool:
    """Serialize a Boltz trajectory stack to compressed NPZ for checkpoints."""
    stack = []
    for snap in traj:
        if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
            continue
        stack.append(snapshot_frame_numpy_copy(snap))
    if not stack:
        return False
    arr = np.stack(stack, axis=0)
    if not np.any(np.isfinite(arr)):
        return False
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        coords=arr,
        frame_indices=np.arange(arr.shape[0], dtype=np.int32),
        mc_step=np.int64(mc_step),
    )
    return True


def trajectory_checkpoint_callback(work_root: Path) -> Callable[[int, Trajectory], None]:
    """Return callback invoked each MC step to optionally persist trajectory checkpoints."""

    def cb(mc_step: int, traj: Trajectory) -> None:
        ck = work_root / "trajectory_checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        tagged = ck / f"tps_mc_step_{mc_step:08d}.npz"
        wrote = write_tps_checkpoint_npz(traj, tagged, mc_step=mc_step)
        if wrote:
            latest = ck / "latest.npz"
            shutil.copyfile(tagged, latest)
            snap = traj[-1]
            if isinstance(snap, BoltzSnapshot) and snap.tensor_coords is not None:
                lf = snapshot_frame_numpy_copy(snap)
                if np.any(np.isfinite(lf)):
                    last_only = ck / "last_frame_only.npz"
                    np.savez_compressed(
                        last_only,
                        coords=lf.reshape(1, *lf.shape),
                        mc_step=np.int64(mc_step),
                    )
            print(
                f"[TPS-OPES] checkpoint MC step {mc_step} -> {tagged.name}",
                file=sys.stderr,
                flush=True,
            )

    return cb


def opes_state_checkpoint_callback(
    bias: object,
    work_root: Path,
    every: int,
    bias_cv: str,
    bias_cv_names: list[str],
) -> Callable[[int, Trajectory], None]:
    """Periodic OPES state checkpoints for restart and analysis."""

    def cb(mc_step: int, _traj: Trajectory) -> None:
        if every <= 0 or mc_step % every != 0:
            return
        state_dir = work_root / "opes_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        tagged = state_dir / f"opes_state_{mc_step:08d}.json"
        bias.save_state(tagged, bias_cv=bias_cv, bias_cv_names=bias_cv_names)
        latest = state_dir / "opes_state_latest.json"
        shutil.copyfile(tagged, latest)
        print(
            f"[TPS-OPES] OPES state checkpoint MC step {mc_step} "
            f"({bias.n_kernels} kernels, {bias.counter} depositions)",
            file=sys.stderr,
            flush=True,
        )

    return cb


def initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    """Roll out one diffusion trajectory from noise for TPS initialization."""
    x = core.sample_initial_noise()
    n = n_steps if n_steps is not None else core.num_sampling_steps
    snaps = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(snapshot_from_gpu(x, 0, None, None, None, sig0, center_mean_before_step=None))
    for step in range(n):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            snapshot_from_gpu(
                x,
                step + 1,
                eps,
                rr,
                tr,
                float(meta["sigma_t"]),
                center_mean_before_step=meta.get("center_mean"),
            )
        )
    return Trajectory(snaps)


def generate_structures(
    core: BoltzSamplerCore,
    n_samples: int,
    device: torch.device,
    *,
    progress_every: int = 50,
) -> np.ndarray:
    """Sample ``n_samples`` terminal structures by running the full diffusion schedule.

    This is the non-TPS path: independent draws from the generative model without
    OpenPathSampling.

    Parameters
    ----------
    core
        Initialized sampler with schedule built via ``build_schedule``.
    n_samples
        Number of independent draws (fresh noise each time).
    device
        Torch device kept for API parity with callers (sampling uses ``core`` internal state).
    progress_every
        Print progress every this many samples (0 disables).

    Returns
    -------
    ndarray
        Float array of shape ``(n_samples, N_atoms, 3)``.
    """
    _ = device  # API parity with callers that pin CPU/GPU context
    structures: list[np.ndarray] = []
    for i in range(n_samples):
        torch.manual_seed(i)
        x = core.sample_initial_noise()
        for step in range(core.num_sampling_steps):
            x, _eps, _rr, _tr, _meta = core.single_forward_step(x, step)
        structures.append(x[0].detach().cpu().numpy())
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  Generated {i + 1}/{n_samples}", flush=True)
    return np.array(structures)


# Back-compat aliases matching historical script-private names
_write_tps_checkpoint_npz = write_tps_checkpoint_npz
_trajectory_checkpoint_callback = trajectory_checkpoint_callback
_opes_state_checkpoint_callback = opes_state_checkpoint_callback
_initial_trajectory = initial_trajectory

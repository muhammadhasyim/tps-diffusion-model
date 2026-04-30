"""TPS trajectory checkpoints and initial forward diffusion path (shared by driver scripts)."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, snapshot_frame_numpy_copy

__all__ = [
    "initial_trajectory",
    "trajectory_checkpoint_callback",
    "write_tps_checkpoint_npz",
]

_npz_all_nonfinite_error_count = 0


def _validate_coords_array_for_npz(
    arr: np.ndarray,
    *,
    label: str,
    out_npz: Path,
    mc_step: int | None = None,
) -> bool:
    """Return True if ``arr`` should be written; else log and return False."""
    global _npz_all_nonfinite_error_count
    if not np.any(np.isfinite(arr)):
        _npz_all_nonfinite_error_count += 1
        n = _npz_all_nonfinite_error_count
        if n <= 3 or (mc_step is not None and mc_step % 500 == 0):
            print(
                f"[TPS] ERROR: {label} has no finite coordinates; refusing to write {out_npz}. "
                "Dynamics may have diverged (NaN), or GPU readback failed."
                + (f" (failure #{n})" if n > 1 else ""),
                file=sys.stderr,
                flush=True,
            )
        return False
    n_fin = int(np.isfinite(arr).sum())
    n_tot = int(arr.size)
    if n_fin < n_tot:
        print(
            f"[TPS] Warning: {label} is only {100.0 * n_fin / n_tot:.1f}% finite "
            f"({n_fin}/{n_tot} values).",
            file=sys.stderr,
            flush=True,
        )
    return True


def write_tps_checkpoint_npz(traj: Trajectory, out_npz: Path, *, mc_step: int) -> bool:
    """Stack accepted path coordinates to NPZ; return True if written.

    Refuses to write if stacked coordinates contain no finite values.
    """
    stack = []
    for snap in traj:
        if not isinstance(snap, BoltzSnapshot) or snap.tensor_coords is None:
            continue
        stack.append(snapshot_frame_numpy_copy(snap))
    if not stack:
        return False
    arr = np.stack(stack, axis=0)
    if not _validate_coords_array_for_npz(
        arr,
        label=f"trajectory checkpoint (mc_step={mc_step})",
        out_npz=out_npz,
        mc_step=mc_step,
    ):
        return False
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        coords=arr,
        frame_indices=np.arange(arr.shape[0], dtype=np.int32),
        mc_step=np.int64(mc_step),
    )
    return True


def trajectory_checkpoint_callback(
    work_root: Path,
    *,
    bracket_tag: str = "[TPS]",
    verbose_log: bool = False,
) -> Callable[[int, Trajectory], None]:
    """Return a callback ``(mc_step, traj)`` that writes ``trajectory_checkpoints/*.npz``."""

    def cb(mc_step: int, traj: Trajectory) -> None:
        ck = work_root / "trajectory_checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        tagged = ck / f"tps_mc_step_{mc_step:08d}.npz"
        wrote_ckpt = write_tps_checkpoint_npz(traj, tagged, mc_step=mc_step)
        latest = ck / "latest.npz"
        if wrote_ckpt:
            shutil.copyfile(tagged, latest)
        last_only = ck / "last_frame_only.npz"
        snap = traj[-1]
        if wrote_ckpt and isinstance(snap, BoltzSnapshot) and snap.tensor_coords is not None:
            lf = snapshot_frame_numpy_copy(snap)
            if np.any(np.isfinite(lf)):
                np.savez_compressed(
                    last_only,
                    coords=lf.reshape(1, *lf.shape),
                    mc_step=np.int64(mc_step),
                )
        if wrote_ckpt:
            if verbose_log:
                print(
                    f"{bracket_tag} trajectory checkpoint MC step {mc_step} -> {tagged.name} "
                    "(latest.npz updated)",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"{bracket_tag} checkpoint MC step {mc_step} -> {tagged.name}",
                    file=sys.stderr,
                    flush=True,
                )

    return cb


def initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    """One full forward diffusion path (noise to coordinates)."""
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

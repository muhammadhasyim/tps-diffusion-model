#!/usr/bin/env python3
"""Orchestrate TPS-style sampling with Boltz-2 (GPU) and OpenPathSampling.

Requires a Lightning checkpoint and a conditioning bundle (pickle) containing
``s_trunk``, ``s_inputs``, ``feats``, ``diffusion_conditioning`` tensors (see
``genai_tps.backends.boltz.utils.load_conditioning_bundle``). Produce such a bundle from a
Boltz inference run or a regression-style tensor dump.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import openpathsampling as paths
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.path_probability import compute_log_path_prob
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
from genai_tps.backends.boltz.states import state_volume_high_sigma, state_volume_quality
from genai_tps.backends.boltz.collective_variables import make_plddt_proxy_cv, make_sigma_cv
from genai_tps.backends.boltz.utils import (
    build_network_condition_kwargs,
    load_boltz2_module,
    load_conditioning_bundle,
)


def initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    """Run one full forward diffusion path (noise to structure)."""
    n = n_steps if n_steps is not None else core.num_sampling_steps
    x = core.sample_initial_noise()
    snaps: list[BoltzSnapshot] = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(
        snapshot_from_gpu(x, 0, None, None, None, sig0, center_mean_before_step=None)
    )
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


def run_tps_rounds(
    engine: BoltzDiffusionEngine,
    core: BoltzSamplerCore,
    init_traj: Trajectory,
    n_rounds: int,
    log_path: Path,
) -> None:
    """Simple shooting: resample from random interior frames (demonstration)."""
    L = len(init_traj) - 1
    log_path.write_text("")
    for r in range(n_rounds):
        k = int(np.random.randint(1, max(2, L)))
        x0 = init_traj[k]._tensor_coords_gpu
        if x0 is None:
            continue
        _, eps_new, _, _, meta_new = core.generate_segment(x0, k, L)
        lp = compute_log_path_prob(
            eps_new,
            meta_new,
            initial_coords=None,
            sigma0=None,
            include_jacobian=True,
            n_atoms=x0.shape[1],
        )
        with log_path.open("a") as f:
            f.write(f"round {r} shoot {k} log_path_prob {float(lp):.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TPS orchestration for Boltz-2 diffusion")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Boltz2 .ckpt path")
    parser.add_argument("--conditioning", type=Path, required=True, help="Pickle with diffusion conditioning tensors")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("genai_tps_boltz_run.log"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_conditioning_bundle(args.conditioning)
    model = load_boltz2_module(args.checkpoint, device)
    model.to(device)
    model.eval()

    diffusion = model.structure_module
    kwargs = build_network_condition_kwargs(bundle)
    for k, v in kwargs.items():
        if hasattr(v, "to"):
            kwargs[k] = v.to(device)
    kwargs["feats"] = {fk: fv.to(device) if hasattr(fv, "to") else fv for fk, fv in kwargs["feats"].items()}
    atom_mask = kwargs["feats"]["atom_pad_mask"].float()

    core = BoltzSamplerCore(diffusion, atom_mask, kwargs, multiplicity=1)
    core.build_schedule()
    n_atoms = atom_mask.shape[1]
    descriptor = boltz_snapshot_descriptor(n_atoms=n_atoms)
    engine = BoltzDiffusionEngine(
        core,
        descriptor,
        options={"n_frames_max": core.num_sampling_steps + 2},
    )

    sigma_cv = make_sigma_cv()
    state_a = state_volume_high_sigma(sigma_min=float(core.schedule[0].sigma_tm) * 0.5)
    state_b = state_volume_quality(make_plddt_proxy_cv(), plddt_min=30.0)
    _ = paths.FixedLengthTPSNetwork(
        initial_states=[state_a],
        final_states=[state_b],
        length=core.num_sampling_steps + 1,
    )

    init_traj = initial_trajectory(core)
    run_tps_rounds(engine, core, init_traj, args.rounds, args.out)

    meta = {
        "n_frames": len(init_traj),
        "sigma0": float(core.schedule[0].sigma_tm),
    }
    with Path(args.out.with_suffix(".pkl")).open("wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    main()

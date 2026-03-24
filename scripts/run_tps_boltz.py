#!/usr/bin/env python3
"""Orchestrate TPS sampling with Boltz-2 (GPU) and OpenPathSampling.

Requires a Lightning checkpoint and a conditioning bundle (pickle) containing
``s_trunk``, ``s_inputs``, ``feats``, ``diffusion_conditioning`` tensors (see
``genai_tps.backends.boltz.utils.load_conditioning_bundle``). Produce such a bundle from a
Boltz inference run or a regression-style tensor dump.

Runs **fixed-length** one-way shooting with Metropolis acceptance via OPS
:class:`PathSampling` (see ``genai_tps.backends.boltz.tps_sampling``).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import openpathsampling as paths
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.backends.boltz.utils import (
    build_network_condition_kwargs,
    load_boltz2_module,
    load_conditioning_bundle,
)


def initial_trajectory(core: BoltzSamplerCore, n_steps: int | None = None) -> Trajectory:
    """Run one full forward diffusion path (noise to structure)."""
    x = core.sample_initial_noise()
    n = n_steps if n_steps is not None else core.num_sampling_steps
    snaps: list = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="TPS orchestration for Boltz-2 diffusion")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Boltz2 .ckpt path")
    parser.add_argument(
        "--conditioning",
        type=Path,
        required=True,
        help="Pickle with diffusion conditioning tensors",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help="Print TPS progress every N MC steps to stderr (0 = quiet).",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        default=False,
        help=(
            "Use only forward shooting moves (no backward re-noising). "
            "Forward shooting always accepts reactive paths (bias = 1), making this "
            "a guaranteed-correct baseline for validating TPS ensemble wiring."
        ),
    )
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
        options={"n_frames_max": core.num_sampling_steps + 4},
    )

    init_traj = initial_trajectory(core)
    log_path = args.out
    final_traj, tps_step_log = run_tps_path_sampling(
        engine,
        init_traj,
        args.rounds,
        log_path,
        progress_every=args.progress_every,
        forward_only=args.forward_only,
    )

    meta = {
        "n_frames": len(final_traj),
        "sigma0": float(core.schedule[0].sigma_tm),
        "tps_steps": tps_step_log,
        "shooting_log": str(log_path.resolve()),
    }
    with Path(args.out.with_suffix(".pkl")).open("wb") as f:
        pickle.dump(meta, f)
    summary_json = args.out.with_suffix(".json")
    summary_json.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()

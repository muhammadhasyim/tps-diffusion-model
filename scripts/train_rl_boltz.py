#!/usr/bin/env python3
"""Offline RLDiff-style RL fine-tuning for Boltz-2 co-folding diffusion.

Rolls out trajectories with :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore`,
scores terminal frames with GPU-native PoseBusters-style checks
(:mod:`genai_tps.analysis.posebusters_gpu`, **no** ``posebusters`` package), and applies a
clipped surrogate with denoiser-velocity importance weights
(:mod:`genai_tps.rl`).

Example::

    pip install -e ".[boltz,dev]" && pip install -e ./boltz
    python scripts/train_rl_boltz.py \\
        --out ./rl_boltz_out \\
        --yaml examples/cofolding_multimer_msa_empty.yaml \\
        --epochs 2 \\
        --rollouts-per-epoch 1 \\
        --diffusion-steps 8

Requires a Boltz-2 checkpoint under ``--cache`` (same as ``run_cofolding_tps_demo.py``).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def _build_boltz_session(
    *,
    yaml_path: Path,
    cache: Path,
    boltz_prep_dir: Path,
    device: torch.device,
    diffusion_steps: int,
    recycling_steps: int,
    kernels: bool,
):
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        check_inputs,
        download_boltz2,
        process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2

    from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore

    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    boltz_run_dir = boltz_prep_dir / f"boltz_prep_{yaml_path.stem}"
    boltz_run_dir.mkdir(parents=True, exist_ok=True)

    data_list = check_inputs(yaml_path)
    process_inputs(
        data=data_list,
        out_dir=boltz_run_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=False,
        boltz2=True,
        preprocessing_threads=1,
    )

    manifest = Manifest.load(boltz_run_dir / "processed" / "manifest.json")
    if not manifest.records:
        raise RuntimeError("No records in manifest after preprocessing.")

    processed_dir = boltz_run_dir / "processed"
    dm = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed_dir / "constraints"
        if (processed_dir / "constraints").exists()
        else None,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False

    ckpt = cache / "boltz2_conf.ckpt"
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": diffusion_steps,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model = Boltz2.load_from_checkpoint(
        str(ckpt),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering),
    )
    model.to(device)

    atom_mask, network_kwargs = boltz2_trunk_to_network_kwargs(
        model, batch, recycling_steps=recycling_steps
    )
    for k, v in list(network_kwargs.items()):
        if hasattr(v, "to"):
            network_kwargs[k] = v.to(device)
    if isinstance(network_kwargs.get("feats"), dict):
        network_kwargs["feats"] = {
            fk: fv.to(device) if hasattr(fv, "to") else fv
            for fk, fv in network_kwargs["feats"].items()
        }

    diffusion = model.structure_module
    core = BoltzSamplerCore(diffusion, atom_mask, network_kwargs, multiplicity=1)
    core.build_schedule(diffusion_steps)

    struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
    topo_npz = struct_candidates[0] if struct_candidates else None

    return model, core, batch, processed_dir, topo_npz, boltz_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RLDiff-style offline RL for Boltz-2 diffusion.")
    parser.add_argument("--yaml", type=Path, default=None, help="Boltz input YAML (co-folding).")
    parser.add_argument("--cache", type=Path, default=None, help="Boltz cache dir (default ~/.boltz).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rollouts-per-epoch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--tau-sq", type=float, default=1e-2, help="Velocity surrogate variance.")
    parser.add_argument("--topo-npz", type=Path, default=None, help="structures/*.npz for GPU PoseBusters topology.")
    parser.add_argument("--train-diffusion-only", action="store_true", default=True)
    parser.add_argument("--train-full-model", action="store_true", default=False)
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "examples" / "cofolding_multimer_msa_empty.yaml")
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from genai_tps.analysis.boltz_npz_export import load_topo
        from genai_tps.backends.boltz.snapshot import BoltzSnapshot, snapshot_frame_numpy_copy
        from genai_tps.rl.config import BoltzRLConfig
        from genai_tps.rl.reward_gpu import gpu_pass_fraction_reward_from_coords
        from genai_tps.rl.rollout import rollout_forward_trajectory
        from genai_tps.rl.training import replay_trajectory_loss
        from genai_tps.analysis.posebusters_gpu import GPUPoseBustersEvaluator
    except ImportError as e:
        print(f"Import error (install boltz + genai-tps with [boltz,dev]): {e}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    model, core, _batch, processed_dir, topo_auto, _prep_dir = _build_boltz_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=work_root,
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=args.kernels,
    )

    topo_npz = Path(args.topo_npz).expanduser().resolve() if args.topo_npz else topo_auto
    if topo_npz is None or not topo_npz.is_file():
        print("Topology NPZ required for GPU PoseBusters reward; pass --topo-npz.", file=sys.stderr)
        sys.exit(1)

    structure, n_struct = load_topo(topo_npz)

    cfg = BoltzRLConfig(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        velocity_log_prob_tau_sq=args.tau_sq,
    )

    train_full = bool(args.train_full_model)
    params = list(model.parameters()) if train_full else list(model.structure_module.parameters())
    optimizer = Adam(params, lr=cfg.learning_rate)

    # Reference coordinates for PoseBusters GPU: initial rollout terminal frame.
    model.eval()
    with torch.inference_mode():
        probe_traj = rollout_forward_trajectory(core, num_steps=args.diffusion_steps)
    final_coords = probe_traj[-1].x_next
    probe_snap = BoltzSnapshot.from_gpu_batch(
        final_coords.detach().cpu(),
        step_index=args.diffusion_steps,
        sigma=None,
        defer_numpy_coords=True,
    )
    probe_np = snapshot_frame_numpy_copy(probe_snap)[: int(n_struct)].astype(np.float32)

    gpu_ev = GPUPoseBustersEvaluator(
        structure,
        int(n_struct),
        probe_np,
        backend_mode="gpu_fast",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if not train_full:
            model.structure_module.train()
        epoch_loss = 0.0
        n_batches = 0
        for _r in range(args.rollouts_per_epoch):
            with torch.inference_mode():
                core.diffusion.eval()
                tr = rollout_forward_trajectory(core, num_steps=args.diffusion_steps)
            raw_r = gpu_pass_fraction_reward_from_coords(gpu_ev, tr[-1].x_next, step_index=args.diffusion_steps)
            from genai_tps.rl.ppo_surrogate import normalize_rewards_per_trajectory

            z_list, _, _ = normalize_rewards_per_trajectory([raw_r])
            reward_t = torch.tensor(z_list[0], device=device, dtype=torch.float32)

            model.train()
            if not train_full:
                model.structure_module.train()
            core.diffusion.train()

            optimizer.zero_grad()
            loss = replay_trajectory_loss(core, tr, reward_t, cfg=cfg)
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        print(f"[RL] epoch {epoch} mean_loss={avg:.6f}", flush=True)
        ckpt_path = work_root / f"boltz2_rl_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[RL] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()

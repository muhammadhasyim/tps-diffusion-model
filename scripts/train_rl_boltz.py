#!/usr/bin/env python3
"""Offline RLDiff-style RL fine-tuning for Boltz-2 protein-ligand diffusion.

Rolls out trajectories with :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore`,
scores terminal frames using classic FES collective variables (ligand pose RMSD,
protein-ligand coordination number, ligand-pocket COM distance), and applies a
clipped surrogate with denoiser-velocity importance weights (:mod:`genai_tps.rl`).

Example::

    python scripts/train_rl_boltz.py \
        --out ./rl_boltz_out \
        --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
        --epochs 15 --rollouts-per-epoch 4 --diffusion-steps 16
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _cv_reward_from_coords(
    coords_batched: torch.Tensor,
    indexer,
    n_struct: int,
    *,
    w_rmsd: float = -0.05,
    w_contacts: float = 0.04,
    w_pocket_dist: float = -0.1,
    contact_r0: float = 4.0,
) -> tuple[float, dict[str, float]]:
    """Compute a composite reward from classic protein-ligand FES CVs.

    R = w_rmsd * ligand_RMSD + w_contacts * coordination_number
        + w_pocket_dist * pocket_distance

    Default weights are scaled so each CV contributes roughly O(1) to the
    reward at typical initial values (RMSD ~40A, contacts ~25, pocket_dist ~10A),
    giving the z-score normalization clean variance to work with.
    """
    from genai_tps.backends.boltz.collective_variables import (
        ligand_pocket_distance,
        ligand_pose_rmsd,
        protein_ligand_contacts,
        protein_ligand_hbond_count,
    )

    c = coords_batched.detach()
    if c.dim() == 2:
        c = c.unsqueeze(0)
    c = c[:, :n_struct, :]

    class _MinimalSnap:
        def __init__(self, tc):
            self._tensor_coords_gpu = tc
    snap = _MinimalSnap(c)

    rmsd = ligand_pose_rmsd(snap, indexer)
    contacts = protein_ligand_contacts(snap, indexer, r0=contact_r0)
    pocket_dist = ligand_pocket_distance(snap, indexer)
    hbonds = protein_ligand_hbond_count(snap, indexer)

    reward = w_rmsd * rmsd + w_contacts * contacts + w_pocket_dist * pocket_dist

    cv_dict = {
        "ligand_rmsd": rmsd,
        "contacts": contacts,
        "pocket_dist": pocket_dist,
        "hbonds": hbonds,
    }
    return float(reward), cv_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="RLDiff-style offline RL for Boltz-2 diffusion (classic FES CVs).")
    parser.add_argument("--yaml", type=Path, default=None, help="Boltz input YAML (protein-ligand).")
    parser.add_argument("--cache", type=Path, default=None, help="Boltz cache dir (default ~/.boltz).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--rollouts-per-epoch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--tau-sq", type=float, default=1e-2, help="Velocity surrogate variance.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--topo-npz", type=Path, default=None, help="structures/*.npz for topology.")
    parser.add_argument("--pocket-radius", type=float, default=8.0, help="Pocket definition radius (A).")
    parser.add_argument("--contact-r0", type=float, default=4.0, help="Coordination switching function r0 (A).")
    parser.add_argument("--w-rmsd", type=float, default=-0.05, help="Reward weight for ligand RMSD.")
    parser.add_argument("--w-contacts", type=float, default=0.04, help="Reward weight for contact count.")
    parser.add_argument("--w-pocket-dist", type=float, default=-0.1, help="Reward weight for pocket distance.")
    parser.add_argument("--train-diffusion-only", action="store_true", default=True)
    parser.add_argument("--train-full-model", action="store_true", default=False)
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml")
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from genai_tps.analysis.boltz_npz_export import load_topo
        from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
        from genai_tps.rl.config import BoltzRLConfig
        from genai_tps.rl.ppo_surrogate import normalize_rewards_per_trajectory
        from genai_tps.rl.rollout import rollout_forward_trajectory
        from genai_tps.rl.training import replay_trajectory_loss
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
        print("Topology NPZ required; pass --topo-npz.", file=sys.stderr)
        sys.exit(1)

    structure, n_struct = load_topo(topo_npz)

    ref_coords_np = np.asarray(structure.atoms["coords"], dtype=np.float64)[:int(n_struct)]
    indexer = PoseCVIndexer(structure, ref_coords_np, pocket_radius=float(args.pocket_radius))
    if len(indexer.ligand_idx) == 0:
        print("ERROR: No NONPOLYMER (ligand) chain found in the Boltz structure.", file=sys.stderr)
        sys.exit(1)

    print(f"[CV setup] ligand atoms: {len(indexer.ligand_idx)}, "
          f"pocket Calpha: {len(indexer.pocket_ca_idx)}, "
          f"pocket heavy: {len(indexer.pocket_heavy_idx)}", flush=True)

    cfg = BoltzRLConfig(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        velocity_log_prob_tau_sq=args.tau_sq,
        max_grad_norm=args.max_grad_norm,
    )

    train_full = bool(args.train_full_model)
    params = list(model.parameters()) if train_full else list(model.structure_module.parameters())
    optimizer = Adam(params, lr=cfg.learning_rate)

    # Fix gradient flow through torch.utils.checkpoint in the diffusion module.
    # The single_conditioner checkpoint receives (times, s_trunk, s_inputs) —
    # none have requires_grad, so the reentrant checkpoint skips building the
    # backward graph. Setting requires_grad on s_trunk/s_inputs enables the
    # checkpoint to track gradients through single_conditioner's parameters
    # without disabling checkpointing (which would cause OOM with Boltz-2).
    nk = core.network_condition_kwargs
    _grad_fix_count = 0
    for key in ("s_trunk", "s_inputs"):
        if key in nk and isinstance(nk[key], torch.Tensor) and not nk[key].requires_grad:
            nk[key] = nk[key].detach().clone().requires_grad_(True)
            _grad_fix_count += 1
    if _grad_fix_count:
        print(f"[grad fix] Set requires_grad=True on {_grad_fix_count} conditioning tensors "
              f"for checkpoint gradient flow", flush=True)

    # --- CSV log setup ---
    log_path = work_root / "training_log.csv"
    log_fields = [
        "epoch", "rollout", "ligand_rmsd", "contacts", "pocket_dist",
        "hbonds", "raw_reward", "z_reward", "loss",
    ]
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    writer.writeheader()

    # --- Summary tracking ---
    epoch_summaries: list[dict] = []
    best_avg_reward = float("-inf")
    best_epoch = 0

    print(f"\n{'='*72}")
    print(f"  RL fine-tuning: {args.epochs} epochs x {args.rollouts_per_epoch} rollouts")
    print(f"  LR={args.learning_rate:.1e}  clip={args.clip_range}  "
          f"grad_norm={args.max_grad_norm}  diffusion_steps={args.diffusion_steps}")
    print(f"  Reward weights: rmsd={args.w_rmsd}  contacts={args.w_contacts}  "
          f"pocket_dist={args.w_pocket_dist}")
    print(f"{'='*72}\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        # --- Phase 1: Collect rollouts ---
        rollout_data: list[tuple[list, float, dict]] = []  # (trajectory, raw_reward, cv_dict)

        model.eval()
        core.diffusion.eval()

        for _r in range(args.rollouts_per_epoch):
            with torch.inference_mode():
                tr = rollout_forward_trajectory(core, num_steps=args.diffusion_steps)

            raw_r, cv_vals = _cv_reward_from_coords(
                tr[-1].x_next,
                indexer,
                int(n_struct),
                w_rmsd=args.w_rmsd,
                w_contacts=args.w_contacts,
                w_pocket_dist=args.w_pocket_dist,
                contact_r0=args.contact_r0,
            )
            rollout_data.append((tr, raw_r, cv_vals))

        # --- Phase 2: Normalize rewards across all rollouts in this epoch ---
        raw_rewards = [r for _, r, _ in rollout_data]
        z_rewards, r_mean, r_std = normalize_rewards_per_trajectory(raw_rewards)

        # --- Phase 3: Train on each rollout with normalized reward ---
        model.train()
        if not train_full:
            model.structure_module.train()
        core.diffusion.train()

        epoch_loss = 0.0
        epoch_cvs = {"ligand_rmsd": [], "contacts": [], "pocket_dist": [], "hbonds": []}

        for rollout_idx, ((tr, raw_r, cv_vals), z_r) in enumerate(zip(rollout_data, z_rewards)):
            reward_t = torch.tensor([z_r], device=device, dtype=torch.float32)

            optimizer.zero_grad()
            loss = replay_trajectory_loss(core, tr, reward_t, cfg=cfg)
            loss.backward()
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            epoch_loss += loss_val

            for k in epoch_cvs:
                epoch_cvs[k].append(cv_vals[k])

            writer.writerow({
                "epoch": epoch,
                "rollout": rollout_idx + 1,
                "ligand_rmsd": f"{cv_vals['ligand_rmsd']:.3f}",
                "contacts": f"{cv_vals['contacts']:.2f}",
                "pocket_dist": f"{cv_vals['pocket_dist']:.3f}",
                "hbonds": f"{cv_vals['hbonds']:.0f}",
                "raw_reward": f"{raw_r:.4f}",
                "z_reward": f"{z_r:.4f}",
                "loss": f"{loss_val:.4f}",
            })

            print(f"  e{epoch:02d} r{rollout_idx+1}: "
                  f"rmsd={cv_vals['ligand_rmsd']:6.2f}A  "
                  f"contacts={cv_vals['contacts']:5.1f}  "
                  f"pkt_d={cv_vals['pocket_dist']:6.2f}A  "
                  f"hbonds={cv_vals['hbonds']:3.0f}  "
                  f"R={raw_r:+.3f} z={z_r:+.3f} L={loss_val:.3f}", flush=True)

        log_f.flush()

        avg_loss = epoch_loss / args.rollouts_per_epoch
        avg_reward = np.mean(raw_rewards)
        avg_rmsd = np.mean(epoch_cvs["ligand_rmsd"])
        avg_contacts = np.mean(epoch_cvs["contacts"])
        avg_pocket = np.mean(epoch_cvs["pocket_dist"])
        avg_hbonds = np.mean(epoch_cvs["hbonds"])
        best_rmsd = min(epoch_cvs["ligand_rmsd"])
        best_contacts = max(epoch_cvs["contacts"])

        summary = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_reward": float(avg_reward),
            "avg_rmsd": float(avg_rmsd),
            "avg_contacts": float(avg_contacts),
            "avg_pocket_dist": float(avg_pocket),
            "avg_hbonds": float(avg_hbonds),
            "best_rmsd": float(best_rmsd),
            "best_contacts": float(best_contacts),
        }
        epoch_summaries.append(summary)

        print(f"[Epoch {epoch:02d}] avg_loss={avg_loss:.3f}  "
              f"avg_R={avg_reward:+.3f}  "
              f"avg_rmsd={avg_rmsd:.2f}A (best {best_rmsd:.2f}A)  "
              f"avg_contacts={avg_contacts:.1f} (best {best_contacts:.1f})  "
              f"avg_pocket_dist={avg_pocket:.2f}A  "
              f"avg_hbonds={avg_hbonds:.1f}", flush=True)

        # Save checkpoint every epoch
        ckpt_path = work_root / f"boltz2_rl_epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Track best model by average reward
        if float(avg_reward) > best_avg_reward:
            best_avg_reward = float(avg_reward)
            best_epoch = epoch
            best_path = work_root / "boltz2_rl_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best model (avg_reward={best_avg_reward:+.4f})", flush=True)

        print("", flush=True)

    log_f.close()

    # --- Final summary ---
    summary_path = work_root / "training_summary.json"
    final = {
        "config": {
            "epochs": args.epochs,
            "rollouts_per_epoch": args.rollouts_per_epoch,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "max_grad_norm": args.max_grad_norm,
            "diffusion_steps": args.diffusion_steps,
            "pocket_radius": args.pocket_radius,
            "contact_r0": args.contact_r0,
            "w_rmsd": args.w_rmsd,
            "w_contacts": args.w_contacts,
            "w_pocket_dist": args.w_pocket_dist,
        },
        "best_epoch": best_epoch,
        "best_avg_reward": best_avg_reward,
        "epoch_summaries": epoch_summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"{'='*72}")
    print(f"  Training complete.")
    print(f"  Best model: epoch {best_epoch} (avg_reward={best_avg_reward:+.4f})")
    print(f"  Log: {log_path}")
    print(f"  Summary: {summary_path}")

    # Print trend table
    print(f"\n  {'Epoch':>5} {'AvgRMSD':>8} {'BestRMSD':>9} {'AvgContacts':>12} {'AvgPktDist':>11} {'AvgReward':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*9} {'-'*12} {'-'*11} {'-'*10}")
    for s in epoch_summaries:
        print(f"  {s['epoch']:5d} {s['avg_rmsd']:8.2f} {s['best_rmsd']:9.2f} "
              f"{s['avg_contacts']:12.1f} {s['avg_pocket_dist']:11.2f} {s['avg_reward']:+10.4f}")
    print(f"{'='*72}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Weighted denoising score matching fine-tuning for Boltz 2.

Trains the diffusion head on pre-collected reweighted structures using an
offline supervised loss — no online rollouts or RL surrogates required.

Example::

    python scripts/train_weighted_dsm.py \
        --out ./wdsm_out \
        --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
        --data structures.npz \
        --epochs 10 --batch-size 4

The data NPZ must contain:
  - coords: (N, M, 3) atom coordinates
  - logw: (N,) log importance weights
  - atom_mask: (N, M) optional binary atom mask
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

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
        template_dir=processed_dir / "templates"
        if (processed_dir / "templates").exists()
        else None,
        extra_mols_dir=processed_dir / "mols"
        if (processed_dir / "mols").exists()
        else None,
    )
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
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

    return model, core, batch, processed_dir, boltz_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weighted DSM fine-tuning for Boltz 2."
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="NPZ with coords, logw, atom_mask.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.01, help="Regularization strength.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight tempering exponent.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--save-every-batches", type=int, default=0,
                        help="Save mid-epoch checkpoint every N batches (0=disabled).")
    parser.add_argument("--resume-from-batch", type=int, default=0,
                        help="Skip this many batches when resuming mid-epoch.")
    parser.add_argument("--val-data", type=Path, default=None, help="Optional val NPZ for per-epoch val loss.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a .pt checkpoint (model state_dict).")
    parser.add_argument("--start-epoch", type=int, default=1, help="Starting epoch number (for resumed runs).")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"],
                        help="LR schedule: 'constant' or 'cosine' annealing (default: cosine).")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5, help="Linear warmup epochs before cosine decay.")
    parser.add_argument("--lr-min", type=float, default=1e-7, help="Minimum LR for cosine annealing.")
    parser.add_argument("--early-stopping-patience", type=int, default=50,
                        help="Stop if val loss doesn't improve for N epochs (0=disabled).")
    parser.add_argument("--loss-type", type=str, default="cartesian", choices=["cartesian", "quotient"],
                        help="Loss: 'cartesian' (original DSM) or 'quotient' (SE(3)-invariant).")
    args = parser.parse_args()

    from genai_tps.weighted_dsm.config import WeightedDSMConfig
    from genai_tps.weighted_dsm.dataset import ReweightedStructureDataset
    from genai_tps.weighted_dsm.diagnostics import temper_log_weights, weight_statistics
    from genai_tps.weighted_dsm.loss import regularized_weighted_dsm_loss, quotient_weighted_dsm_loss
    from genai_tps.weighted_dsm.noise_schedule import EDMNoiseParams

    yaml_path = args.yaml or (
        _REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml"
    )
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        print(f"Data NPZ not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    work_root = args.out.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    cfg = WeightedDSMConfig(
        learning_rate=args.learning_rate,
        beta=args.beta,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )

    # Load dataset with optional weight tempering
    ds = ReweightedStructureDataset.from_npz(data_path)
    if cfg.gamma < 1.0:
        ds.logw = temper_log_weights(ds.logw, cfg.gamma)
        print(f"[WDSM] Applied weight tempering gamma={cfg.gamma}", flush=True)

    stats = weight_statistics(ds.logw)
    print(f"[WDSM] Weight stats: N_eff={stats['n_eff']:.1f} "
          f"({stats['n_eff_fraction']*100:.1f}%), "
          f"max_weight_frac={stats['max_weight_fraction']:.4f}", flush=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Optional validation set
    val_loader = None
    if args.val_data is not None:
        val_path = Path(args.val_data).expanduser().resolve()
        if val_path.is_file():
            val_ds = ReweightedStructureDataset.from_npz(val_path)
            if cfg.gamma < 1.0:
                val_ds.logw = temper_log_weights(val_ds.logw, cfg.gamma)
            val_stats = weight_statistics(val_ds.logw)
            print(f"[WDSM] Val set: {len(val_ds)} samples, N_eff={val_stats['n_eff']:.1f}", flush=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Build Boltz session
    model, core, _batch, processed_dir, _prep_dir = _build_boltz_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=work_root,
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=args.kernels,
    )

    # Extract network conditioning kwargs for score network calls.
    # Filter to only the keys accepted by DiffusionModule.forward().
    _valid_nck_keys = {"s_inputs", "s_trunk", "feats", "multiplicity", "diffusion_conditioning"}
    nck = {k: v for k, v in core.network_condition_kwargs.items() if k in _valid_nck_keys}

    # Gradient fix: enable requires_grad on conditioning tensors so
    # torch.utils.checkpoint builds the backward graph in the score model.
    for key in ("s_trunk", "s_inputs"):
        if key in nck and isinstance(nck[key], torch.Tensor) and not nck[key].requires_grad:
            nck[key] = nck[key].detach().clone().requires_grad_(True)
    print("[grad fix] Set requires_grad on conditioning tensors", flush=True)

    n_disabled = 0
    for name, module in model.structure_module.named_modules():
        if hasattr(module, 'activation_checkpointing') and module.activation_checkpointing:
            module.activation_checkpointing = False
            n_disabled += 1
    print(f"[grad fix] Disabled activation_checkpointing on {n_disabled} submodules", flush=True)

    # Frozen reference model
    frozen_diffusion = copy.deepcopy(model.structure_module)
    frozen_diffusion.eval()
    frozen_diffusion.requires_grad_(False)

    model.structure_module.requires_grad_(True)
    n_grad = sum(1 for p in model.structure_module.parameters() if p.requires_grad)
    print(f"[grad fix] Enabled requires_grad on {n_grad} parameters", flush=True)

    noise_params = EDMNoiseParams.from_diffusion(model.structure_module)
    print(f"[WDSM] Noise params: sigma_data={noise_params.sigma_data}, "
          f"P_mean={noise_params.P_mean}, P_std={noise_params.P_std}", flush=True)

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from).expanduser().resolve()
        print(f"[WDSM] Resuming from {ckpt_path}", flush=True)
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state)

    params = list(model.structure_module.parameters())
    optimizer = Adam(params, lr=cfg.learning_rate)

    # LR scheduler
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.lr_warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - args.lr_warmup_epochs, eta_min=args.lr_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.lr_warmup_epochs])
        print(f"[LR] Cosine annealing: warmup={args.lr_warmup_epochs} epochs, "
              f"eta_min={args.lr_min:.1e}", flush=True)
    else:
        scheduler = None

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = args.early_stopping_patience
    if early_stop_patience > 0:
        print(f"[early stop] Patience={early_stop_patience} epochs", flush=True)

    # CSV log
    log_path = work_root / "training_log.csv"
    log_fields = ["epoch", "batch", "loss", "grad_norm", "val_loss"]
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    writer.writeheader()

    model_n_atoms = int(core.atom_mask.shape[1])
    data_n_atoms = ds.coords.shape[1]
    if data_n_atoms != model_n_atoms:
        print(f"ERROR: Dataset has {data_n_atoms} atoms but model expects {model_n_atoms}. "
              f"Ensure the dataset includes padding atoms.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Weighted DSM fine-tuning: {cfg.epochs} epochs, "
          f"batch_size={args.batch_size}")
    print(f"  LR={cfg.learning_rate:.1e}  beta={cfg.beta}  "
          f"gamma={cfg.gamma}  grad_norm={cfg.max_grad_norm}")
    print(f"  Dataset: {len(ds)} samples, N_eff={stats['n_eff']:.1f}")
    print(f"{'='*60}\n", flush=True)

    epoch_losses: list[float] = []
    start_ep = max(1, int(args.start_epoch))

    for epoch in range(start_ep, cfg.epochs + 1):
        model.structure_module.train()
        epoch_loss = 0.0
        n_batches = 0

        skip_batches = args.resume_from_batch if epoch == start_ep else 0
        for batch in loader:
            n_batches += 1
            if n_batches <= skip_batches:
                continue

            x0 = batch["coords"].float().to(device)
            lw = batch["logw"].float().to(device)
            am = batch["atom_mask"].float().to(device)

            optimizer.zero_grad()
            batch_nck = {**nck, "multiplicity": x0.shape[0]}
            if args.loss_type == "quotient":
                loss = quotient_weighted_dsm_loss(
                    model.structure_module, x0, lw, am, noise_params,
                    frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                    beta=cfg.beta, network_condition_kwargs=batch_nck,
                )
            else:
                loss = regularized_weighted_dsm_loss(
                    model.structure_module, frozen_diffusion,
                    x0, lw, am, noise_params, cfg,
                    network_condition_kwargs=batch_nck,
                )
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            epoch_loss += loss_val

            writer.writerow({
                "epoch": epoch,
                "batch": n_batches,
                "loss": f"{loss_val:.6f}",
                "grad_norm": f"{float(grad_norm):.4f}",
                "val_loss": "",
            })

            if args.save_every_batches > 0 and n_batches % args.save_every_batches == 0:
                mid_ckpt = work_root / f"boltz2_wdsm_e{epoch:03d}_b{n_batches:06d}.pt"
                torch.save(model.state_dict(), mid_ckpt)
                log_f.flush()
                print(f"  [batch {n_batches}] loss={loss_val:.4f} grad={float(grad_norm):.4f} >> {mid_ckpt.name}", flush=True)

        log_f.flush()
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        val_loss_str = ""
        if val_loader is not None:
            model.structure_module.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx0 = vbatch["coords"].float().to(device)
                    vlw = vbatch["logw"].float().to(device)
                    vam = vbatch["atom_mask"].float().to(device)
                    vbatch_nck = {**nck, "multiplicity": vx0.shape[0]}
                    if args.loss_type == "quotient":
                        vloss = quotient_weighted_dsm_loss(
                            model.structure_module, vx0, vlw, vam, noise_params,
                            frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                            beta=cfg.beta, network_condition_kwargs=vbatch_nck,
                        )
                    else:
                        vloss = regularized_weighted_dsm_loss(
                            model.structure_module, frozen_diffusion,
                            vx0, vlw, vam, noise_params, cfg,
                            network_condition_kwargs=vbatch_nck,
                        )
                    val_total += float(vloss.detach().cpu())
                    val_n += 1
            avg_val = val_total / max(val_n, 1)
            val_loss_str = f"  val_loss={avg_val:.6f}"
            writer.writerow({"epoch": epoch, "batch": "val", "loss": f"{avg_val:.6f}", "grad_norm": "", "val_loss": f"{avg_val:.6f}"})
            log_f.flush()

        # LR scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        lr_str = f"  lr={new_lr:.2e}" if scheduler else ""
        print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.6f} "
              f"({n_batches} batches){val_loss_str}{lr_str}", flush=True)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt = work_root / f"boltz2_wdsm_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  >> Saved {ckpt.name}", flush=True)

        # Early stopping + best model tracking
        if val_loader is not None and early_stop_patience > 0:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_ckpt = work_root / "boltz2_wdsm_best.pt"
                torch.save(model.state_dict(), best_ckpt)
                print(f"  >> New best val_loss={avg_val:.6f}, saved {best_ckpt.name}", flush=True)
            else:
                patience_counter += 1
                print(f"  >> No val improvement ({patience_counter}/{early_stop_patience})", flush=True)
                if patience_counter >= early_stop_patience:
                    print(f"\n[early stop] Val loss hasn't improved for {early_stop_patience} epochs. Stopping.", flush=True)
                    break

    log_f.close()

    # Final checkpoint
    final_ckpt = work_root / "boltz2_wdsm_final.pt"
    torch.save(model.state_dict(), final_ckpt)

    stopped_early = patience_counter >= early_stop_patience if early_stop_patience > 0 else False
    summary = {
        "config": {
            "epochs": cfg.epochs,
            "batch_size": args.batch_size,
            "learning_rate": cfg.learning_rate,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
            "max_grad_norm": cfg.max_grad_norm,
            "data_path": str(data_path),
            "n_samples": len(ds),
            "n_eff": stats["n_eff"],
            "lr_schedule": args.lr_schedule,
            "early_stopping_patience": early_stop_patience,
        },
        "epoch_losses": epoch_losses,
        "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
        "stopped_early": stopped_early,
        "final_epoch": epoch,
    }
    with open(work_root / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete. Final checkpoint: {final_ckpt.name}")
    print(f"  Epoch losses: {' -> '.join(f'{l:.4f}' for l in epoch_losses)}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

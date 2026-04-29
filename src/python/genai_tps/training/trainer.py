"""Weighted DSM fine-tuning loop (Boltz-2 diffusion head)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from genai_tps.backends.boltz.inference import BoltzInferenceBundle
from genai_tps.training.config import WeightedDSMConfig
from genai_tps.training.dataset import ReweightedStructureDataset
from genai_tps.training.loss import (
    alignment_weighted_dsm_loss,
    quotient_weighted_dsm_loss,
    regularized_weighted_dsm_loss,
    true_quotient_dsm_loss,
)
from genai_tps.training.noise_schedule import EDMNoiseParams


def run_weighted_dsm_training(
    args: argparse.Namespace,
    *,
    cfg: WeightedDSMConfig,
    ds: ReweightedStructureDataset,
    val_loader: DataLoader | None,
    stats: dict[str, Any],
    bundle: BoltzInferenceBundle,
    device: torch.device,
    work_root: Path,
    data_path: Path,
    loader: DataLoader,
) -> None:
    """Execute epochs after Boltz session and datasets are prepared."""
    model = bundle.model
    core = bundle.core

    _valid_nck_keys = {"s_inputs", "s_trunk", "feats", "multiplicity", "diffusion_conditioning"}
    nck = {k: v for k, v in core.network_condition_kwargs.items() if k in _valid_nck_keys}

    for key in ("s_trunk", "s_inputs"):
        if key in nck and isinstance(nck[key], torch.Tensor) and not nck[key].requires_grad:
            nck[key] = nck[key].detach().clone().requires_grad_(True)
    print("[grad fix] Set requires_grad on conditioning tensors", flush=True)

    n_disabled = 0
    for _name, module in model.structure_module.named_modules():
        if hasattr(module, "activation_checkpointing") and module.activation_checkpointing:
            module.activation_checkpointing = False
            n_disabled += 1
    print(f"[grad fix] Disabled activation_checkpointing on {n_disabled} submodules", flush=True)

    frozen_diffusion = copy.deepcopy(model.structure_module)
    frozen_diffusion.eval()
    frozen_diffusion.requires_grad_(False)

    model.structure_module.requires_grad_(True)
    n_grad = sum(1 for p in model.structure_module.parameters() if p.requires_grad)
    print(f"[grad fix] Enabled requires_grad on {n_grad} parameters", flush=True)

    noise_params = EDMNoiseParams.from_diffusion(model.structure_module)
    print(
        f"[WDSM] Noise params: sigma_data={noise_params.sigma_data}, "
        f"P_mean={noise_params.P_mean}, P_std={noise_params.P_std}",
        flush=True,
    )

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from).expanduser().resolve()
        print(f"[WDSM] Resuming from {ckpt_path}", flush=True)
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state)

    params = list(model.structure_module.parameters())
    optimizer = Adam(params, lr=cfg.learning_rate)

    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.lr_warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.lr_warmup_epochs]
        )
        print(
            f"[LR] Cosine annealing: warmup={args.lr_warmup_epochs} epochs, "
            f"eta_min={args.lr_min:.1e}",
            flush=True,
        )
    else:
        scheduler = None

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = args.early_stopping_patience
    if early_stop_patience > 0:
        print(f"[early stop] Patience={early_stop_patience} epochs", flush=True)

    log_path = work_root / "training_log.csv"
    log_fields = ["epoch", "batch", "loss", "grad_norm", "val_loss"]
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    writer.writeheader()

    model_n_atoms = int(core.atom_mask.shape[1])
    data_n_atoms = ds.coords.shape[1]
    if data_n_atoms != model_n_atoms:
        print(
            f"ERROR: Dataset has {data_n_atoms} atoms but model expects {model_n_atoms}. "
            "Ensure the dataset includes padding atoms.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Weighted DSM fine-tuning: {cfg.epochs} epochs, batch_size={args.batch_size}")
    print(
        f"  LR={cfg.learning_rate:.1e}  beta={cfg.beta}  "
        f"grad_norm={cfg.max_grad_norm}"
    )
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
            if args.loss_type == "true-quotient":
                loss = true_quotient_dsm_loss(
                    model.structure_module,
                    x0,
                    lw,
                    am,
                    noise_params,
                    frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                    beta=cfg.beta,
                    network_condition_kwargs=batch_nck,
                )
            elif args.loss_type == "quotient":
                loss = alignment_weighted_dsm_loss(
                    model.structure_module,
                    x0,
                    lw,
                    am,
                    noise_params,
                    frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                    beta=cfg.beta,
                    network_condition_kwargs=batch_nck,
                )
            else:
                loss = regularized_weighted_dsm_loss(
                    model.structure_module,
                    frozen_diffusion,
                    x0,
                    lw,
                    am,
                    noise_params,
                    cfg,
                    network_condition_kwargs=batch_nck,
                )
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            epoch_loss += loss_val

            writer.writerow(
                {
                    "epoch": epoch,
                    "batch": n_batches,
                    "loss": f"{loss_val:.6f}",
                    "grad_norm": f"{float(grad_norm):.4f}",
                    "val_loss": "",
                }
            )

            if args.save_every_batches > 0 and n_batches % args.save_every_batches == 0:
                mid_ckpt = work_root / f"boltz2_wdsm_e{epoch:03d}_b{n_batches:06d}.pt"
                torch.save(model.state_dict(), mid_ckpt)
                log_f.flush()
                print(
                    f"  [batch {n_batches}] loss={loss_val:.4f} grad={float(grad_norm):.4f} "
                    f">> {mid_ckpt.name}",
                    flush=True,
                )

        log_f.flush()
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        val_loss_str = ""
        avg_val = float("nan")
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
                    if args.loss_type == "true-quotient":
                        vloss = true_quotient_dsm_loss(
                            model.structure_module,
                            vx0,
                            vlw,
                            vam,
                            noise_params,
                            frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                            beta=cfg.beta,
                            network_condition_kwargs=vbatch_nck,
                        )
                    elif args.loss_type == "quotient":
                        vloss = alignment_weighted_dsm_loss(
                            model.structure_module,
                            vx0,
                            vlw,
                            vam,
                            noise_params,
                            frozen_model=frozen_diffusion if cfg.beta > 0 else None,
                            beta=cfg.beta,
                            network_condition_kwargs=vbatch_nck,
                        )
                    else:
                        vloss = regularized_weighted_dsm_loss(
                            model.structure_module,
                            frozen_diffusion,
                            vx0,
                            vlw,
                            vam,
                            noise_params,
                            cfg,
                            network_condition_kwargs=vbatch_nck,
                        )
                    val_total += float(vloss.detach().cpu())
                    val_n += 1
            avg_val = val_total / max(val_n, 1)
            val_loss_str = f"  val_loss={avg_val:.6f}"
            writer.writerow(
                {
                    "epoch": epoch,
                    "batch": "val",
                    "loss": f"{avg_val:.6f}",
                    "grad_norm": "",
                    "val_loss": f"{avg_val:.6f}",
                }
            )
            log_f.flush()

        if scheduler is not None:
            scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        lr_str = f"  lr={new_lr:.2e}" if scheduler else ""
        print(
            f"[Epoch {epoch:03d}] avg_loss={avg_loss:.6f} ({n_batches} batches)"
            f"{val_loss_str}{lr_str}",
            flush=True,
        )

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt = work_root / f"boltz2_wdsm_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  >> Saved {ckpt.name}", flush=True)

        if val_loader is not None and early_stop_patience > 0:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_ckpt = work_root / "boltz2_wdsm_best.pt"
                torch.save(model.state_dict(), best_ckpt)
                print(f"  >> New best val_loss={avg_val:.6f}, saved {best_ckpt.name}", flush=True)
            else:
                patience_counter += 1
                print(
                    f"  >> No val improvement ({patience_counter}/{early_stop_patience})",
                    flush=True,
                )
                if patience_counter >= early_stop_patience:
                    print(
                        f"\n[early stop] Val loss hasn't improved for {early_stop_patience} epochs. "
                        "Stopping.",
                        flush=True,
                    )
                    break

    log_f.close()

    final_ckpt = work_root / "boltz2_wdsm_final.pt"
    torch.save(model.state_dict(), final_ckpt)

    stopped_early = patience_counter >= early_stop_patience if early_stop_patience > 0 else False
    summary = {
        "config": {
            "epochs": cfg.epochs,
            "batch_size": args.batch_size,
            "learning_rate": cfg.learning_rate,
            "beta": cfg.beta,
            "max_grad_norm": cfg.max_grad_norm,
            "loss_type": args.loss_type,
            "quotient_space_sampling": args.loss_type == "true-quotient",
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

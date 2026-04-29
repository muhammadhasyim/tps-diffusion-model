"""Multi-protein WDSM training loop for Boltz-2 diffusion head fine-tuning."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from genai_tps.training.config import WeightedDSMConfig
from genai_tps.training.loss import (
    alignment_weighted_dsm_loss,
    regularized_weighted_dsm_loss,
    true_quotient_dsm_loss,
)
from genai_tps.training.noise_schedule import EDMNoiseParams


def run_multi_protein_wdsm_training(
    *,
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: WeightedDSMConfig,
    loss_type: str,
    device: torch.device,
    work_root: Path,
    recycling_steps: int = 1,
    lr_schedule: str = "cosine",
    lr_warmup_epochs: int = 5,
    lr_min: float = 1e-7,
    early_stopping_patience: int = 20,
    resume_from: Path | None = None,
    save_every_batches: int = 0,
) -> None:
    """Fine-tune Boltz-2's ``structure_module`` across heterogeneous proteins."""
    work_root = Path(work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.structure_module.requires_grad_(True)

    n_disabled = 0
    for _name, module in model.structure_module.named_modules():
        if hasattr(module, "activation_checkpointing") and module.activation_checkpointing:
            module.activation_checkpointing = False
            n_disabled += 1
    print(f"[multi-WDSM] Disabled activation_checkpointing on {n_disabled} submodules", flush=True)

    frozen_diffusion = copy.deepcopy(model.structure_module).to(device)
    frozen_diffusion.eval()
    frozen_diffusion.requires_grad_(False)
    noise_params = EDMNoiseParams.from_diffusion(model.structure_module)

    if resume_from is not None:
        resume_from = Path(resume_from).expanduser().resolve()
        print(f"[multi-WDSM] Resuming from {resume_from}", flush=True)
        state = torch.load(str(resume_from), map_location=device)
        model.load_state_dict(state)

    params = list(model.structure_module.parameters())
    optimizer = Adam(params, lr=cfg.learning_rate)
    scheduler = _build_scheduler(
        optimizer,
        cfg=cfg,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_min=lr_min,
    )

    log_path = work_root / "training_log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_f, fieldnames=["epoch", "batch", "loss", "grad_norm", "val_loss"])
    writer.writeheader()

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_losses: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.structure_module.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            n_batches += 1
            batch = transfer_multi_protein_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = _multi_protein_batch_loss(
                model=model,
                frozen_diffusion=frozen_diffusion,
                batch=batch,
                cfg=cfg,
                loss_type=loss_type,
                noise_params=noise_params,
                recycling_steps=recycling_steps,
                training_mode=True,
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

            if save_every_batches > 0 and n_batches % save_every_batches == 0:
                ckpt = work_root / f"boltz2_wdsm_e{epoch:03d}_b{n_batches:06d}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"[multi-WDSM] saved {ckpt.name}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        avg_val = None
        if val_loader is not None:
            model.structure_module.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = transfer_multi_protein_batch_to_device(batch, device)
                    vloss = _multi_protein_batch_loss(
                        model=model,
                        frozen_diffusion=frozen_diffusion,
                        batch=batch,
                        cfg=cfg,
                        loss_type=loss_type,
                        noise_params=noise_params,
                        recycling_steps=recycling_steps,
                        training_mode=False,
                    )
                    val_total += float(vloss.detach().cpu())
                    val_n += 1
            avg_val = val_total / max(val_n, 1)
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
        val_str = "" if avg_val is None else f" val_loss={avg_val:.6f}"
        print(f"[multi-WDSM] epoch={epoch:03d} loss={avg_loss:.6f}{val_str}", flush=True)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt = work_root / f"boltz2_wdsm_epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"[multi-WDSM] saved {ckpt.name}", flush=True)

        if avg_val is not None and early_stopping_patience > 0:
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), work_root / "boltz2_wdsm_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("[multi-WDSM] early stopping", flush=True)
                    break

    log_f.close()
    final_ckpt = work_root / "boltz2_wdsm_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    summary = {
        "config": {
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "beta": cfg.beta,
            "max_grad_norm": cfg.max_grad_norm,
            "loss_type": loss_type,
            "quotient_space_sampling": loss_type == "true-quotient",
            "multi_protein": True,
            "recycling_steps": recycling_steps,
        },
        "epoch_losses": epoch_losses,
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "final_checkpoint": str(final_ckpt),
    }
    (work_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _multi_protein_batch_loss(
    *,
    model: Any,
    frozen_diffusion: Any,
    batch: dict[str, Any],
    cfg: WeightedDSMConfig,
    loss_type: str,
    noise_params: EDMNoiseParams,
    recycling_steps: int,
    training_mode: bool,
) -> torch.Tensor:
    atom_mask, nck = compute_boltz2_trunk_to_network_kwargs(
        model,
        batch,
        recycling_steps=recycling_steps,
    )
    model.structure_module.train(training_mode)
    x0 = _coords_from_boltz_batch(batch)
    logw = batch["logw"].float().to(x0.device)
    atom_mask = atom_mask.to(x0.device)
    nck = {k: v for k, v in nck.items() if k in {"s_inputs", "s_trunk", "feats", "diffusion_conditioning"}}
    nck["multiplicity"] = x0.shape[0]

    if loss_type == "true-quotient":
        return true_quotient_dsm_loss(
            model.structure_module,
            x0,
            logw,
            atom_mask,
            noise_params,
            frozen_model=frozen_diffusion if cfg.beta > 0 else None,
            beta=cfg.beta,
            network_condition_kwargs=nck,
        )
    if loss_type == "quotient":
        return alignment_weighted_dsm_loss(
            model.structure_module,
            x0,
            logw,
            atom_mask,
            noise_params,
            frozen_model=frozen_diffusion if cfg.beta > 0 else None,
            beta=cfg.beta,
            network_condition_kwargs=nck,
        )
    return regularized_weighted_dsm_loss(
        model.structure_module,
        frozen_diffusion,
        x0,
        logw,
        atom_mask,
        noise_params,
        cfg,
        network_condition_kwargs=nck,
    )


def _coords_from_boltz_batch(batch: dict[str, Any]) -> torch.Tensor:
    coords = batch["coords"].float()
    if coords.ndim == 4:
        if coords.shape[1] != 1:
            raise ValueError(f"Expected one conformer per WDSM sample, got coords shape {coords.shape}")
        coords = coords[:, 0]
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords as (B, M, 3), got {coords.shape}")
    return coords


def compute_boltz2_trunk_to_network_kwargs(model: Any, batch: dict[str, Any], *, recycling_steps: int):
    """Lazy wrapper around Boltz-2 trunk conditioning to keep imports lightweight."""
    from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs  # noqa: PLC0415

    return boltz2_trunk_to_network_kwargs(model, batch, recycling_steps=recycling_steps)


def transfer_multi_protein_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors to device while leaving metadata/list fields on host."""
    skip = {
        "all_coords",
        "all_resolved_mask",
        "crop_to_all_atom_map",
        "chain_symmetries",
        "amino_acids_symmetries",
        "ligand_symmetries",
        "record",
        "affinity_mw",
        "wdsm_record_id",
    }
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if key in skip:
            out[key] = val
        elif torch.is_tensor(val):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def _build_scheduler(
    optimizer: Adam,
    *,
    cfg: WeightedDSMConfig,
    lr_schedule: str,
    lr_warmup_epochs: int,
    lr_min: float,
):
    if lr_schedule != "cosine":
        return None
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_epochs = max(0, int(lr_warmup_epochs))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=lr_min)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs - warmup_epochs),
        eta_min=lr_min,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

#!/usr/bin/env python3
"""Multi-protein WDSM fine-tuning for Boltz-2 on ATLAS-prepared data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.utils.compute_device import (  # noqa: E402
    maybe_set_torch_cuda_current_device,
    parse_torch_device,
)

from genai_tps.training.config import WeightedDSMConfig  # noqa: E402
from genai_tps.training.multi_protein_dataset import (  # noqa: E402
    MultiProteinWdsmDataset,
    multi_protein_collate,
    read_frame_map,
)
from genai_tps.training.multi_protein_trainer import run_multi_protein_wdsm_training  # noqa: E402


def load_boltz2_for_multi_protein_training(
    *,
    cache: Path,
    device: torch.device,
    diffusion_steps: int,
    recycling_steps: int,
    kernels: bool = False,
):
    """Load a Boltz-2 checkpoint without constructing a single-YAML inference bundle."""
    from boltz.main import (  # noqa: PLC0415
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        download_boltz2,
    )
    from boltz.model.models.boltz2 import Boltz2  # noqa: PLC0415

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False
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
        str(cache / "boltz2_conf.ckpt"),
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-protein WDSM fine-tuning for Boltz-2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path, required=True)
    parser.add_argument("--mol-dir", type=Path, default=None)
    parser.add_argument("--constraints-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--extra-mols-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device: cpu, cuda, or cuda:N (default: cuda).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--save-every-batches", type=int, default=0)
    parser.add_argument("--loss-type", type=str, default="true-quotient",
                        choices=["cartesian", "quotient", "true-quotient"])
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-seqs", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()

    from boltz.data.types import Manifest  # noqa: PLC0415

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    mol_dir = args.mol_dir.expanduser() if args.mol_dir else cache / "mols"
    if torch.cuda.is_available():
        device = parse_torch_device(args.device)
        maybe_set_torch_cuda_current_device(device)
    else:
        device = torch.device("cpu")
    manifest = Manifest.load(args.manifest.expanduser())
    samples = read_frame_map(args.frame_map.expanduser())
    train_samples, val_samples = _split_samples(samples, val_fraction=args.val_fraction, seed=args.seed)

    dataset_kwargs = dict(
        manifest=manifest,
        target_dir=args.target_dir.expanduser(),
        msa_dir=args.msa_dir.expanduser(),
        mol_dir=mol_dir,
        constraints_dir=args.constraints_dir.expanduser() if args.constraints_dir else None,
        template_dir=args.template_dir.expanduser() if args.template_dir else None,
        extra_mols_dir=args.extra_mols_dir.expanduser() if args.extra_mols_dir else None,
        max_atoms=args.max_atoms,
        max_tokens=args.max_tokens,
        max_seqs=args.max_seqs,
        seed=args.seed,
    )
    train_ds = MultiProteinWdsmDataset(frame_samples=train_samples, **dataset_kwargs)
    val_ds = MultiProteinWdsmDataset(frame_samples=val_samples, **dataset_kwargs) if val_samples else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multi_protein_collate,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=multi_protein_collate,
        )

    model = load_boltz2_for_multi_protein_training(
        cache=cache,
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=args.kernels,
    )
    cfg = WeightedDSMConfig(
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )
    run_multi_protein_wdsm_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        loss_type=args.loss_type,
        device=device,
        work_root=args.out,
        recycling_steps=args.recycling_steps,
        lr_schedule=args.lr_schedule,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min=args.lr_min,
        early_stopping_patience=args.early_stopping_patience,
        resume_from=args.resume_from,
        save_every_batches=args.save_every_batches,
    )


def _split_samples(samples, *, val_fraction: float, seed: int):
    if val_fraction <= 0:
        return list(samples), []
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    split = int(round((1.0 - val_fraction) * len(samples)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


if __name__ == "__main__":
    main()

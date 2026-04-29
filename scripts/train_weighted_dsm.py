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
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from genai_tps.backends.boltz.inference import build_boltz_inference_session
from genai_tps.training.trainer import run_weighted_dsm_training


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
    parser.add_argument("--loss-type", type=str, default="cartesian", choices=["cartesian", "quotient", "true-quotient"],
                        help="Loss: 'cartesian' (plain DSM), 'quotient' (AF3-style alignment, deprecated), "
                             "or 'true-quotient' (principled quotient-space, arXiv:2604.21809).")
    args = parser.parse_args()

    from genai_tps.training.config import WeightedDSMConfig
    from genai_tps.training.dataset import ReweightedStructureDataset
    from genai_tps.training.diagnostics import weight_statistics

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
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )

    ds = ReweightedStructureDataset.from_npz(data_path)

    stats = weight_statistics(ds.logw)
    print(f"[WDSM] Weight stats: N_eff={stats['n_eff']:.1f} "
          f"({stats['n_eff_fraction']*100:.1f}%), "
          f"max_weight_frac={stats['max_weight_fraction']:.4f}", flush=True)

    # Build Boltz session before DataLoaders so we know padded atom width (M_model).
    bundle = build_boltz_inference_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=work_root,
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=args.recycling_steps,
        kernels=args.kernels,
        quotient_space_sampling=args.loss_type == "true-quotient",
    )

    from genai_tps.training.dataset import pad_reweighted_dataset_to_boltz_atom_mask

    pad_ref = bundle.core.atom_mask
    ds = pad_reweighted_dataset_to_boltz_atom_mask(ds, pad_ref)

    # Optional validation set (same Boltz pad mask as training)
    val_loader = None
    if args.val_data is not None:
        val_path = Path(args.val_data).expanduser().resolve()
        if val_path.is_file():
            val_ds = ReweightedStructureDataset.from_npz(val_path)
            val_ds = pad_reweighted_dataset_to_boltz_atom_mask(val_ds, pad_ref)
            val_stats = weight_statistics(val_ds.logw)
            print(f"[WDSM] Val set: {len(val_ds)} samples, N_eff={val_stats['n_eff']:.1f}", flush=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    run_weighted_dsm_training(
        args,
        cfg=cfg,
        ds=ds,
        val_loader=val_loader,
        stats=stats,
        bundle=bundle,
        device=device,
        work_root=work_root,
        data_path=data_path,
        loader=loader,
    )


if __name__ == "__main__":
    main()

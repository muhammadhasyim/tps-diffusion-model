#!/usr/bin/env python3
"""Stage 2 — Supervised fine-tuning of Boltz-2 on physical reference data.

Runs weighted denoising score matching (DSM) fine-tuning for each of the three
diagnostic systems using the reference data assembled in Stage 1.  Two loss
variants are supported and can be run in sequence:

  ``cartesian``   — plain weighted DSM in Euclidean coordinate space
  ``true-quotient`` — principled quotient-space loss (arXiv 2604.21809) that
                      projects out global rigid-body motion before supervision

Both variants can be run; comparing them in Stage 5 (quality) and Stage 7 (TPS
analysis) is one of the experimental conditions.

Pipeline role:
    01_assemble_datasets  →  02_finetune_boltz2  →  03_generate_ensembles

Outputs (one subdirectory per system x loss type)::

    outputs/campaign/case{1,2,3}/sft_{loss_type}/
        boltz2_wdsm_final.pt       # final model state dict
        boltz2_wdsm_best.pt        # best val-loss checkpoint (if val set provided)
        boltz2_wdsm_epoch_*.pt     # periodic epoch checkpoints
        training_log.csv           # per-batch loss, grad norm, val loss
        training_summary.json      # config + epoch losses + convergence metadata

Example::

    # Quick smoke-test (5 epochs, CPU):
    python scripts/campaign/02_finetune_boltz2.py \\
        --out outputs/campaign \\
        --epochs 5 --batch-size 2 --device cpu --loss-types cartesian

    # Minimal VRAM on GPU (e.g. share card with Stage 00 MD): batch size 1
    python scripts/campaign/02_finetune_boltz2.py \\
        --out outputs/00_generate/run_YYYYMMDD_HHMMSS \\
        --cases 1 --epochs 50 --batch-size 1 --device cuda \\
        --diffusion-steps 8 --recycling-steps 1

    # Production run (both loss types, GPU):
    python scripts/campaign/02_finetune_boltz2.py \\
        --out outputs/campaign \\
        --epochs 50 --batch-size 4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

_CASE_YAMLS = {
    "case1_mek1_fzc_novel":      "inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "case2_cdk2_atp_wildtype":   "inputs/tps_diagnostic/case2_cdk2_atp_wildtype.yaml",
    "case3_cdk2_atp_packed":     "inputs/tps_diagnostic/case3_cdk2_atp_packed.yaml",
}


def _resolve_campaign_out_root(raw: Path) -> Path:
    """Resolve ``--out`` for relative paths (same rules as ``01_assemble_datasets``)."""
    expanded = raw.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_root = (Path.cwd() / expanded).resolve()
    repo_root = (_REPO_ROOT / expanded).resolve()
    names = list(_CASE_YAMLS.keys())

    def _looks_like_campaign_root(p: Path) -> bool:
        if not p.is_dir():
            return False
        for name in names:
            if (p / name / "training_dataset.npz").is_file():
                return True
            if (p / name / "openmm_opes_md").is_dir():
                return True
        return False

    if _looks_like_campaign_root(cwd_root):
        return cwd_root
    if _looks_like_campaign_root(repo_root):
        return repo_root
    return cwd_root


def _train(
    *,
    yaml_path: Path,
    data_npz: Path,
    val_npz: Path | None,
    out_dir: Path,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    max_grad_norm: float,
    checkpoint_every: int,
    save_every_batches: int,
    lr_schedule: str,
    lr_warmup_epochs: int,
    lr_min: float,
    early_stopping_patience: int,
    diffusion_steps: int,
    recycling_steps: int,
    device: str,
    cache: Path,
    resume_from: Path | None,
) -> None:
    """Invoke train_weighted_dsm.py as a subprocess."""
    train_script = _REPO_ROOT / "scripts" / "train_weighted_dsm.py"
    cmd = [
        sys.executable, str(train_script),
        "--yaml", str(yaml_path),
        "--data", str(data_npz),
        "--out", str(out_dir),
        "--device", device,
        "--loss-type", loss_type,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--beta", str(beta),
        "--max-grad-norm", str(max_grad_norm),
        "--checkpoint-every", str(checkpoint_every),
        "--save-every-batches", str(save_every_batches),
        "--lr-schedule", lr_schedule,
        "--lr-warmup-epochs", str(lr_warmup_epochs),
        "--lr-min", str(lr_min),
        "--early-stopping-patience", str(early_stopping_patience),
        "--diffusion-steps", str(diffusion_steps),
        "--recycling-steps", str(recycling_steps),
        "--cache", str(cache),
    ]
    if val_npz is not None and val_npz.is_file():
        cmd += ["--val-data", str(val_npz)]
    if resume_from is not None and resume_from.is_file():
        cmd += ["--resume-from", str(resume_from)]

    print(f"  [02] Running: {' '.join(cmd[:6])} ...", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"train_weighted_dsm.py exited with code {result.returncode}. "
            "See stderr above."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Fine-tune Boltz-2 on OPES-MD reference data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cases", type=str, default="1,2,3")
    parser.add_argument("--loss-types", type=str, default="cartesian,true-quotient",
                        help="Comma-separated list of loss variants to train.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-step batch size (minimum 1). Use 1 to minimize VRAM when sharing the GPU.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL regularisation weight vs frozen baseline.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--save-every-batches", type=int, default=0)
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Stop if val loss stalls for this many epochs.")
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Skip training if the final checkpoint already exists.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    out_root = _resolve_campaign_out_root(args.out)
    print(f"[02] Campaign output root: {out_root}", flush=True)
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    loss_types = [lt.strip() for lt in args.loss_types.split(",")]
    case_names = list(_CASE_YAMLS.keys())
    training_log = []

    for idx, name in enumerate(case_names, start=1):
        if idx not in selected_cases:
            continue

        case_dir = out_root / name
        data_npz = case_dir / "training_dataset.npz"
        val_npz = case_dir / "dataset_split" / "val.npz"
        if not data_npz.is_file():
            raise FileNotFoundError(
                f"Training NPZ not found: {data_npz}\n"
                "Run Stage 1 (01_assemble_datasets.py) first."
            )
        val_npz_arg = val_npz if val_npz.is_file() else None

        yaml_path = _REPO_ROOT / _CASE_YAMLS[name]

        for loss_type in loss_types:
            print(f"\n{'='*60}")
            print(f"  Case {idx}: {name}  |  loss={loss_type}")
            print(f"{'='*60}", flush=True)

            sft_dir = case_dir / f"sft_{loss_type}"
            sft_dir.mkdir(parents=True, exist_ok=True)

            final_ckpt = sft_dir / "boltz2_wdsm_final.pt"
            if args.resume and final_ckpt.is_file():
                print(f"  [02] Checkpoint exists at {final_ckpt}; skipping.", flush=True)
                training_log.append({"case": name, "loss_type": loss_type, "status": "skipped_resume"})
                continue

            # Allow resuming mid-run from latest epoch checkpoint
            resume_from = None
            if args.resume:
                ckpt_candidates = sorted(sft_dir.glob("boltz2_wdsm_epoch_*.pt"))
                if ckpt_candidates:
                    resume_from = ckpt_candidates[-1]
                    print(f"  [02] Resuming from {resume_from}", flush=True)

            _train(
                yaml_path=yaml_path,
                data_npz=data_npz,
                val_npz=val_npz_arg,
                out_dir=sft_dir,
                loss_type=loss_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta=args.beta,
                max_grad_norm=args.max_grad_norm,
                checkpoint_every=args.checkpoint_every,
                save_every_batches=args.save_every_batches,
                lr_schedule=args.lr_schedule,
                lr_warmup_epochs=args.lr_warmup_epochs,
                lr_min=args.lr_min,
                early_stopping_patience=args.early_stopping_patience,
                diffusion_steps=args.diffusion_steps,
                recycling_steps=args.recycling_steps,
                device=args.device,
                cache=cache,
                resume_from=resume_from,
            )

            summary_path = sft_dir / "training_summary.json"
            summary = {}
            if summary_path.is_file():
                with open(summary_path) as fh:
                    summary = json.load(fh)

            training_log.append({
                "case": name,
                "loss_type": loss_type,
                "status": "complete",
                "sft_dir": str(sft_dir),
                "final_checkpoint": str(final_ckpt),
                "best_val_loss": summary.get("best_val_loss"),
                "final_epoch": summary.get("final_epoch"),
                "stopped_early": summary.get("stopped_early"),
            })
            print(f"  [02] Done: {final_ckpt}", flush=True)

    log_path = out_root / "02_training_log.json"
    with open(log_path, "w") as fh:
        json.dump(training_log, fh, indent=2)
    print(f"\n[02] Training log: {log_path}", flush=True)
    print("[02] Stage 2 complete. Run 03_generate_ensembles.py next.", flush=True)


if __name__ == "__main__":
    main()

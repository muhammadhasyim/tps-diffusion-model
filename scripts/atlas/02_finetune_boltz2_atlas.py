#!/usr/bin/env python3
"""Fine-tune Boltz-2 on an ATLAS-prepared WDSM NPZ dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def build_train_command(
    *,
    train_script: Path,
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
) -> list[str]:
    """Construct a ``train_weighted_dsm.py`` subprocess command."""
    cmd = [
        sys.executable,
        str(train_script),
        "--yaml",
        str(yaml_path),
        "--data",
        str(data_npz),
        "--out",
        str(out_dir),
        "--device",
        device,
        "--loss-type",
        loss_type,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--beta",
        str(beta),
        "--max-grad-norm",
        str(max_grad_norm),
        "--checkpoint-every",
        str(checkpoint_every),
        "--save-every-batches",
        str(save_every_batches),
        "--lr-schedule",
        lr_schedule,
        "--lr-warmup-epochs",
        str(lr_warmup_epochs),
        "--lr-min",
        str(lr_min),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--diffusion-steps",
        str(diffusion_steps),
        "--recycling-steps",
        str(recycling_steps),
        "--cache",
        str(cache),
    ]
    if val_npz is not None:
        cmd += ["--val-data", str(val_npz)]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Boltz-2 on an ATLAS WDSM dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml", type=Path, default=None,
                        help="Boltz YAML for the target represented by --data.")
    parser.add_argument("--yaml-map", type=Path, default=None,
                        help="Optional JSON mapping ATLAS IDs to YAMLs for --ids-file mode.")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="Optional IDs file; trains each prepared ID under --prepared-dir.")
    parser.add_argument("--prepared-dir", type=Path, default=None,
                        help="Directory created by 01_prepare_atlas_wdsm.py.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--loss-types", type=str, default="true-quotient")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--save-every-batches", type=int, default=0)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    runs = _resolve_runs(args)
    train_script = _REPO_ROOT / "scripts" / "train_weighted_dsm.py"
    loss_types = [loss_type.strip() for loss_type in args.loss_types.split(",") if loss_type.strip()]
    log = []

    for run_name, yaml_path, data_npz, val_npz in runs:
        for loss_type in loss_types:
            out_dir = args.out.expanduser().resolve() / run_name / f"sft_{loss_type}"
            final_ckpt = out_dir / "boltz2_wdsm_final.pt"
            if args.resume and final_ckpt.is_file():
                print(f"[atlas-finetune] skipping existing checkpoint: {final_ckpt}")
                log.append({"run": run_name, "loss_type": loss_type, "status": "skipped_resume"})
                continue
            resume_from = None
            if args.resume:
                candidates = sorted(out_dir.glob("boltz2_wdsm_epoch_*.pt"))
                if candidates:
                    resume_from = candidates[-1]

            cmd = build_train_command(
                train_script=train_script,
                yaml_path=yaml_path,
                data_npz=data_npz,
                val_npz=val_npz,
                out_dir=out_dir,
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
            print(f"[atlas-finetune] {' '.join(cmd[:8])} ... loss={loss_type}")
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"train_weighted_dsm.py failed with exit code {result.returncode}.")
            log.append(
                {
                    "run": run_name,
                    "loss_type": loss_type,
                    "yaml": str(yaml_path),
                    "data": str(data_npz),
                    "val_data": str(val_npz) if val_npz is not None else None,
                    "out_dir": str(out_dir),
                    "status": "dry-run" if args.dry_run else "complete",
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "atlas_finetune_log.json").write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_runs(args) -> list[tuple[str, Path, Path, Path | None]]:
    if args.ids_file is not None:
        if args.prepared_dir is None:
            raise FileNotFoundError("--prepared-dir is required with --ids-file.")
        yaml_map = _load_yaml_map(args.yaml_map)
        from genai_tps.data.atlas import read_ids_file  # noqa: PLC0415

        runs = []
        for atlas_id in read_ids_file(args.ids_file):
            yaml_path = yaml_map.get(atlas_id)
            if yaml_path is None:
                raise FileNotFoundError(f"No YAML mapping found for {atlas_id} in --yaml-map.")
            data_npz = args.prepared_dir / atlas_id / "training_dataset.npz"
            val_npz = args.prepared_dir / atlas_id / "dataset_split" / "val.npz"
            runs.append((atlas_id, yaml_path, data_npz, val_npz if val_npz.is_file() else None))
        _validate_runs(runs)
        return runs

    if args.yaml is None or args.data is None:
        raise FileNotFoundError("Provide either --ids-file/--prepared-dir/--yaml-map or --yaml and --data.")
    runs = [(
        args.data.stem,
        args.yaml.expanduser(),
        args.data.expanduser(),
        args.val_data.expanduser() if args.val_data is not None else None,
    )]
    _validate_runs(runs)
    return runs


def _load_yaml_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): Path(v).expanduser() for k, v in payload.items()}


def _validate_runs(runs: list[tuple[str, Path, Path, Path | None]]) -> None:
    for run_name, yaml_path, data_npz, val_npz in runs:
        if not yaml_path.is_file():
            raise FileNotFoundError(f"YAML not found for {run_name}: {yaml_path}")
        if not data_npz.is_file():
            raise FileNotFoundError(f"Training NPZ not found for {run_name}: {data_npz}")
        if val_npz is not None and not val_npz.is_file():
            raise FileNotFoundError(f"Validation NPZ not found for {run_name}: {val_npz}")


if __name__ == "__main__":
    main()

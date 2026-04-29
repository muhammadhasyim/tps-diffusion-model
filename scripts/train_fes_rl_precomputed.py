#!/usr/bin/env python3
"""FES-guided RL using a pre-converged OPES bias as teacher (no OpenMM needed).

Loads a converged OPES state from OPES-TPS exploration and uses it as the
target distribution for PPO-IS fine-tuning of the Boltz-2 diffusion model.

The advantage signal is: log p_target(cv) - log p_student(cv)
where p_target comes from the OPES bias: log p_target = -V(s) / kT
and p_student is estimated via a sliding-window KDE over Boltz rollout CVs.

Example::

    python scripts/train_fes_rl_precomputed.py \
        --out ./fes_rl_out \
        --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
        --opes-state /path/to/opes_state_final.json \
        --n-iters 20 --rollouts-per-iter 4 --diffusion-steps 8
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


def _exit_if_rl_excluded() -> None:
    try:
        import genai_tps.rl.config  # noqa: F401
    except ImportError:
        print(
            "This script requires the optional genai_tps.rl package.\n"
            "Restore it with: git checkout HEAD -- src/python/genai_tps/rl",
            file=sys.stderr,
        )
        sys.exit(2)


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

    struct_candidates = sorted((processed_dir / "structures").glob("*.npz"))
    topo_npz = struct_candidates[0] if struct_candidates else None

    return model, core, batch, processed_dir, topo_npz, boltz_run_dir


class PrecomputedOPESTeacher:
    """Lightweight teacher using a pre-converged OPES bias (no OpenMM)."""

    def __init__(self, opes, kbt: float = 1.0):
        self.opes = opes
        self.kbt = kbt

    def log_p_target(self, cv) -> float:
        v = float(self.opes.evaluate(cv))
        return -v / self.kbt


def main() -> None:
    _exit_if_rl_excluded()
    parser = argparse.ArgumentParser(
        description="FES-guided RL with pre-converged OPES teacher (no OpenMM)."
    )
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--rollouts-per-iter", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--tau-sq", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--topo-npz", type=Path, default=None)
    parser.add_argument("--pocket-radius", type=float, default=8.0)
    parser.add_argument(
        "--opes-state", type=Path, required=True,
        help="Path to converged OPES state JSON from OPES-TPS run.",
    )
    parser.add_argument("--opes-kbt", type=float, default=1.0)
    parser.add_argument("--student-kde-window", type=int, default=200)
    parser.add_argument("--student-kde-bandwidth", type=float, default=None)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    args = parser.parse_args()

    yaml_path = args.yaml or (
        _REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml"
    )
    if not yaml_path.is_file():
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from genai_tps.io.boltz_npz_export import load_topo
        from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
        from genai_tps.simulation import OPESBias
        from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
        from genai_tps.rl.fes_teacher import boltz_terminal_pose_cv_numpy
        from genai_tps.rl.rollout import rollout_forward_trajectory
        from genai_tps.rl.student_distribution import BoltzStudentKDE
        from genai_tps.rl.training import fes_guided_trajectory_loss
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
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

    topo_npz = (
        Path(args.topo_npz).expanduser().resolve() if args.topo_npz else topo_auto
    )
    if topo_npz is None or not topo_npz.is_file():
        print("Topology NPZ required; pass --topo-npz.", file=sys.stderr)
        sys.exit(1)

    structure, n_struct = load_topo(topo_npz)
    ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)[
        : int(n_struct)
    ]
    indexer = PoseCVIndexer(
        structure, ref_coords, pocket_radius=float(args.pocket_radius)
    )
    if len(indexer.ligand_idx) == 0:
        print("ERROR: No ligand chain found.", file=sys.stderr)
        sys.exit(1)

    print(
        f"[setup] ligand atoms: {len(indexer.ligand_idx)}, "
        f"pocket Calpha: {len(indexer.pocket_ca_idx)}, "
        f"pocket heavy: {len(indexer.pocket_heavy_idx)}",
        flush=True,
    )

    # Load pre-converged OPES bias
    opes_path = Path(args.opes_state).expanduser().resolve()
    opes = OPESBias.load_state(opes_path)
    print(
        f"[OPES teacher] Loaded {opes.n_kernels} kernels, "
        f"{opes.counter} depositions from {opes_path.name}",
        flush=True,
    )

    teacher = PrecomputedOPESTeacher(opes, kbt=float(args.opes_kbt))
    student_kde = BoltzStudentKDE(
        2,
        window=args.student_kde_window,
        bandwidth=args.student_kde_bandwidth,
    )

    rl_cfg = BoltzRLConfig(
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        velocity_log_prob_tau_sq=args.tau_sq,
        max_grad_norm=args.max_grad_norm,
    )
    fes_cfg = FESTeacherConfig(
        advantage_clip=float(args.advantage_clip),
    )

    params = list(model.structure_module.parameters())
    optimizer = Adam(params, lr=rl_cfg.learning_rate)

    # Gradient fix: enable requires_grad on conditioning tensors
    nk = core.network_condition_kwargs
    for key in ("s_trunk", "s_inputs"):
        if key in nk and isinstance(nk[key], torch.Tensor) and not nk[key].requires_grad:
            nk[key] = nk[key].detach().clone().requires_grad_(True)
    print("[grad fix] Set requires_grad on conditioning tensors", flush=True)

    # CSV log
    log_path = work_root / "training_log.csv"
    log_fields = [
        "iter", "rollout", "ligand_rmsd", "pocket_dist",
        "log_p_target", "log_p_student", "advantage", "loss",
    ]
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    writer.writeheader()

    iter_summaries: list[dict] = []
    best_avg_adv = float("-inf")
    best_iter = 0

    print(f"\n{'='*72}")
    print(f"  FES-guided RL: {args.n_iters} iters x {args.rollouts_per_iter} rollouts")
    print(f"  LR={args.learning_rate:.1e}  clip={args.clip_range}  "
          f"grad_norm={args.max_grad_norm}  diffusion_steps={args.diffusion_steps}")
    print(f"  OPES teacher: {opes.n_kernels} kernels, kbt={args.opes_kbt}")
    print(f"  Advantage clip: {args.advantage_clip}")
    print(f"{'='*72}\n", flush=True)

    for it in range(1, args.n_iters + 1):
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        # Collect rollouts
        trajectories = []
        cvs = []
        model.eval()
        with torch.inference_mode():
            core.diffusion.eval()
            for _ in range(args.rollouts_per_iter):
                tr = rollout_forward_trajectory(
                    core, num_steps=args.diffusion_steps
                )
                trajectories.append(tr)
                cv = boltz_terminal_pose_cv_numpy(
                    tr[-1].x_next, int(n_struct), indexer
                )
                cvs.append(cv)

        # Update student KDE
        for cv in cvs:
            student_kde.update(cv)

        # Train
        model.train()
        model.structure_module.train()
        core.diffusion.train()

        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        iter_advantages = []
        iter_rmsds = []
        iter_dists = []

        for r_idx, (tr, cv) in enumerate(zip(trajectories, cvs)):
            log_pt = float(teacher.log_p_target(cv))
            log_ps = float(student_kde.log_density(cv))
            adv = float(
                np.clip(
                    log_pt - log_ps,
                    -args.advantage_clip,
                    args.advantage_clip,
                )
            )

            loss = fes_guided_trajectory_loss(
                core, tr, cv, teacher, student_kde,
                fes_cfg=fes_cfg, rl_cfg=rl_cfg,
            )
            total_loss = total_loss + loss

            loss_val = float(loss.detach().cpu().item())
            iter_advantages.append(adv)
            iter_rmsds.append(float(cv[0]))
            iter_dists.append(float(cv[1]))

            writer.writerow({
                "iter": it,
                "rollout": r_idx + 1,
                "ligand_rmsd": f"{cv[0]:.4f}",
                "pocket_dist": f"{cv[1]:.4f}",
                "log_p_target": f"{log_pt:.4f}",
                "log_p_student": f"{log_ps:.4f}",
                "advantage": f"{adv:.4f}",
                "loss": f"{loss_val:.4f}",
            })

            print(
                f"  i{it:03d} r{r_idx+1}: "
                f"rmsd={cv[0]:6.2f}A  pkt_d={cv[1]:6.2f}A  "
                f"logT={log_pt:+.2f} logS={log_ps:+.2f} "
                f"A={adv:+.3f} L={loss_val:.3f}",
                flush=True,
            )

        total_loss.backward()
        if rl_cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, rl_cfg.max_grad_norm)
        optimizer.step()

        log_f.flush()

        avg_loss = float(total_loss.detach().cpu().item()) / args.rollouts_per_iter
        avg_adv = float(np.mean(iter_advantages))
        avg_rmsd = float(np.mean(iter_rmsds))
        avg_dist = float(np.mean(iter_dists))

        summary = {
            "iter": it,
            "avg_loss": avg_loss,
            "avg_advantage": avg_adv,
            "avg_rmsd": avg_rmsd,
            "avg_pocket_dist": avg_dist,
            "best_rmsd": float(min(iter_rmsds)),
        }
        iter_summaries.append(summary)

        print(
            f"[Iter {it:03d}] avg_L={avg_loss:.3f}  avg_A={avg_adv:+.3f}  "
            f"avg_rmsd={avg_rmsd:.2f}A  avg_pkt_d={avg_dist:.2f}A  "
            f"best_rmsd={min(iter_rmsds):.2f}A",
            flush=True,
        )

        if avg_adv > best_avg_adv:
            best_avg_adv = avg_adv
            best_iter = it
            torch.save(
                model.state_dict(), work_root / "boltz2_fes_rl_best.pt"
            )
            print(f"  >> New best (avg_advantage={best_avg_adv:+.4f})", flush=True)

        # Save checkpoints periodically
        if it % max(1, args.n_iters // 5) == 0 or it == args.n_iters:
            ckpt = work_root / f"boltz2_fes_rl_iter_{it:03d}.pt"
            torch.save(model.state_dict(), ckpt)

        print("", flush=True)

    log_f.close()

    # Final summary
    final = {
        "config": {
            "n_iters": args.n_iters,
            "rollouts_per_iter": args.rollouts_per_iter,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "max_grad_norm": args.max_grad_norm,
            "diffusion_steps": args.diffusion_steps,
            "advantage_clip": args.advantage_clip,
            "opes_state": str(args.opes_state),
            "opes_kbt": args.opes_kbt,
        },
        "best_iter": best_iter,
        "best_avg_advantage": best_avg_adv,
        "iter_summaries": iter_summaries,
    }
    summary_path = work_root / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"{'='*72}")
    print(f"  Training complete. Best iter: {best_iter} "
          f"(avg_advantage={best_avg_adv:+.4f})")
    print(f"\n  {'Iter':>4} {'AvgRMSD':>8} {'AvgPktDist':>11} "
          f"{'AvgAdvantage':>13} {'AvgLoss':>8}")
    print(f"  {'-'*4} {'-'*8} {'-'*11} {'-'*13} {'-'*8}")
    for s in iter_summaries:
        print(
            f"  {s['iter']:4d} {s['avg_rmsd']:8.2f} {s['avg_pocket_dist']:11.2f} "
            f"{s['avg_advantage']:+13.4f} {s['avg_loss']:8.3f}"
        )
    print(f"{'='*72}", flush=True)


if __name__ == "__main__":
    main()

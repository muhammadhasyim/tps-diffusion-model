#!/usr/bin/env python3
"""Stage 6 — Run OPES-TPS diagnostics on all 3 systems × 2 model variants.

Uses the Boltz-2 diffusion engine (not OpenMM) as the path-sampling engine.
TPS path ensembles characterize the model's generative tube: how sharply do
its paths funnel toward a target state?  The sharpness, variance, and
acceptance rate of paths reveal failure modes invisible to single-shot
evaluation.

State definitions (from inputs/tps_diagnostic/README.md):

  Case 1 — MEK1+FZC novel:
    State A: Gaussian noise (high σ, diffusion t=0)
    State B: RMSD < 2 Å to 7XLP crystal FZC pose AND LDDT-PLI proxy > 0.8
    Expected: High path variance, low B-fraction → negative control

  Case 2 — CDK2+ATP WT memorised:
    State A: Gaussian noise
    State B: RMSD < 2 Å to 1B38 crystal ATP pose AND LDDT-PLI proxy > 0.8
    Expected: Narrow tube, high B-fraction → positive control

  Case 3 — CDK2+ATP Phe-packed adversarial:
    State A: Gaussian noise
    State B_crystal: RMSD < 2 Å to 1B38 crystal ATP pose (model's spurious target)
    State B_phys: No steric clashes above threshold + ≥ 1 polar contact present
    Expected: Funnels to B_crystal but NOT B_phys → exposes confidence failure

Pipeline role:
    02_finetune_boltz2  →  06_run_tps_diagnostic  →  07_analyze_tps_failure_modes

Outputs::

    outputs/campaign/case{1,2,3}/{model_variant}/tps_run/
        trajectory_checkpoints/tps_mc_step_*.npz
        opes_states/
        opes_state_final.json
        opes_tps_summary.json
        cv_values.json

Example::

    # Quick smoke test:
    python scripts/campaign/06_run_tps_diagnostic.py \\
        --out outputs/campaign --shoot-rounds 200 --diffusion-steps 16

    # Production:
    python scripts/campaign/06_run_tps_diagnostic.py \\
        --out outputs/campaign --shoot-rounds 5000 --diffusion-steps 200
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

from genai_tps.subprocess_support import child_env_with_repo_src_python

_CASE_YAMLS = {
    "case1_mek1_fzc_novel":    "inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml",
    "case2_cdk2_atp_wildtype": "inputs/tps_diagnostic/case2_cdk2_atp_wildtype.yaml",
    "case3_cdk2_atp_packed":   "inputs/tps_diagnostic/case3_cdk2_atp_packed.yaml",
}

# TPS bias CV: ligand RMSD biased toward low values (native binding geometry)
# For Cases 1 & 2 this is sufficient.  For Case 3 we run two separate TPS
# runs: one biasing toward B_crystal (RMSD-based) and one biasing toward
# B_phys (defined via clash_count + ligand_contacts composite).
_CASE_BIAS_CONFIGS: dict[str, list[dict]] = {
    "case1_mek1_fzc_novel": [
        {
            "label": "b_crystal",
            "bias_cv": "ligand_rmsd",
            "description": "Bias toward RMSD < 2 Å (crystal pose) — negative control",
        }
    ],
    "case2_cdk2_atp_wildtype": [
        {
            "label": "b_crystal",
            "bias_cv": "ligand_rmsd",
            "description": "Bias toward RMSD < 2 Å (crystal pose) — positive control",
        }
    ],
    "case3_cdk2_atp_packed": [
        {
            "label": "b_crystal",
            "bias_cv": "ligand_rmsd",
            "description": "Bias toward RMSD < 2 Å (B_crystal) — exposes spurious confidence",
        },
        {
            "label": "b_phys",
            "bias_cv": "ligand_rmsd,clash_count",
            "description": "2D bias: minimise RMSD AND clash_count — probes physical plausibility",
        },
    ],
}


def _checkpoint_path(model_variant: str, case_dir: Path) -> Path | None:
    """Resolve the fine-tuned checkpoint for a given variant name."""
    if model_variant == "baseline":
        return None
    ckpt = case_dir / model_variant / "boltz2_wdsm_final.pt"
    if ckpt.is_file():
        return ckpt
    ckpt_best = case_dir / model_variant / "boltz2_wdsm_best.pt"
    if ckpt_best.is_file():
        return ckpt_best
    return None


def _run_tps(
    *,
    yaml_path: Path,
    out_dir: Path,
    cache: Path,
    device: str,
    checkpoint: Path | None,
    bias_cv: str,
    shoot_rounds: int,
    diffusion_steps: int,
    recycling_steps: int,
    opes_barrier: float,
    opes_biasfactor: float,
    opes_pace: int,
    save_trajectory_every: int,
    save_opes_state_every: int,
    progress_every: int,
    topo_npz: Path | None,
    n_fixed_point: int,
    diagnostic_cvs: str,
    opes_restart: Path | None,
    openmm_device_index: int | None = None,
) -> None:
    """Invoke run_opes_tps.py as an isolated subprocess."""
    tps_script = _REPO_ROOT / "scripts" / "run_opes_tps.py"
    cmd = [
        sys.executable, str(tps_script),
        "--yaml", str(yaml_path),
        "--out", str(out_dir),
        "--cache", str(cache),
        "--device", device,
        "--bias-cv", bias_cv,
        "--shoot-rounds", str(shoot_rounds),
        "--diffusion-steps", str(diffusion_steps),
        "--recycling-steps", str(recycling_steps),
        "--opes-barrier", str(opes_barrier),
        "--opes-biasfactor", str(opes_biasfactor),
        "--opes-pace", str(opes_pace),
        "--save-trajectory-every", str(save_trajectory_every),
        "--save-opes-state-every", str(save_opes_state_every),
        "--progress-every", str(progress_every),
        "--n-fixed-point", str(n_fixed_point),
        "--diagnostic-cvs", diagnostic_cvs,
    ]
    if topo_npz is not None:
        cmd += ["--topo-npz", str(topo_npz)]
    if checkpoint is not None:
        cmd += ["--finetuned-checkpoint", str(checkpoint)]
    if opes_restart is not None:
        cmd += ["--opes-restart", str(opes_restart)]
    if openmm_device_index is not None:
        cmd += ["--openmm-device-index", str(int(openmm_device_index))]

    print(f"  [06] Launching TPS: {' '.join(cmd[:8])} ...", flush=True)
    result = subprocess.run(cmd, check=False, env=child_env_with_repo_src_python())
    if result.returncode != 0:
        raise RuntimeError(
            f"run_opes_tps.py exited with code {result.returncode}. See stderr above."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: OPES-TPS diagnostic runs for all systems × model variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--cases", type=str, default="1,2,3")
    parser.add_argument("--variants", type=str, default="baseline,sft_cartesian,sft_true-quotient",
                        help="Model variants to evaluate. Use 'baseline' for unmodified Boltz-2.")

    # TPS parameters
    parser.add_argument("--shoot-rounds", type=int, default=5000)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--opes-barrier", type=float, default=5.0)
    parser.add_argument("--opes-biasfactor", type=float, default=10.0)
    parser.add_argument("--opes-pace", type=int, default=1)
    parser.add_argument("--save-trajectory-every", type=int, default=10)
    parser.add_argument("--save-opes-state-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--n-fixed-point", type=int, default=4)
    parser.add_argument("--diagnostic-cvs", type=str,
                        default="contact_order,clash_count,rg,ramachandran_outlier",
                        help="Additional CVs logged per MC step for post-hoc analysis.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Forwarded to run_opes_tps.py: cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--openmm-device-index",
        type=int,
        default=None,
        metavar="N",
        help="Forwarded to run_opes_tps.py for OpenMM bias CVs (optional override).",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip if opes_state_final.json already exists for that run.")
    args = parser.parse_args()

    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    out_root = args.out.expanduser().resolve()
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    variants = [v.strip() for v in args.variants.split(",")]
    case_names = list(_CASE_YAMLS.keys())
    run_log = []

    for idx, name in enumerate(case_names, start=1):
        if idx not in selected_cases:
            continue

        yaml_path = _REPO_ROOT / _CASE_YAMLS[name]
        case_dir = out_root / name
        bias_configs = _CASE_BIAS_CONFIGS[name]

        for variant in variants:
            ckpt = _checkpoint_path(variant, case_dir)
            if variant != "baseline" and ckpt is None:
                print(f"\n[06] WARNING: No checkpoint for {name}/{variant}; skipping.", flush=True)
                run_log.append({"case": name, "variant": variant, "status": "no_checkpoint"})
                continue

            # Locate topology NPZ (from Stage 3 prep or Stage 0 prep)
            topo_candidates = sorted(case_dir.rglob("processed/structures/*.npz"))
            topo_npz = topo_candidates[0] if topo_candidates else None

            for bias_cfg in bias_configs:
                label = bias_cfg["label"]
                bias_cv = bias_cfg["bias_cv"]

                tps_out = case_dir / variant / "tps_run" / label
                tps_out.mkdir(parents=True, exist_ok=True)

                print(f"\n{'='*60}")
                print(f"  {name} | {variant} | {label}")
                print(f"  Bias CV: {bias_cv}")
                print(f"  {bias_cfg['description']}")
                print(f"{'='*60}", flush=True)

                final_state = tps_out / "opes_state_final.json"
                if args.resume and final_state.is_file():
                    print(f"  [06] Exists: {final_state}; skipping.", flush=True)
                    run_log.append({"case": name, "variant": variant, "label": label, "status": "skipped_resume"})
                    continue

                # Resume from latest OPES state if available
                opes_restart = None
                if args.resume:
                    latest = tps_out / "opes_states" / "opes_state_latest.json"
                    if latest.is_file():
                        opes_restart = latest
                        print(f"  [06] Resuming OPES from {opes_restart}", flush=True)

                _run_tps(
                    yaml_path=yaml_path,
                    out_dir=tps_out,
                    cache=cache,
                    device=args.device,
                    checkpoint=ckpt,
                    bias_cv=bias_cv,
                    shoot_rounds=args.shoot_rounds,
                    diffusion_steps=args.diffusion_steps,
                    recycling_steps=args.recycling_steps,
                    opes_barrier=args.opes_barrier,
                    opes_biasfactor=args.opes_biasfactor,
                    opes_pace=args.opes_pace,
                    save_trajectory_every=args.save_trajectory_every,
                    save_opes_state_every=args.save_opes_state_every,
                    progress_every=args.progress_every,
                    topo_npz=topo_npz,
                    n_fixed_point=args.n_fixed_point,
                    diagnostic_cvs=args.diagnostic_cvs,
                    opes_restart=opes_restart,
                    openmm_device_index=args.openmm_device_index,
                )

                # Count checkpoints
                ckpt_count = len(list((tps_out / "trajectory_checkpoints").glob("tps_mc_step_*.npz")))
                print(f"  [06] Done. {ckpt_count} trajectory checkpoints.", flush=True)
                run_log.append({
                    "case": name,
                    "variant": variant,
                    "label": label,
                    "bias_cv": bias_cv,
                    "status": "complete",
                    "n_checkpoints": ckpt_count,
                    "out_dir": str(tps_out),
                })

    log_path = out_root / "06_tps_run_log.json"
    with open(log_path, "w") as fh:
        json.dump(run_log, fh, indent=2)
    print(f"\n[06] Run log: {log_path}", flush=True)
    print("[06] Stage 6 complete. Run 07_analyze_tps_failure_modes.py next.", flush=True)


if __name__ == "__main__":
    main()

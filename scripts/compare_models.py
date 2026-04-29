#!/usr/bin/env python3
"""Rigorous 4-way model comparison: Training target vs Baseline vs Cartesian vs Quotient.

Generates:
  1. 6-CV histogram comparison (reweighted target, baseline, cartesian, quotient)
  2. KS statistics + Cohen's d effect sizes for each pair
  3. 2D FES heatmaps in (ligand_rmsd, pocket_dist) for each model
  4. Summary JSON with all metrics

Usage::

    python scripts/compare_models.py \
        --training-dir /mnt/shared/.../openmm_opes_md/wdsm_samples \
        --cartesian-checkpoint /mnt/shared/.../wdsm_training_v6/boltz2_wdsm_final.pt \
        --quotient-checkpoint /mnt/shared/.../wdsm_training_quotient/boltz2_wdsm_best.pt \
        --out /mnt/shared/.../comparison \
        --n-samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.inference import build_boltz_inference_session
from genai_tps.backends.boltz.collective_variables import compute_simple_cvs
from evaluation_dashboard import generate_four_way_dashboard, load_training_cvs
from genai_tps.evaluation.tps_runner import generate_structures


def main():
    parser = argparse.ArgumentParser(description="4-way model comparison")
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--cartesian-checkpoint", type=Path, default=None)
    parser.add_argument("--quotient-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml")
    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  4-Way Model Comparison")
    print(f"  Samples per model: {args.n_samples}")
    print(f"{'='*60}\n", flush=True)

    # Load training target
    print("[compare] Loading training CVs...", flush=True)
    training_cvs, training_weights = load_training_cvs(args.training_dir)
    print(f"  {len(training_cvs)} training samples loaded", flush=True)

    # Build baseline session
    print("[compare] Building Boltz session...", flush=True)
    bundle = build_boltz_inference_session(
        yaml_path=yaml_path,
        cache=cache,
        boltz_prep_dir=out / "prep",
        device=device,
        diffusion_steps=args.diffusion_steps,
        recycling_steps=1,
        kernels=False,
        quotient_space_sampling=False,
    )
    model, core, atom_mask, nkw = bundle.model, bundle.core, bundle.atom_mask, bundle.network_kwargs

    ref_x = core.sample_initial_noise()
    for step in range(core.num_sampling_steps):
        ref_x, *_ = core.single_forward_step(ref_x, step)
    ref_coords = ref_x[0].detach().cpu().numpy()

    models_data = {}

    # Baseline
    print("\n[compare] Generating BASELINE structures...", flush=True)
    baseline_structs = generate_structures(core, args.n_samples, device)
    models_data["Baseline"] = compute_simple_cvs(baseline_structs, ref_coords)

    # Cartesian-trained
    if args.cartesian_checkpoint and args.cartesian_checkpoint.is_file():
        print("\n[compare] Loading CARTESIAN checkpoint...", flush=True)
        state = torch.load(str(args.cartesian_checkpoint), map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        core_c = BoltzSamplerCore(
            model.structure_module,
            atom_mask,
            nkw,
            multiplicity=1,
            compile_model=core.compile_model,
            n_fixed_point=core.n_fixed_point,
            inference_dtype=core.inference_dtype,
            quotient_space_sampling=False,
        )
        core_c.build_schedule(args.diffusion_steps)
        print("[compare] Generating CARTESIAN structures...", flush=True)
        cart_structs = generate_structures(core_c, args.n_samples, device)
        models_data["Cartesian-trained"] = compute_simple_cvs(cart_structs, ref_coords)

    # Quotient-trained
    if args.quotient_checkpoint and args.quotient_checkpoint.is_file():
        print("\n[compare] Loading QUOTIENT checkpoint...", flush=True)
        state_q = torch.load(str(args.quotient_checkpoint), map_location="cpu")
        model.load_state_dict(state_q)
        model.to(device)
        core_q = BoltzSamplerCore(
            model.structure_module,
            atom_mask,
            nkw,
            multiplicity=1,
            compile_model=core.compile_model,
            n_fixed_point=core.n_fixed_point,
            inference_dtype=core.inference_dtype,
            quotient_space_sampling=True,
        )
        core_q.build_schedule(args.diffusion_steps)
        print("[compare] Generating QUOTIENT structures...", flush=True)
        quot_structs = generate_structures(core_q, args.n_samples, device)
        models_data["Quotient-trained"] = compute_simple_cvs(quot_structs, ref_coords)

    # Generate comparison
    print("\n[compare] Generating dashboard...", flush=True)
    ks_results = generate_four_way_dashboard(models_data, training_cvs, training_weights, out)

    # Summary
    report = {
        "n_samples": args.n_samples,
        "models": list(models_data.keys()),
        "ks_tests": ks_results,
        "summaries": {},
    }
    for name, data in models_data.items():
        report["summaries"][name] = {}
        for cv, vals in data.items():
            if len(vals) > 0:
                report["summaries"][name][cv] = {
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "min": float(vals.min()), "max": float(vals.max()),
                }

    with open(out / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Comparison Complete")
    print(f"  CV histograms: {out / 'cv_comparison.png'}")
    print(f"  FES heatmaps:  {out / 'fes_comparison.png'}")
    print(f"  Report:        {out / 'comparison_report.json'}")
    print(f"\n  KS Test Results (vs Baseline):")
    for cv, pairs in ks_results.items():
        for name, stats in pairs.items():
            sig = "***" if stats["p_value"] < 0.001 else "**" if stats["p_value"] < 0.01 else "*" if stats["p_value"] < 0.05 else ""
            print(f"    {cv} [{name}]: KS={stats['ks_statistic']:.3f} p={stats['p_value']:.3g} "
                  f"d={stats['cohens_d']:.3f} {sig}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

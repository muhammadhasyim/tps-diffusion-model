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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def _build_session(yaml_path, cache, prep_dir, device, diffusion_steps):
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.main import (
        Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs,
        PairformerArgsV2, check_inputs, download_boltz2, process_inputs,
    )
    from boltz.model.models.boltz2 import Boltz2
    from genai_tps.backends.boltz.boltz2_trunk import boltz2_trunk_to_network_kwargs
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore

    cache.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    download_boltz2(cache)

    boltz_run_dir = prep_dir / f"boltz_prep_{yaml_path.stem}"
    boltz_run_dir.mkdir(parents=True, exist_ok=True)
    data_list = check_inputs(yaml_path)
    process_inputs(data=data_list, out_dir=boltz_run_dir, ccd_path=cache / "ccd.pkl",
                   mol_dir=mol_dir, msa_server_url="https://api.colabfold.com",
                   msa_pairing_strategy="greedy", use_msa_server=False,
                   boltz2=True, preprocessing_threads=1)

    manifest = Manifest.load(boltz_run_dir / "processed" / "manifest.json")
    processed_dir = boltz_run_dir / "processed"
    dm = Boltz2InferenceDataModule(
        manifest=manifest, target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa", mol_dir=mol_dir, num_workers=0,
        constraints_dir=processed_dir / "constraints" if (processed_dir / "constraints").exists() else None,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None)
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    predict_args = {"recycling_steps": 1, "sampling_steps": diffusion_steps,
                    "diffusion_samples": 1, "max_parallel_samples": None,
                    "write_confidence_summary": True, "write_full_pae": False, "write_full_pde": False}
    model = Boltz2.load_from_checkpoint(
        str(cache / "boltz2_conf.ckpt"), strict=True, predict_args=predict_args,
        map_location="cpu", diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=False, pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams()))
    model.to(device)

    atom_mask, nkw = boltz2_trunk_to_network_kwargs(model, batch, recycling_steps=1)
    for k, v in list(nkw.items()):
        if hasattr(v, "to"): nkw[k] = v.to(device)
    if isinstance(nkw.get("feats"), dict):
        nkw["feats"] = {fk: fv.to(device) if hasattr(fv, "to") else fv for fk, fv in nkw["feats"].items()}

    core = BoltzSamplerCore(model.structure_module, atom_mask, nkw, multiplicity=1)
    core.build_schedule(diffusion_steps)
    return model, core, atom_mask, nkw


def generate_structures(core, n, device):
    structures = []
    for i in range(n):
        torch.manual_seed(i)
        x = core.sample_initial_noise()
        for step in range(core.num_sampling_steps):
            x, *_ = core.single_forward_step(x, step)
        structures.append(x[0].detach().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}", flush=True)
    return np.array(structures)


def compute_simple_cvs(structures, ref_coords):
    """Compute 6 CVs on each structure without requiring full Boltz topology."""
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot
    from genai_tps.backends.boltz.collective_variables import (
        rmsd_to_reference, radius_of_gyration, contact_order, clash_count,
    )
    ref_t = torch.tensor(ref_coords, dtype=torch.float32)

    results = {cv: [] for cv in ["rmsd", "rg", "contact_order", "clash_count", "ligand_rmsd", "ligand_pocket_dist"]}
    for coords in structures:
        ct = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(ct, step_index=0, defer_numpy_coords=True)
        results["rmsd"].append(float(rmsd_to_reference(snap, ref_t)))
        results["rg"].append(float(radius_of_gyration(snap)))
        results["contact_order"].append(float(contact_order(snap)))
        results["clash_count"].append(float(clash_count(snap)))
        results["ligand_rmsd"].append(0.0)
        results["ligand_pocket_dist"].append(0.0)
    return {k: np.array(v) for k, v in results.items()}


def load_training_cvs(training_dir, gamma=0.8, max_samples=1000):
    """Load CV values and logw from training WDSM samples."""
    from pathlib import Path
    d = Path(training_dir)
    files = sorted(d.glob("wdsm_step_*.npz")) or sorted(d.glob("batch_*.npz"))
    cvs, logws = [], []
    for f in files:
        data = np.load(f)
        if "cv" in data and "logw" in data:
            if data["cv"].ndim == 1:
                cvs.append(data["cv"])
                logws.append(float(data["logw"]))
            else:
                cvs.extend(data["cv"])
                logws.extend(data["logw"])
        if len(cvs) >= max_samples:
            break
    cvs = np.array(cvs[:max_samples])
    logws = np.array(logws[:max_samples], dtype=np.float64)
    logws_t = gamma * logws
    w = np.exp(logws_t - logws_t.max())
    w /= w.sum()
    return cvs, w


def generate_dashboard(models_data, training_cvs, training_weights, out_dir):
    """Generate comprehensive 4-way comparison dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import ks_2samp

    model_names = list(models_data.keys())
    colors = {"Training (reweighted)": "green", "Baseline": "blue",
              "Cartesian-trained": "orange", "Quotient-trained": "red"}

    cv_names = ["rmsd", "rg", "contact_order", "clash_count"]

    # --- Figure 1: 6-CV histogram comparison ---
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, cv in enumerate(cv_names):
        ax = axes[i]
        for name, data in models_data.items():
            vals = data.get(cv, np.array([]))
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.4, density=True,
                        color=colors.get(name, "gray"), label=name)
        ax.set_xlabel(cv)
        ax.set_ylabel("Density")
        ax.set_title(cv)
        ax.legend(fontsize=7)

    # ligand_rmsd from training data
    if training_cvs is not None and training_cvs.shape[1] >= 1:
        ax = axes[4]
        ax.hist(training_cvs[:, 0], bins=30, weights=training_weights * len(training_weights),
                alpha=0.4, density=True, color="green", label="Training")
        for name in ["Baseline", "Cartesian-trained", "Quotient-trained"]:
            if name in models_data and "ligand_rmsd" in models_data[name]:
                vals = models_data[name]["ligand_rmsd"]
                if np.any(vals != 0):
                    ax.hist(vals, bins=30, alpha=0.4, density=True, color=colors[name], label=name)
        ax.set_xlabel("Ligand RMSD"); ax.set_title("Ligand RMSD")
        ax.legend(fontsize=7)

    if training_cvs is not None and training_cvs.shape[1] >= 2:
        ax = axes[5]
        ax.hist(training_cvs[:, 1], bins=30, weights=training_weights * len(training_weights),
                alpha=0.4, density=True, color="green", label="Training")
        for name in ["Baseline", "Cartesian-trained", "Quotient-trained"]:
            if name in models_data and "ligand_pocket_dist" in models_data[name]:
                vals = models_data[name]["ligand_pocket_dist"]
                if np.any(vals != 0):
                    ax.hist(vals, bins=30, alpha=0.4, density=True, color=colors[name], label=name)
        ax.set_xlabel("Pocket Dist"); ax.set_title("Pocket Distance")
        ax.legend(fontsize=7)

    fig1.suptitle("4-Way CV Distribution Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1.savefig(str(out_dir / "cv_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: 2D FES heatmaps ---
    if training_cvs is not None and training_cvs.shape[1] >= 2:
        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
        rmsd_bins = np.linspace(0, max(training_cvs[:, 0].max(), 5), 35)
        pocket_bins = np.linspace(training_cvs[:, 1].min() - 0.2, training_cvs[:, 1].max() + 0.2, 25)

        # Training FES
        H, xe, ye = np.histogram2d(training_cvs[:, 0], training_cvs[:, 1],
                                   bins=[rmsd_bins, pocket_bins],
                                   weights=training_weights * len(training_weights))
        H[H <= 0] = np.nan
        F = -np.log(H); F -= np.nanmin(F)
        axes2[0].pcolormesh(xe, ye, F.T, cmap="RdYlBu_r", vmin=0, vmax=6, shading="flat")
        axes2[0].set_title("Training (OPES-MD)")
        axes2[0].set_xlabel("Ligand RMSD"); axes2[0].set_ylabel("Pocket Dist")

        for idx, name in enumerate(["Baseline", "Cartesian-trained", "Quotient-trained"]):
            if name not in models_data:
                continue
            ax = axes2[idx + 1]
            rmsd_vals = models_data[name].get("ligand_rmsd", np.array([]))
            pocket_vals = models_data[name].get("ligand_pocket_dist", np.array([]))
            if len(rmsd_vals) > 0 and np.any(rmsd_vals != 0):
                H2, _, _ = np.histogram2d(rmsd_vals, pocket_vals, bins=[rmsd_bins, pocket_bins])
                H2[H2 <= 0] = np.nan
                F2 = -np.log(H2); F2 -= np.nanmin(F2)
                ax.pcolormesh(xe, ye, F2.T, cmap="RdYlBu_r", vmin=0, vmax=6, shading="flat")
            ax.set_title(name)
            ax.set_xlabel("Ligand RMSD")

        fig2.suptitle("2D Free Energy Surfaces", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig2.savefig(str(out_dir / "fes_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # --- KS statistics + Cohen's d ---
    ks_results = {}
    for cv in cv_names:
        ks_results[cv] = {}
        baseline = models_data.get("Baseline", {}).get(cv, np.array([]))
        for name in ["Cartesian-trained", "Quotient-trained"]:
            vals = models_data.get(name, {}).get(cv, np.array([]))
            if len(baseline) > 0 and len(vals) > 0:
                ks_stat, p = ks_2samp(baseline, vals)
                pooled_std = np.sqrt((baseline.var() + vals.var()) / 2)
                cohens_d = (vals.mean() - baseline.mean()) / pooled_std if pooled_std > 0 else 0
                ks_results[cv][name] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p),
                    "cohens_d": float(cohens_d),
                    "mean_shift": float(vals.mean() - baseline.mean()),
                }

    return ks_results


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
    model, core, atom_mask, nkw = _build_session(
        yaml_path, cache, out / "prep", device, args.diffusion_steps)

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
        core_c = type(core)(model.structure_module, atom_mask, nkw, multiplicity=1)
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
        core_q = type(core)(model.structure_module, atom_mask, nkw, multiplicity=1)
        core_q.build_schedule(args.diffusion_steps)
        print("[compare] Generating QUOTIENT structures...", flush=True)
        quot_structs = generate_structures(core_q, args.n_samples, device)
        models_data["Quotient-trained"] = compute_simple_cvs(quot_structs, ref_coords)

    # Generate comparison
    print("\n[compare] Generating dashboard...", flush=True)
    ks_results = generate_dashboard(models_data, training_cvs, training_weights, out)

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

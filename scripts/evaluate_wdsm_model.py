#!/usr/bin/env python3
"""Evaluate fine-tuned WDSM model vs baseline Boltz-2.

Generates N structures from each model, computes CVs, and compares
distributions with KS tests and visualization.

Example::

    python scripts/evaluate_wdsm_model.py \
        --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \
        --finetuned-checkpoint /mnt/shared/.../wdsm_training/boltz2_wdsm_final.pt \
        --out /mnt/shared/.../evaluation \
        --n-samples 500 \
        --diffusion-steps 32
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


def _build_session(yaml_path, cache, prep_dir, device, diffusion_steps, recycling_steps):
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
    process_inputs(
        data=data_list, out_dir=boltz_run_dir, ccd_path=cache / "ccd.pkl",
        mol_dir=mol_dir, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", use_msa_server=False,
        boltz2=True, preprocessing_threads=1,
    )

    manifest = Manifest.load(boltz_run_dir / "processed" / "manifest.json")
    processed_dir = boltz_run_dir / "processed"
    dm = Boltz2InferenceDataModule(
        manifest=manifest, target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa", mol_dir=mol_dir, num_workers=0,
        constraints_dir=processed_dir / "constraints" if (processed_dir / "constraints").exists() else None,
        template_dir=processed_dir / "templates" if (processed_dir / "templates").exists() else None,
        extra_mols_dir=processed_dir / "mols" if (processed_dir / "mols").exists() else None,
    )
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    batch = dm.transfer_batch_to_device(batch, device, dataloader_idx=0)

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False

    ckpt = cache / "boltz2_conf.ckpt"
    predict_args = {
        "recycling_steps": recycling_steps, "sampling_steps": diffusion_steps,
        "diffusion_samples": 1, "max_parallel_samples": None,
        "write_confidence_summary": True, "write_full_pae": False, "write_full_pde": False,
    }
    model = Boltz2.load_from_checkpoint(
        str(ckpt), strict=True, predict_args=predict_args, map_location="cpu",
        diffusion_process_args=asdict(diffusion_params), ema=False, use_kernels=False,
        pairformer_args=asdict(pairformer_args), msa_args=asdict(msa_args),
        steering_args=asdict(steering),
    )
    model.to(device)

    atom_mask, network_kwargs = boltz2_trunk_to_network_kwargs(model, batch, recycling_steps=recycling_steps)
    for k, v in list(network_kwargs.items()):
        if hasattr(v, "to"):
            network_kwargs[k] = v.to(device)
    if isinstance(network_kwargs.get("feats"), dict):
        network_kwargs["feats"] = {fk: fv.to(device) if hasattr(fv, "to") else fv
                                   for fk, fv in network_kwargs["feats"].items()}

    core = BoltzSamplerCore(model.structure_module, atom_mask, network_kwargs, multiplicity=1)
    core.build_schedule(diffusion_steps)

    topo_npz = sorted((processed_dir / "structures").glob("*.npz"))[0]
    return model, core, atom_mask, network_kwargs, topo_npz


def generate_structures(core, n_samples, device):
    """Generate N structures from noise using the diffusion core."""
    from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy, BoltzSnapshot
    from genai_tps.backends.boltz.bridge import snapshot_from_gpu

    structures = []
    for i in range(n_samples):
        torch.manual_seed(i)
        x = core.sample_initial_noise()
        for step in range(core.num_sampling_steps):
            x, eps, rr, tr, meta = core.single_forward_step(x, step)
        structures.append(x[0].detach().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples}", flush=True)
    return np.array(structures)


def compute_cvs(structures, reference_coords, atom_mask_np, topo_npz):
    """Compute CVs on an array of structures."""
    from genai_tps.backends.boltz.collective_variables import (
        rmsd_to_reference, radius_of_gyration, contact_order, clash_count,
        PoseCVIndexer, ligand_pose_rmsd, ligand_pocket_distance,
    )
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot
    from genai_tps.analysis.boltz_npz_export import load_topo

    ref_t = torch.tensor(reference_coords, dtype=torch.float32)
    mask_t = torch.tensor(atom_mask_np[:1], dtype=torch.float32) if atom_mask_np is not None else None

    structure, n_struct = load_topo(Path(topo_npz))
    n_s = int(n_struct)
    ref_np = reference_coords[:n_s].astype(np.float32)
    indexer = PoseCVIndexer(structure, ref_np, pocket_radius=6.0)

    results = {"rmsd": [], "rg": [], "contact_order": [], "clash_count": [],
               "ligand_rmsd": [], "ligand_pocket_dist": []}

    for i, coords in enumerate(structures):
        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(coords_t, step_index=0, defer_numpy_coords=True)

        results["rmsd"].append(float(rmsd_to_reference(snap, ref_t, mask_t)))
        results["rg"].append(float(radius_of_gyration(snap, mask_t)))
        results["contact_order"].append(float(contact_order(snap)))
        results["clash_count"].append(float(clash_count(snap)))

        try:
            results["ligand_rmsd"].append(float(ligand_pose_rmsd(snap, indexer)))
            results["ligand_pocket_dist"].append(float(ligand_pocket_distance(snap, indexer)))
        except Exception:
            results["ligand_rmsd"].append(float("nan"))
            results["ligand_pocket_dist"].append(float("nan"))

        if (i + 1) % 100 == 0:
            print(f"  CVs computed for {i+1}/{len(structures)}", flush=True)

    return {k: np.array(v) for k, v in results.items()}


def generate_dashboard(baseline_cvs, finetuned_cvs, output_dir, n_baseline, n_finetuned):
    """Generate comparison dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    cv_names = sorted(set(baseline_cvs.keys()) & set(finetuned_cvs.keys()))
    n_cvs = len(cv_names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ks_results = {}
    for i, cv in enumerate(cv_names[:6]):
        ax = axes[i]
        b = baseline_cvs[cv]
        f = finetuned_cvs[cv]
        b = b[np.isfinite(b)]
        f = f[np.isfinite(f)]

        if len(b) > 0 and len(f) > 0:
            ks_stat, p_val = ks_2samp(b, f)
            ks_results[cv] = {"ks_statistic": float(ks_stat), "p_value": float(p_val)}

            lo = min(b.min(), f.min())
            hi = max(b.max(), f.max())
            bins = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 40)

            ax.hist(b, bins=bins, alpha=0.6, color="steelblue", label=f"Baseline (n={len(b)})", density=True)
            ax.hist(f, bins=bins, alpha=0.6, color="coral", label=f"Fine-tuned (n={len(f)})", density=True)
            sig = "*" if p_val < 0.05 else ""
            ax.set_title(f"{cv}\nKS={ks_stat:.3f}, p={p_val:.3g}{sig}")
            ax.legend(fontsize=8)
        ax.set_xlabel(cv)
        ax.set_ylabel("Density")

    fig.suptitle(
        f"WDSM Fine-tuned vs Baseline — MEK1+FZC\n"
        f"Baseline: {n_baseline} samples | Fine-tuned: {n_finetuned} samples",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / "evaluation_dashboard.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Dashboard saved: {path}")
    return ks_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDSM fine-tuned vs baseline Boltz-2.")
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    yaml_path = args.yaml or (_REPO_ROOT / "inputs" / "tps_diagnostic" / "case1_mek1_fzc_novel.yaml")
    cache = Path(args.cache).expanduser() if args.cache else Path.home() / ".boltz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  WDSM Evaluation: Baseline vs Fine-tuned")
    print(f"  Samples per model: {args.n_samples}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"{'='*60}\n")

    print("[eval] Building Boltz session (baseline)...", flush=True)
    model, core, atom_mask, nck, topo_npz = _build_session(
        yaml_path, cache, out / "baseline_prep", device,
        args.diffusion_steps, args.recycling_steps,
    )

    ref_x = core.sample_initial_noise()
    for step in range(core.num_sampling_steps):
        ref_x, *_ = core.single_forward_step(ref_x, step)
    reference_coords = ref_x[0].detach().cpu().numpy()
    atom_mask_np = atom_mask.cpu().numpy()

    print("\n[eval] Generating baseline structures...", flush=True)
    baseline_structs = generate_structures(core, args.n_samples, device)
    np.savez(out / "baseline_structures.npz", coords=baseline_structs)

    print("[eval] Computing baseline CVs...", flush=True)
    baseline_cvs = compute_cvs(baseline_structs, reference_coords, atom_mask_np, topo_npz)

    print("\n[eval] Loading fine-tuned checkpoint...", flush=True)
    state_dict = torch.load(str(args.finetuned_checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    core_ft = type(core)(model.structure_module, atom_mask, nck, multiplicity=1)
    core_ft.build_schedule(args.diffusion_steps)

    print("[eval] Generating fine-tuned structures...", flush=True)
    finetuned_structs = generate_structures(core_ft, args.n_samples, device)
    np.savez(out / "finetuned_structures.npz", coords=finetuned_structs)

    print("[eval] Computing fine-tuned CVs...", flush=True)
    finetuned_cvs = compute_cvs(finetuned_structs, reference_coords, atom_mask_np, topo_npz)

    print("\n[eval] Generating comparison dashboard...", flush=True)
    ks_results = generate_dashboard(baseline_cvs, finetuned_cvs, out, args.n_samples, args.n_samples)

    report = {
        "n_baseline": args.n_samples,
        "n_finetuned": args.n_samples,
        "finetuned_checkpoint": str(args.finetuned_checkpoint),
        "diffusion_steps": args.diffusion_steps,
        "ks_tests": ks_results,
        "baseline_summary": {k: {"mean": float(v[np.isfinite(v)].mean()), "std": float(v[np.isfinite(v)].std()),
                                  "min": float(v[np.isfinite(v)].min()), "max": float(v[np.isfinite(v)].max())}
                             for k, v in baseline_cvs.items() if np.any(np.isfinite(v))},
        "finetuned_summary": {k: {"mean": float(v[np.isfinite(v)].mean()), "std": float(v[np.isfinite(v)].std()),
                                   "min": float(v[np.isfinite(v)].min()), "max": float(v[np.isfinite(v)].max())}
                              for k, v in finetuned_cvs.items() if np.any(np.isfinite(v))},
    }
    with open(out / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete")
    print(f"  Report: {out / 'evaluation_report.json'}")
    print(f"  Dashboard: {out / 'evaluation_dashboard.png'}")
    for cv, ks in ks_results.items():
        sig = "***" if ks["p_value"] < 0.001 else "**" if ks["p_value"] < 0.01 else "*" if ks["p_value"] < 0.05 else ""
        print(f"  {cv}: KS={ks['ks_statistic']:.3f} p={ks['p_value']:.3g} {sig}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

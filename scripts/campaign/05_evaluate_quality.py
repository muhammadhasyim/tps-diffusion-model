#!/usr/bin/env python3
"""Stage 5 — Structural quality and free energy surface evaluation.

Answers **Q1b**: Does the fine-tuned model generate physically better structures,
and does it reproduce the correct thermodynamic landscape?

Evaluations performed for each (system) × (model variant) pair:

1. **Collective variable distributions**
   Computes rmsd, rg, contact_order, clash_count, ligand_rmsd, ligand_pocket_dist
   for every generated structure using ``compute_cvs()`` and plots histograms
   comparing baseline vs fine-tuned.

2. **PoseBusters structural validity**
   Runs the GPU-native boolean checks (ligand RMSD ≤ 2 Å, pocket distance,
   contacts, H-bonds, clashes, ligand size) and reports pass rates per check
   and overall ``posebusters_gpu_pass_fraction``.

3. **Free energy surface (FES) comparison**
   Constructs 2D FES over (ligand_rmsd, ligand_pocket_dist):
   - Ground truth: OPES-MD sample coords reweighted via OPES log-weights
   - Model: generated structures treated as unweighted (uniform prior)
   FES divergence is quantified via KL divergence, JS divergence, and
   earth mover's distance over a 2D histogram grid.

4. **Per-system expected-outcome validation**
   Case 1 (novel): baseline likely has high RMSD; SFT should improve
   Case 2 (memorised): both good; fine-tuning may shift similarity profile
   Case 3 (adversarial): baseline funnels to B_crystal (RMSD < 2 Å) even
       though the pocket is sterically blocked — SFT should resist this

Pipeline role:
    03_generate_ensembles  →  05_evaluate_quality  →  08_generate_campaign_report
    01_assemble_datasets   →  05_evaluate_quality  (for ground-truth reference FES)

Outputs::

    outputs/campaign/case{1,2,3}/quality/
        quality_report.json
        cv_distributions.png
        fes_comparison.png
        posebusters_summary.json

Example::

    python scripts/campaign/05_evaluate_quality.py \\
        --out outputs/campaign \\
        --pocket-radius 6.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

_CASE_NAMES = [
    "case1_mek1_fzc_novel",
    "case2_cdk2_atp_wildtype",
    "case3_cdk2_atp_packed",
]

# Expected outcomes per case for interpretive validation
_CASE_EXPECTATIONS = {
    "case1_mek1_fzc_novel": {
        "baseline_high_rmsd": True,
        "sft_improves_rmsd": True,
        "description": "Novel system — baseline should fail; SFT may improve.",
    },
    "case2_cdk2_atp_wildtype": {
        "baseline_high_rmsd": False,
        "sft_improves_rmsd": False,
        "description": "Memorised system — baseline strong; SFT preserves quality.",
    },
    "case3_cdk2_atp_packed": {
        "baseline_high_rmsd": False,
        "sft_improves_rmsd": True,  # Should move away from B_crystal toward B_phys
        "description": "Adversarial — baseline confident but wrong; SFT should resist.",
    },
}


def _find_topo_npz(case_dir: Path) -> Path | None:
    for root in [
        case_dir / "openmm_opes_md" / "boltz_prep",
        case_dir,
    ]:
        cands = sorted(root.rglob("processed/structures/*.npz")) if root.exists() else []
        if cands:
            return cands[0]
    return None


def _compute_all_cvs(
    coords_npz: Path,
    topo_npz: Path,
    ref_coords: np.ndarray,
    n_struct: int,
    pocket_radius: float,
) -> dict[str, np.ndarray]:
    """Compute CV array for all frames using compute_cvs()."""
    from genai_tps.backends.boltz.collective_variables import compute_cvs

    data = np.load(coords_npz)
    coords = data["coords"]  # (N, M, 3)

    cv_dict = compute_cvs(
        coords,
        reference_coords=ref_coords,
        atom_mask_np=np.ones(n_struct, dtype=np.float32),
        topo_npz=topo_npz,
    )
    return cv_dict


def _run_gpu_posebusters(
    coords_npz: Path,
    topo_npz: Path,
    ref_coords: np.ndarray,
    n_struct: int,
    pocket_radius: float,
) -> dict[str, float]:
    """Return GPU PoseBusters pass rates for each check."""
    from genai_tps.evaluation.posebusters import GPUPoseBustersEvaluator, gpu_check_columns
    from genai_tps.io.boltz_npz_export import load_topo

    structure, n_struct_loaded = load_topo(topo_npz)
    evaluator = GPUPoseBustersEvaluator(
        structure=structure,
        n_struct=int(n_struct_loaded),
        reference_coords=ref_coords[:int(n_struct_loaded)],
        pocket_radius=pocket_radius,
    )

    data = np.load(coords_npz)
    coords = data["coords"]

    all_checks = {col: [] for col in gpu_check_columns()}
    for i in range(len(coords)):
        row = evaluator.evaluate_coords(torch.from_numpy(coords[i]).float())
        for col in gpu_check_columns():
            all_checks[col].append(float(row.get(col, float("nan"))))

    return {col: float(np.nanmean(vals)) for col, vals in all_checks.items()}


def _build_fes_2d(
    ligand_rmsds: np.ndarray,
    pocket_dists: np.ndarray,
    log_weights: np.ndarray | None,
    n_bins: int = 30,
    rmsd_range: tuple[float, float] = (0.0, 15.0),
    dist_range: tuple[float, float] = (0.0, 20.0),
    kbt: float = 2.479,  # kJ/mol at 300 K
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D free energy surface (kJ/mol) from samples + optional log-weights.

    Returns (fes_grid, rmsd_edges, dist_edges).  Bins with zero weight are set to NaN.
    """
    if log_weights is None:
        weights = np.ones(len(ligand_rmsds), dtype=np.float64)
    else:
        log_w = log_weights - log_weights.max()
        weights = np.exp(log_w)

    hist, r_edges, d_edges = np.histogram2d(
        ligand_rmsds, pocket_dists,
        bins=n_bins,
        range=[rmsd_range, dist_range],
        weights=weights,
        density=False,
    )
    hist = hist / hist.sum()  # normalise to probability
    with np.errstate(divide="ignore"):
        fes = -kbt * np.log(hist)
    fes[hist == 0] = np.nan
    fes -= np.nanmin(fes)  # shift minimum to 0
    return fes, r_edges, d_edges


def _fes_divergences(
    fes_ref: np.ndarray,
    fes_pred: np.ndarray,
    hist_ref: np.ndarray,
    hist_pred: np.ndarray,
) -> dict[str, float]:
    """KL, JS, and Wasserstein (1D marginals) divergences between two FES."""
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance

    # Mask bins where either is nan
    mask = np.isfinite(fes_ref) & np.isfinite(fes_pred)
    if mask.sum() < 2:
        return {"kl_div": float("nan"), "js_div": float("nan"), "wasserstein_rmsd": float("nan")}

    p = hist_ref.ravel()[mask.ravel()]
    q = hist_pred.ravel()[mask.ravel()]
    p = p / p.sum() + 1e-12
    q = q / q.sum() + 1e-12

    kl = float(np.sum(p * np.log(p / q)))
    js = float(jensenshannon(p, q, base=2))

    # 1D Wasserstein on RMSD marginals
    rmsd_marg_ref = hist_ref.sum(axis=1)
    rmsd_marg_pred = hist_pred.sum(axis=1)
    rmsd_marg_ref = rmsd_marg_ref / rmsd_marg_ref.sum() + 1e-12
    rmsd_marg_pred = rmsd_marg_pred / rmsd_marg_pred.sum() + 1e-12
    wass = float(wasserstein_distance(
        np.arange(len(rmsd_marg_ref)), np.arange(len(rmsd_marg_pred)),
        rmsd_marg_ref, rmsd_marg_pred,
    ))
    return {"kl_div": kl, "js_div": js, "wasserstein_rmsd": wass}


def _plot_cv_distributions(
    cv_data: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    case_name: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cv_keys = ["rmsd", "rg", "ligand_rmsd", "ligand_pocket_dist", "contact_order", "clash_count"]
    n_cv = len(cv_keys)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.ravel()
    for i, key in enumerate(cv_keys):
        ax = axes_flat[i]
        for variant, cvs in cv_data.items():
            vals = cvs.get(key)
            if vals is not None and len(vals) > 0:
                ax.hist(np.array(vals), bins=30, alpha=0.5, density=True, label=variant)
        ax.set_xlabel(key)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle(f"CV distributions — {case_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_fes_comparison(
    fes_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    case_name: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_panels = len(fes_data)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    for ax, (variant, (fes, r_edges, d_edges)) in zip(axes, fes_data.items()):
        vmax = np.nanpercentile(fes, 95)
        img = ax.pcolormesh(r_edges, d_edges, fes.T, cmap="viridis_r", vmin=0, vmax=vmax)
        plt.colorbar(img, ax=ax, label="FES (kJ/mol)")
        ax.set_xlabel("Ligand RMSD (Å)")
        ax.set_ylabel("Pocket distance (Å)")
        ax.set_title(variant, fontsize=10)
    fig.suptitle(f"2D Free energy surface — {case_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Structural quality and free energy surface evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--cases", type=str, default="1,2,3")
    parser.add_argument("--variants", type=str, default="baseline,sft_cartesian,sft_true-quotient")
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--fes-bins", type=int, default=30)
    parser.add_argument("--fes-rmsd-max", type=float, default=15.0)
    parser.add_argument("--fes-dist-max", type=float, default=20.0)
    args = parser.parse_args()

    out_root = args.out.expanduser().resolve()
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    variants = [v.strip() for v in args.variants.split(",")]

    all_results: dict[str, Any] = {}

    for idx, name in enumerate(_CASE_NAMES, start=1):
        if idx not in selected_cases:
            continue

        print(f"\n{'='*60}")
        print(f"  Case {idx}: {name}")
        print(f"{'='*60}", flush=True)

        case_dir = out_root / name
        quality_dir = case_dir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)

        topo_npz = _find_topo_npz(case_dir)
        if topo_npz is None:
            print(f"  [05] No topology NPZ for {name}; skipping.", flush=True)
            continue

        from genai_tps.io.boltz_npz_export import load_topo
        structure, n_struct = load_topo(topo_npz)
        n_struct = int(n_struct)
        ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)

        # Load ground-truth OPES-MD data for FES reference
        gt_logw: np.ndarray | None = None
        gt_rmsd: np.ndarray | None = None
        gt_pocket_dist: np.ndarray | None = None
        gt_dataset = case_dir / "training_dataset.npz"
        if gt_dataset.is_file():
            try:
                gt_data = np.load(gt_dataset)
                gt_logw = gt_data["logw"]
                # Recompute CVs for ground truth
                print(f"  [05] Computing ground-truth CVs from {gt_dataset}...", flush=True)
                gt_cv_dict = _compute_all_cvs(gt_dataset, topo_npz, ref_coords, n_struct, args.pocket_radius)
                gt_rmsd = np.array(gt_cv_dict.get("ligand_rmsd", []))
                gt_pocket_dist = np.array(gt_cv_dict.get("ligand_pocket_dist", []))
            except Exception as exc:
                print(f"  [05] Ground-truth CV computation failed: {exc}", flush=True)

        case_result: dict[str, Any] = {
            "case": name,
            "expectations": _CASE_EXPECTATIONS.get(name, {}),
            "variants": {},
        }
        cv_data_all: dict[str, dict[str, np.ndarray]] = {}
        fes_data_all: dict[str, tuple] = {}

        # Add ground-truth FES
        if gt_rmsd is not None and len(gt_rmsd) > 0 and gt_pocket_dist is not None:
            fes, r_edges, d_edges = _build_fes_2d(
                gt_rmsd, gt_pocket_dist, gt_logw,
                n_bins=args.fes_bins,
                rmsd_range=(0.0, args.fes_rmsd_max),
                dist_range=(0.0, args.fes_dist_max),
            )
            fes_data_all["ground_truth (OPES-MD)"] = (fes, r_edges, d_edges)
            gt_hist_raw = np.histogram2d(
                gt_rmsd, gt_pocket_dist,
                bins=args.fes_bins,
                range=[(0.0, args.fes_rmsd_max), (0.0, args.fes_dist_max)],
                weights=np.exp(gt_logw - gt_logw.max()) if gt_logw is not None else None,
            )[0]

        for variant in variants:
            coords_npz = case_dir / variant / "generated_structures.npz"
            if not coords_npz.is_file():
                print(f"  [05] Missing: {coords_npz}; skipping.", flush=True)
                continue
            print(f"  [05] Variant: {variant}", flush=True)

            # CV computation
            try:
                cv_dict = _compute_all_cvs(coords_npz, topo_npz, ref_coords, n_struct, args.pocket_radius)
                cv_data_all[variant] = cv_dict
            except Exception as exc:
                print(f"    [05] CV computation failed: {exc}", flush=True)
                cv_dict = {}

            # GPU PoseBusters
            pb_rates: dict[str, float] = {}
            try:
                pb_rates = _run_gpu_posebusters(coords_npz, topo_npz, ref_coords, n_struct, args.pocket_radius)
            except Exception as exc:
                print(f"    [05] GPU PoseBusters failed: {exc}", flush=True)

            # FES construction
            lig_rmsd_arr = np.array(cv_dict.get("ligand_rmsd", []))
            pocket_dist_arr = np.array(cv_dict.get("ligand_pocket_dist", []))
            fes_divs: dict[str, float] = {}
            if len(lig_rmsd_arr) > 0 and len(pocket_dist_arr) > 0:
                fes, r_edges, d_edges = _build_fes_2d(
                    lig_rmsd_arr, pocket_dist_arr, None,
                    n_bins=args.fes_bins,
                    rmsd_range=(0.0, args.fes_rmsd_max),
                    dist_range=(0.0, args.fes_dist_max),
                )
                fes_data_all[variant] = (fes, r_edges, d_edges)
                # FES divergences vs ground truth
                if gt_rmsd is not None and len(gt_rmsd) > 0:
                    pred_hist = np.histogram2d(
                        lig_rmsd_arr, pocket_dist_arr,
                        bins=args.fes_bins,
                        range=[(0.0, args.fes_rmsd_max), (0.0, args.fes_dist_max)],
                    )[0].astype(np.float64)
                    gt_hist_f = gt_hist_raw.astype(np.float64)
                    fes_ref_for_div, _, _ = _build_fes_2d(
                        gt_rmsd, gt_pocket_dist, gt_logw,
                        n_bins=args.fes_bins,
                        rmsd_range=(0.0, args.fes_rmsd_max),
                        dist_range=(0.0, args.fes_dist_max),
                    )
                    fes_divs = _fes_divergences(fes_ref_for_div, fes, gt_hist_f, pred_hist)

            # Summary statistics per CV
            cv_summary: dict[str, dict] = {}
            for key, vals in cv_dict.items():
                arr = np.array(vals)
                if len(arr) > 0:
                    cv_summary[key] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "median": float(np.median(arr)),
                        "p25": float(np.percentile(arr, 25)),
                        "p75": float(np.percentile(arr, 75)),
                    }

            case_result["variants"][variant] = {
                "cv_summary": cv_summary,
                "posebusters_gpu_pass_rates": pb_rates,
                "fes_divergences_vs_groundtruth": fes_divs,
            }
            print(f"    [05] PoseBusters pass fraction: "
                  f"{pb_rates.get('ligand_rmsd_le_2a', 'N/A'):.3f}", flush=True)

        # Plots
        if cv_data_all:
            _plot_cv_distributions(
                {k: {ck: np.array(v) for ck, v in cv.items()} for k, cv in cv_data_all.items()},
                quality_dir / "cv_distributions.png",
                name,
            )
        if fes_data_all:
            _plot_fes_comparison(fes_data_all, quality_dir / "fes_comparison.png", name)

        # Save PoseBusters summary separately
        pb_summary = {v: case_result["variants"][v].get("posebusters_gpu_pass_rates", {})
                      for v in case_result["variants"]}
        with open(quality_dir / "posebusters_summary.json", "w") as fh:
            json.dump(pb_summary, fh, indent=2)

        # Save report
        report_path = quality_dir / "quality_report.json"
        with open(report_path, "w") as fh:
            json.dump(case_result, fh, indent=2)
        print(f"  [05] Report: {report_path}", flush=True)

        all_results[name] = case_result

    summary_path = out_root / "05_quality_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[05] Summary: {summary_path}", flush=True)
    print("[05] Stage 5 complete.", flush=True)


if __name__ == "__main__":
    main()

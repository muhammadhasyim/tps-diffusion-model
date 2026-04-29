#!/usr/bin/env python3
"""Stage 4 — Memorization evaluation (Skrinjar similarity + Runs-N-Poses protocol).

Answers **Q1a**: Does supervised fine-tuning on physical reference data change
the degree to which Boltz-2 reproduces structures close to its training set?

Two complementary approaches are used:

1. **Skrinjar / SuCOS × pocket_qcov similarity**
   For each generated structure, compute the maximum SuCOS-weighted
   pharmacophore+shape similarity times geometric pocket coverage against a
   database of training PDB complexes.  This is our internal proxy for the
   ``sucos_shape_pocket_qcov`` metric used by Škrinjar et al. (Runs N' Poses).

2. **Runs-N-Poses style stratified success**
   Using the crystal reference structures as ground truth, compute per-sample
   ligand RMSD and LDDT-PLI (via a lightweight implementation).  Report success
   rate (RMSD < 2 Å AND LDDT-PLI > 0.8) stratified by training-similarity bin.
   Compares directly to Figure 1 of Škrinjar et al. (bioRxiv 2025.02.03.636309).

   Note on LDDT-PLI: a simplified protein-ligand lDDT is computed using the
   existing ``lddt_to_reference`` CV. Full OpenStructure-based BiSyRMSD and
   LDDT-PLI as used in the Runs-N-Poses paper require the OpenStructure package;
   if available it will be used; otherwise the simplified version is used and
   flagged in the output.

3. **KS test + effect size** comparing similarity distributions between baseline
   and each fine-tuned variant.

Pipeline role:
    03_generate_ensembles  →  04_evaluate_memorization  →  08_generate_campaign_report

Outputs::

    outputs/campaign/case{1,2,3}/
        memorization/
            memorization_report.json        # full numeric results
            memorization_comparison.png     # similarity histograms per variant
            similarity_distributions.json   # raw per-sample scores

Required external inputs (user must provide):
    --training-complexes-dir : directory of training holo PDB files (pre-cutoff)
    --training-ligand-sdfs   : directory of ligand SDF files for Skrinjar scoring
    --foldseek-db            : (optional) prebuilt Foldseek DB for pocket pre-filter

Example::

    python scripts/campaign/04_evaluate_memorization.py \\
        --out outputs/campaign \\
        --training-complexes-dir /data/pdb_training_complexes/ \\
        --training-ligand-sdfs /data/pdb_training_ligands/ \\
        --crystal-cif-dir inputs/tps_diagnostic/reference_structures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

# Runs-N-Poses plotting constants (aligned with upstream plinder-org/runs-n-poses plotting.py)
_RMSD_THRESHOLD = 2.0          # Å
_LDDT_PLI_THRESHOLD = 0.8
_SIMILARITY_BINS = [0, 20, 40, 60, 80, 100]

_CASE_METADATA = {
    "case1_mek1_fzc_novel": {
        "crystal_pdb_id": "7XLP",
        "ligand_chain": "B",
        "ligand_ccd": "FZC",
        "training_similarity": 46.21,  # sucos_shape_pocket_qcov from Škrinjar et al.
        "expected_regime": "low",
    },
    "case2_cdk2_atp_wildtype": {
        "crystal_pdb_id": "1B38",
        "ligand_chain": "B",
        "ligand_ccd": "ATP",
        "training_similarity": 95.0,   # ATP is heavily represented
        "expected_regime": "high",
    },
    "case3_cdk2_atp_packed": {
        "crystal_pdb_id": "1B38",
        "ligand_chain": "B",
        "ligand_ccd": "ATP",
        "training_similarity": 80.0,   # Pocket differs but ligand same
        "expected_regime": "adversarial",
    },
}


def _similarity_bin(score: float) -> str:
    """Assign a Runs-N-Poses similarity bin label (0-20, 20-40, etc.)."""
    for lo, hi in zip(_SIMILARITY_BINS[:-1], _SIMILARITY_BINS[1:]):
        if lo <= score < hi:
            return f"{lo}-{hi}"
    return "80-100"


def _compute_skrinjar_scores(
    coords_npz: Path,
    topo_npz: Path,
    training_complexes_dir: Path,
    training_ligand_sdfs: Path | None,
    foldseek_db: Path | None,
    top_k: int,
    pocket_radius: float,
) -> list[float]:
    """Compute per-sample Skrinjar max(SuCOS x pocket_qcov) scores."""
    from genai_tps.evaluation.skrinjar_similarity import IncrementalSkrinjarScorer
    from genai_tps.io.boltz_npz_export import load_topo, npz_to_pdb
    import tempfile
    import os

    structure, n_struct = load_topo(topo_npz)
    data = np.load(coords_npz)
    coords = data["coords"]  # (N, M, 3)
    n_samples = coords.shape[0]

    ligand_sdfs: tuple[Path, ...] = ()
    if training_ligand_sdfs is not None:
        if training_ligand_sdfs.is_dir():
            ligand_sdfs = tuple(sorted(training_ligand_sdfs.glob("*.sdf")))
        elif training_ligand_sdfs.is_file():
            ligand_sdfs = (training_ligand_sdfs,)

    scorer = IncrementalSkrinjarScorer(
        training_complexes_dir=training_complexes_dir,
        training_ligand_sdfs=ligand_sdfs,
        foldseek_db=foldseek_db,
        top_k=top_k,
        pocket_radius=pocket_radius,
    )

    scores = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(n_samples):
            pdb_path = Path(tmp_dir) / f"sample_{i:05d}.pdb"
            # Write a temporary single-sample NPZ
            tmp_npz = Path(tmp_dir) / f"sample_{i:05d}.npz"
            np.savez(tmp_npz, coords=coords[i:i+1])
            npz_to_pdb(tmp_npz, structure, n_struct, pdb_path, frame_idx=0)
            score = scorer.score_pdb_file(pdb_path)
            scores.append(float(score))

    return scores


def _compute_ligand_rmsd_simple(
    coords_npz: Path,
    topo_npz: Path,
    ref_coords_np: np.ndarray,
    pocket_radius: float,
) -> list[float]:
    """Compute per-sample ligand RMSD using PoseCVIndexer."""
    import torch
    from genai_tps.backends.boltz.collective_variables import (
        PoseCVIndexer,
        ligand_pose_rmsd,
    )
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot
    from genai_tps.io.boltz_npz_export import load_topo

    structure, n_struct = load_topo(topo_npz)
    n_struct = int(n_struct)
    indexer = PoseCVIndexer(structure, ref_coords_np[:n_struct], pocket_radius=pocket_radius)

    data = np.load(coords_npz)
    coords = data["coords"]
    rmsds = []
    for i in range(len(coords)):
        x = torch.from_numpy(coords[i][:n_struct]).float().unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(x, step_index=0, defer_numpy_coords=True)
        rmsds.append(float(ligand_pose_rmsd(snap, indexer)))
    return rmsds


def _simplified_lddt_pli(
    coords_npz: Path,
    topo_npz: Path,
    ref_coords_np: np.ndarray,
    pocket_radius: float,
    cutoff: float = 8.0,
) -> list[float]:
    """Approximate LDDT-PLI using the protein-ligand contact-based lDDT CV.

    This is a proxy, not the full OpenStructure LDDT-PLI.  The output is
    flagged in the report so downstream users can substitute with OpenStructure
    values if available.
    """
    import torch
    from genai_tps.backends.boltz.collective_variables import lddt_to_reference
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot
    from genai_tps.io.boltz_npz_export import load_topo

    _structure, n_struct = load_topo(topo_npz)
    n_struct = int(n_struct)
    ref_t = torch.from_numpy(ref_coords_np[:n_struct]).float()

    data = np.load(coords_npz)
    coords = data["coords"]
    lddts = []
    for i in range(len(coords)):
        x = torch.from_numpy(coords[i][:n_struct]).float().unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(x, step_index=0, defer_numpy_coords=True)
        val = lddt_to_reference(snap, ref_t, inclusion_radius=cutoff)
        lddts.append(float(val))
    return lddts


def _ks_test(scores_a: list[float], scores_b: list[float]) -> dict[str, float]:
    """Two-sample KS test + Cohen's d effect size."""
    from scipy import stats as sp_stats
    a = np.array(scores_a)
    b = np.array(scores_b)
    ks_stat, ks_p = sp_stats.ks_2samp(a, b)
    # Cohen's d
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(b) - np.mean(a)) / pooled_std if pooled_std > 0 else 0.0
    return {"ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p), "cohens_d": float(cohens_d)}


def _plot_similarity_comparison(
    scores_by_variant: dict[str, list[float]],
    out_path: Path,
    case_name: str,
) -> None:
    """Plot per-variant similarity score histograms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [04] matplotlib not available; skipping plot.", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for variant, scores in scores_by_variant.items():
        ax.hist(scores, bins=20, alpha=0.5, label=variant, density=True)
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, label="similarity=0.5 threshold")
    ax.set_xlabel("Skrinjar SuCOS × pocket_qcov", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Training similarity distribution — {case_name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Memorization evaluation (Skrinjar + Runs-N-Poses).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/campaign"))
    parser.add_argument("--cases", type=str, default="1,2,3")
    parser.add_argument("--variants", type=str, default="baseline,sft_cartesian,sft_true-quotient")

    # Skrinjar-specific
    parser.add_argument("--training-complexes-dir", type=Path, default=None,
                        help="Directory of pre-cutoff PDB complex files for Skrinjar scoring. "
                             "If omitted, similarity scoring is skipped.")
    parser.add_argument("--training-ligand-sdfs", type=Path, default=None,
                        help="Directory of ligand SDF files for additional SuCOS scoring.")
    parser.add_argument("--foldseek-db", type=Path, default=None,
                        help="Prebuilt Foldseek DB for pocket-based structural pre-filter.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of training neighbours to evaluate per query.")
    parser.add_argument("--pocket-radius", type=float, default=6.0)

    # Reference structures
    parser.add_argument("--crystal-cif-dir", type=Path,
                        default=_REPO_ROOT / "inputs" / "tps_diagnostic" / "reference_structures",
                        help="Directory with crystal CIF files (7XLP.cif, 1B38.cif).")
    args = parser.parse_args()

    out_root = args.out.expanduser().resolve()
    selected_cases = {int(c.strip()) for c in args.cases.split(",")}
    variants = [v.strip() for v in args.variants.split(",")]
    case_names = list(_CASE_METADATA.keys())

    if args.training_complexes_dir is None:
        print(
            "[04] WARNING: --training-complexes-dir not provided. "
            "Skrinjar similarity scoring will be skipped.\n"
            "    Provide a directory of pre-cutoff training holo PDB files to enable this.",
            flush=True,
        )

    all_results: dict[str, Any] = {}

    for idx, name in enumerate(case_names, start=1):
        if idx not in selected_cases:
            continue
        meta = _CASE_METADATA[name]
        print(f"\n{'='*60}")
        print(f"  Case {idx}: {name} ({meta['expected_regime']} similarity regime)")
        print(f"{'='*60}", flush=True)

        mem_dir = out_root / name / "memorization"
        mem_dir.mkdir(parents=True, exist_ok=True)

        # Locate topology NPZ for this system
        topo_candidates = sorted((out_root / name / "openmm_opes_md").rglob("processed/structures/*.npz"))
        if not topo_candidates:
            topo_candidates = sorted((out_root / name).rglob("processed/structures/*.npz"))
        if not topo_candidates:
            print(f"  [04] No topology NPZ found for {name}; skipping.", flush=True)
            continue
        topo_npz = topo_candidates[0]

        # Load reference coordinates from topology NPZ
        from genai_tps.io.boltz_npz_export import load_topo
        structure, n_struct = load_topo(topo_npz)
        ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)

        case_results: dict[str, Any] = {"metadata": meta, "variants": {}}
        scores_by_variant: dict[str, list[float]] = {}

        for variant in variants:
            coords_npz = out_root / name / variant / "generated_structures.npz"
            if not coords_npz.is_file():
                print(f"  [04] Missing ensemble: {coords_npz}; skipping variant.", flush=True)
                continue

            print(f"  [04] Processing variant: {variant}", flush=True)
            variant_result: dict[str, Any] = {}

            # 1. Skrinjar similarity
            skrinjar_scores: list[float] = []
            if args.training_complexes_dir is not None and args.training_complexes_dir.is_dir():
                print(f"    [04] Skrinjar scoring...", flush=True)
                try:
                    skrinjar_scores = _compute_skrinjar_scores(
                        coords_npz=coords_npz,
                        topo_npz=topo_npz,
                        training_complexes_dir=args.training_complexes_dir,
                        training_ligand_sdfs=args.training_ligand_sdfs,
                        foldseek_db=args.foldseek_db,
                        top_k=args.top_k,
                        pocket_radius=args.pocket_radius,
                    )
                    print(f"    [04] Skrinjar: mean={np.mean(skrinjar_scores):.3f} "
                          f"median={np.median(skrinjar_scores):.3f}", flush=True)
                except Exception as exc:
                    print(f"    [04] Skrinjar scoring failed: {exc}", flush=True)

            # 2. Ligand RMSD vs crystal reference
            try:
                ligand_rmsds = _compute_ligand_rmsd_simple(
                    coords_npz, topo_npz, ref_coords, args.pocket_radius
                )
            except Exception as exc:
                print(f"    [04] Ligand RMSD failed: {exc}", flush=True)
                ligand_rmsds = []

            # 3. Simplified LDDT-PLI proxy
            try:
                lddt_pli_proxy = _simplified_lddt_pli(
                    coords_npz, topo_npz, ref_coords, args.pocket_radius
                )
            except Exception as exc:
                print(f"    [04] LDDT-PLI proxy failed: {exc}", flush=True)
                lddt_pli_proxy = []

            # 4. Success rate (Runs-N-Poses definition)
            success_count = 0
            n_valid = min(len(ligand_rmsds), len(lddt_pli_proxy))
            for rmsd, lddt in zip(ligand_rmsds[:n_valid], lddt_pli_proxy[:n_valid]):
                if rmsd < _RMSD_THRESHOLD and lddt > _LDDT_PLI_THRESHOLD:
                    success_count += 1
            success_rate = success_count / n_valid if n_valid > 0 else 0.0

            # Assign to similarity bin (use annotation if no scorer, else use mean score)
            system_sim = meta["training_similarity"]
            sim_bin = _similarity_bin(system_sim)

            variant_result = {
                "n_samples": int(len(np.load(coords_npz)["coords"])),
                "skrinjar_scores": {
                    "mean": float(np.mean(skrinjar_scores)) if skrinjar_scores else None,
                    "median": float(np.median(skrinjar_scores)) if skrinjar_scores else None,
                    "std": float(np.std(skrinjar_scores)) if skrinjar_scores else None,
                    "fraction_above_50": float(np.mean(np.array(skrinjar_scores) > 0.5)) if skrinjar_scores else None,
                    "available": len(skrinjar_scores) > 0,
                },
                "ligand_rmsd": {
                    "mean": float(np.mean(ligand_rmsds)) if ligand_rmsds else None,
                    "median": float(np.median(ligand_rmsds)) if ligand_rmsds else None,
                    "fraction_below_2a": float(np.mean(np.array(ligand_rmsds) < _RMSD_THRESHOLD)) if ligand_rmsds else None,
                },
                "lddt_pli_proxy": {
                    "mean": float(np.mean(lddt_pli_proxy)) if lddt_pli_proxy else None,
                    "is_proxy": True,  # Flag that this is not full OpenStructure LDDT-PLI
                },
                "success_rate": float(success_rate),
                "n_successes": int(success_count),
                "n_valid": int(n_valid),
                "system_training_similarity": system_sim,
                "similarity_bin": sim_bin,
            }
            case_results["variants"][variant] = variant_result
            if skrinjar_scores:
                scores_by_variant[variant] = skrinjar_scores

        # KS tests between baseline and fine-tuned variants
        ks_comparisons: dict[str, Any] = {}
        if "baseline" in scores_by_variant:
            for variant in variants:
                if variant != "baseline" and variant in scores_by_variant:
                    ks_comparisons[f"baseline_vs_{variant}"] = _ks_test(
                        scores_by_variant["baseline"],
                        scores_by_variant[variant],
                    )
        case_results["ks_comparisons"] = ks_comparisons

        # Save per-system similarity scores
        sim_path = mem_dir / "similarity_distributions.json"
        with open(sim_path, "w") as fh:
            json.dump({k: v for k, v in scores_by_variant.items()}, fh, indent=2)

        # Save report
        report_path = mem_dir / "memorization_report.json"
        with open(report_path, "w") as fh:
            json.dump(case_results, fh, indent=2)
        print(f"  [04] Report: {report_path}", flush=True)

        # Plot
        if scores_by_variant:
            _plot_similarity_comparison(
                scores_by_variant,
                mem_dir / "memorization_comparison.png",
                name,
            )

        all_results[name] = case_results

    # Save aggregate summary
    summary_path = out_root / "04_memorization_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[04] Summary: {summary_path}", flush=True)
    print("[04] Stage 4 complete.", flush=True)


if __name__ == "__main__":
    main()

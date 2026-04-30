#!/usr/bin/env python3
"""Compute Cα-RMSD between raw Boltz terminal structures and their OpenMM local minima.

For each PDB in ``--pdb-dir``:

1. Load the heavy-atom structure produced by Boltz.
2. Add hydrogens with AMBER14 (``Modeller.addHydrogens``).
3. Energy-minimise with AMBER14 + explicit TIP3P solvent in a periodic box
   (PME) on a CUDA GPU (falls back to OpenCL then CPU if CUDA is unavailable).
4. Compute the Kabsch-aligned Cα-RMSD between the raw and minimised structures.
5. Write per-structure results to ``--out-dir/rmsd_results.json``.
6. Plot the distribution as ``--out-dir/rmsd_distribution.png``.

Ligand support::

    Boltz writes non-polymer ligands as ``HETATM`` records with ``resName = LIG``.
    AMBER14 has no template for ``LIG``.  Providing a SMILES string for each
    ligand chain activates ``GAFFTemplateGenerator`` (from ``openmmforcefields``),
    which registers GAFF2 parameters on-the-fly.

    Via CLI::

        python scripts/compute_cv_rmsd.py \\
            --pdb-dir  ... \\
            --out-dir  ... \\
            --ligand-smiles-json '{"B": "CC(=O)Oc1ccccc1C(=O)O", "C": "[Mg+2]"}'

    ``[Mg+2]`` and similar single-atom SMILES use ``amber14/tip3p.xml`` ion
    templates instead of GAFF2.

    Via Python API::

        result = minimize_pdb(
            pdb_path,
            ligand_smiles={"B": "CC(=O)Oc1ccccc1C(=O)O"},
        )

Usage::

    python scripts/compute_cv_rmsd.py \\
        --pdb-dir  cofolding_tps_out_fixed/checkpoint_last_frames \\
        --out-dir  cofolding_tps_out_fixed/cv_rmsd_analysis \\
        --max-iter 1000 \\
        --platform CUDA

The Cα-RMSD is a measure of how far each Boltz-generated conformation sits from
the nearest local energy minimum.  Large values indicate that Boltz is sampling
geometrically strained or physically unrealistic structures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from genai_tps.simulation.openmm_system_builder import (  # noqa: F401
    _get_platform,
    _ligand_openmm_positions_from_pdb_heavy_atoms,
    _ligand_topology_relabeled_chain,
    _platform_runtime_smoke_test,
    _rdkit_ligand_positions_nm_from_pdb,
    build_md_simulation_from_pdb,
    get_ca_coords_angstrom,
    minimize_pdb,
    select_ca_indices,
    tip3p_monoatomic_template,
)
from genai_tps.utils.compute_device import openmm_device_index_properties

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------

def plot_rmsd_distribution(
    rmsd_values: list[float],
    out_path: Path,
    *,
    title: str = "Boltz terminal structures: Cα RMSD to local energy minimum",
) -> None:
    """Save a histogram + KDE of RMSD values.

    Parameters
    ----------
    rmsd_values:
        List of per-structure Cα-RMSD values in Angstroms.
    out_path:
        Output PNG file path.
    title:
        Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    n_bins = max(5, min(20, len(rmsd_values)))
    ax.hist(
        rmsd_values,
        bins=n_bins,
        density=True,
        alpha=0.55,
        color="#4A86E8",
        edgecolor="white",
        linewidth=0.6,
        label="histogram",
    )

    if len(rmsd_values) >= 3:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(rmsd_values, bw_method="scott")
            x_grid = np.linspace(
                min(rmsd_values) - 0.5, max(rmsd_values) + 0.5, 300
            )
            ax.plot(x_grid, kde(x_grid), color="#E97132", linewidth=2.0, label="KDE")
        except Exception:
            pass

    ax.set_xlabel("Cα RMSD to local energy minimum (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False)

    mean_v = np.mean(rmsd_values)
    ax.axvline(mean_v, color="gray", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_v:.2f} Å")
    ax.annotate(
        f"mean = {mean_v:.2f} Å\nn = {len(rmsd_values)}",
        xy=(mean_v, ax.get_ylim()[1] * 0.85),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribution plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pdb-dir",
        type=Path,
        required=True,
        help="Directory containing PDB files to analyse (e.g. checkpoint_last_frames/).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for results JSON and plot.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum L-BFGS iterations for energy minimisation (default: 1000).",
    )
    p.add_argument(
        "--platform",
        default="CUDA",
        choices=["CUDA", "OpenCL", "CPU"],
        help="Preferred OpenMM platform (default: CUDA; falls back automatically).",
    )
    p.add_argument(
        "--openmm-device-index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CUDA/OpenCL GPU ordinal for OpenMM (maps to platform property "
            "DeviceIndex). Ignored when the resolved platform is CPU."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        metavar="K",
        help="Temperature in Kelvin for the Langevin integrator (default: 300).",
    )
    p.add_argument(
        "--glob",
        default="*.pdb",
        metavar="PATTERN",
        help="Glob pattern to select PDB files inside --pdb-dir (default: '*.pdb').",
    )
    p.add_argument(
        "--ligand-smiles-json",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            'JSON string (or path to a JSON file) mapping PDB chain ID to SMILES. '
            'Required for structures that contain HETATM LIG residues.  '
            'Example: \'{"B": "CC(=O)Oc1ccccc1C(=O)O", "C": "[Mg+2]"}\''
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)

    ligand_smiles: Optional[dict[str, str]] = None
    if args.ligand_smiles_json is not None:
        raw = args.ligand_smiles_json.strip()
        if raw.startswith("{"):
            import json as _json
            ligand_smiles = _json.loads(raw)
        else:
            import json as _json
            ligand_smiles = _json.loads(Path(raw).read_text())
        logger.info("Ligand SMILES: %s", ligand_smiles)

    pdb_files = sorted(args.pdb_dir.glob(args.glob))
    if not pdb_files:
        logger.error("No PDB files matching '%s' in %s", args.glob, args.pdb_dir)
        sys.exit(1)

    logger.info(
        "Found %d PDB files in %s — platform: %s, max_iter: %d",
        len(pdb_files),
        args.pdb_dir,
        args.platform,
        args.max_iter,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, pdb_path in enumerate(pdb_files, 1):
        logger.info("[%d/%d] Minimising %s …", i, len(pdb_files), pdb_path.name)
        omm_props = openmm_device_index_properties(
            args.platform, args.openmm_device_index
        )
        res = minimize_pdb(
            pdb_path,
            max_iter=args.max_iter,
            platform_name=args.platform,
            temperature_k=args.temperature,
            ligand_smiles=ligand_smiles,
            platform_properties=omm_props if omm_props else None,
        )
        results.append(res)
        status = (
            f"RMSD = {res['ca_rmsd_angstrom']:.3f} Å  "
            f"E = {res['energy_kj_mol']:.1f} kJ/mol"
            if res["converged"]
            else f"FAILED: {res['error']}"
        )
        logger.info("  → %s", status)

    # Write JSON
    json_path = args.out_dir / "rmsd_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info("Results → %s", json_path)

    # Summary statistics
    converged = [r for r in results if r["converged"]]
    failed = [r for r in results if not r["converged"]]
    if failed:
        logger.warning(
            "%d / %d structures failed minimisation:", len(failed), len(results)
        )
        for r in failed:
            logger.warning("  %s  %s", Path(r["pdb_path"]).name, r["error"])

    if converged:
        rmsd_vals = [r["ca_rmsd_angstrom"] for r in converged]
        logger.info(
            "Cα-RMSD summary (n=%d):  mean=%.3f Å  min=%.3f Å  max=%.3f Å",
            len(rmsd_vals),
            np.mean(rmsd_vals),
            np.min(rmsd_vals),
            np.max(rmsd_vals),
        )
        plot_path = args.out_dir / "rmsd_distribution.png"
        plot_rmsd_distribution(rmsd_vals, plot_path)
    else:
        logger.error("All structures failed minimisation — no plot generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()

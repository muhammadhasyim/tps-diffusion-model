#!/usr/bin/env python3
"""Compute Cα-RMSD between raw Boltz terminal structures and their OpenMM local minima.

For each PDB in ``--pdb-dir``:

1. Load the heavy-atom structure produced by Boltz.
2. Add hydrogens with AMBER14 (``Modeller.addHydrogens``).
3. Energy-minimise with AMBER14 + implicit GBSA-OBC2 on a CUDA GPU (falls back
   to OpenCL then CPU if CUDA is unavailable).
4. Compute the Kabsch-aligned Cα-RMSD between the raw and minimised structures.
5. Write per-structure results to ``--out-dir/rmsd_results.json``.
6. Plot the distribution as ``--out-dir/rmsd_distribution.png``.

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
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def select_ca_indices(topology) -> list[int]:
    """Return 0-based atom indices for all Cα atoms in *topology*.

    Parameters
    ----------
    topology:
        An ``openmm.app.Topology`` instance.

    Returns
    -------
    list[int]
        Sorted list of Cα atom indices.
    """
    return [atom.index for atom in topology.atoms() if atom.name == "CA"]


def get_ca_coords_angstrom(positions, ca_indices: list[int]) -> np.ndarray:
    """Extract Cα positions from an OpenMM position array and convert nm → Å.

    Parameters
    ----------
    positions:
        OpenMM quantity or plain numpy array in **nanometres**.
    ca_indices:
        Indices of Cα atoms (from :func:`select_ca_indices`).

    Returns
    -------
    np.ndarray
        Shape ``(N_CA, 3)`` in Angstroms.
    """
    try:
        from openmm import unit
        pos_nm = np.array(
            [[positions[i].x, positions[i].y, positions[i].z] for i in ca_indices],
            dtype=np.float64,
        )
    except AttributeError:
        pos_nm = np.asarray(positions, dtype=np.float64)[ca_indices]
    return pos_nm * 10.0  # nm → Å


# ---------------------------------------------------------------------------
# GPU platform selection
# ---------------------------------------------------------------------------

_PLATFORM_PREFERENCE = ("CUDA", "OpenCL", "CPU")


def _get_platform(requested: str):
    """Return the best available OpenMM Platform, falling back gracefully.

    Parameters
    ----------
    requested:
        Preferred platform name (``"CUDA"``, ``"OpenCL"``, or ``"CPU"``).

    Returns
    -------
    tuple[openmm.Platform, str]
        The platform object and its actual name.
    """
    import openmm

    candidates = (
        [requested] + [p for p in _PLATFORM_PREFERENCE if p != requested]
    )
    for name in candidates:
        try:
            platform = openmm.Platform.getPlatformByName(name)
            if name == "CUDA":
                platform.setPropertyDefaultValue("CudaPrecision", "mixed")
            if name != requested:
                warnings.warn(
                    f"Platform '{requested}' not available; using '{name}'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return platform, name
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform found (tried CUDA, OpenCL, CPU).")


# ---------------------------------------------------------------------------
# Core minimisation function
# ---------------------------------------------------------------------------

def minimize_pdb(
    pdb_path: Path,
    *,
    max_iter: int = 1000,
    platform_name: str = "CUDA",
    temperature_k: float = 300.0,
) -> dict:
    """Load a PDB, add hydrogens, minimise on GPU, return Cα-RMSD and energy.

    The function:

    1. Loads *pdb_path* with ``openmm.app.PDBFile``.
    2. Selects Cα atoms from the **original** (heavy-atom) topology and records
       their coordinates.
    3. Adds hydrogens via ``Modeller.addHydrogens(amber14-all.xml)``.
    4. Re-selects Cα atoms from the hydrogen-added topology (atom indices shift
       after ``addHydrogens``).
    5. Builds an AMBER14 / implicit-GBSA-OBC2 system and minimises.
    6. Returns Kabsch-aligned Cα-RMSD (Å) between pre- and post-minimisation
       coordinates, plus the final potential energy.

    Parameters
    ----------
    pdb_path:
        Path to the heavy-atom PDB file produced by Boltz.
    max_iter:
        Maximum number of L-BFGS steps for ``minimizeEnergy``.
    platform_name:
        Preferred OpenMM platform.  Falls back to softer platforms if
        unavailable.
    temperature_k:
        Temperature used for the Langevin integrator (K).

    Returns
    -------
    dict with keys:
        ``pdb_path`` (str), ``n_residues`` (int), ``n_ca_atoms`` (int),
        ``energy_kj_mol`` (float), ``ca_rmsd_angstrom`` (float),
        ``platform_used`` (str), ``converged`` (bool), ``error`` (str | None).
    """
    result: dict = {
        "pdb_path": str(pdb_path),
        "n_residues": 0,
        "n_ca_atoms": 0,
        "energy_kj_mol": None,
        "ca_rmsd_angstrom": None,
        "platform_used": None,
        "converged": False,
        "error": None,
    }
    try:
        import openmm
        import openmm.app
        from openmm import unit
        from genai_tps.backends.boltz.collective_variables import kabsch_rmsd_aligned

        # 1. Load original heavy-atom structure to record Cα positions before
        #    any atom addition (PDBFixer changes atom count).
        pdb_orig = openmm.app.PDBFile(str(pdb_path))
        orig_topology = pdb_orig.topology
        orig_positions = pdb_orig.positions

        n_residues = orig_topology.getNumResidues()
        result["n_residues"] = n_residues

        ca_orig_idx = select_ca_indices(orig_topology)
        if not ca_orig_idx:
            raise ValueError("No Cα atoms found in the original topology.")
        ca_orig_coords = get_ca_coords_angstrom(orig_positions, ca_orig_idx)

        # 2. Prepare the structure (add terminal caps + hydrogens).
        #
        #    Strategy A — PDBFixer (preferred):
        #      Boltz outputs heavy atoms only; AMBER14 expects terminal residue
        #      templates (NMET, CALA …).  PDBFixer handles all of that via
        #      findMissingAtoms / addMissingHydrogens.
        #
        #    Strategy B — Modeller.addHydrogens (fallback):
        #      Used when PDBFixer's star-import of `from openmm import *`
        #      triggers a CUDA plugin load error on the current machine
        #      (e.g. CUDA_ERROR_UNSUPPORTED_PTX_VERSION).  Works for well-
        #      formed dipeptides; may warn/fail for large proteins.
        #
        #    Strategy C — heavy-atoms-only (last resort):
        #      Sufficient for a Cα-RMSD comparison even if the absolute energy
        #      is physically unreliable (force field will still relax geometry).
        h_topology: openmm.app.Topology
        h_positions: object

        _pdbfixer_ok = False
        try:
            from pdbfixer import PDBFixer  # noqa: PLC0415
            fixer = PDBFixer(filename=str(pdb_path))
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(pH=7.0)
            h_topology = fixer.topology
            h_positions = fixer.positions
            _pdbfixer_ok = True
        except Exception as pdbfixer_exc:
            warnings.warn(
                f"{pdb_path.name}: PDBFixer failed ({pdbfixer_exc}); "
                "falling back to Modeller.addHydrogens.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not _pdbfixer_ok:
            # Strategy B: Modeller fallback
            ff_h = openmm.app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
            modeller = openmm.app.Modeller(orig_topology, orig_positions)
            try:
                modeller.addHydrogens(forcefield=ff_h)
            except Exception as h_exc:
                warnings.warn(
                    f"{pdb_path.name}: addHydrogens also failed ({h_exc}); "
                    "proceeding with heavy atoms only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            h_topology = modeller.topology
            h_positions = modeller.positions

        # 3. Cα indices in the fully protonated topology.
        ca_h_idx = select_ca_indices(h_topology)
        result["n_ca_atoms"] = len(ca_h_idx)

        # 4. Build AMBER14 + GBn2 implicit solvent system (NoCutoff, no periodic box).
        #    Force field: amber14-all.xml + implicit/gbn2.xml (GBn2, igb=8).
        #    GBn2 is the highest-accuracy GB model in OpenMM and the one recommended
        #    in the OpenMM user guide for AMBER14 force fields.
        #    Ref: docs.openmm.org §3.2 Implicit Solvent.
        #    NOTE: do NOT pass implicitSolvent= to createSystem() — that kwarg is
        #    only for AMBER prmtop files.  The GB parameters come from the XML.
        ff = openmm.app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(
            h_topology,
            nonbondedMethod=openmm.app.NoCutoff,
            constraints=openmm.app.HBonds,
        )

        integrator = openmm.LangevinIntegrator(
            temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtoseconds,
        )

        platform, actual_platform = _get_platform(platform_name)
        result["platform_used"] = actual_platform

        sim = openmm.app.Simulation(h_topology, system, integrator, platform)
        sim.context.setPositions(h_positions)

        # 5. Minimise
        sim.minimizeEnergy(maxIterations=max_iter)

        state = sim.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        min_positions = state.getPositions(asNumpy=True)  # nm

        # 6. Kabsch-aligned Cα-RMSD
        ca_min_coords = get_ca_coords_angstrom(min_positions, ca_h_idx)
        rmsd = kabsch_rmsd_aligned(ca_orig_coords, ca_min_coords)

        result["energy_kj_mol"] = float(energy)
        result["ca_rmsd_angstrom"] = float(rmsd)
        result["converged"] = True

    except Exception as exc:
        logger.error("%s: minimisation failed — %s", pdb_path.name, exc)
        result["error"] = str(exc)

    return result


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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)

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
        res = minimize_pdb(
            pdb_path,
            max_iter=args.max_iter,
            platform_name=args.platform,
            temperature_k=args.temperature,
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

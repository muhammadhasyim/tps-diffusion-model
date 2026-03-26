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


def _platform_runtime_smoke_test(platform) -> tuple[bool, str | None]:
    """Return whether *platform* can run a minimal Context (energy evaluation).

    ``Platform.getPlatformByName('CUDA')`` can succeed while the first real
    kernel fails with ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` (driver/toolkit
    mismatch).  PDBFixer and ``Modeller.addHydrogens`` then break before our
    main :class:`Simulation` is built.  This test catches that case so we fall
    back to OpenCL or CPU.
    """
    import openmm as mm
    from openmm import unit as u

    try:
        system = mm.System()
        system.addParticle(12.0)
        system.addParticle(12.0)
        bond = mm.HarmonicBondForce()
        bond.addBond(0, 1, 0.15, 100000.0)
        system.addForce(bond)
        integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
        ctx = mm.Context(system, integrator, platform)
        ctx.setPositions([[0, 0, 0], [0.15, 0, 0]] * u.nanometers)
        ctx.getState(getEnergy=True)
        del ctx
        return True, None
    except Exception as exc:
        return False, str(exc)


def _get_platform(requested: str):
    """Return the best available OpenMM Platform, falling back gracefully.

    CUDA is accepted only if a minimal Context runs on it; otherwise the next
    candidate (OpenCL, then CPU) is used.

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
                ok, err = _platform_runtime_smoke_test(platform)
                if not ok:
                    warnings.warn(
                        f"OpenMM CUDA failed runtime check ({err}); "
                        "trying another platform.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue
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
# Ligand force-field helpers (GAFF2 via openmmforcefields)
# ---------------------------------------------------------------------------

def _register_ligand_params(
    ff,  # openmm.app.ForceField
    topology,  # openmm.app.Topology
    ligand_smiles: dict[str, str],
) -> None:
    """Register GAFF2 parameters for ``LIG`` residues in *topology*.

    For each chain in *topology* that contains a residue named ``LIG`` the
    function looks up the chain ID in *ligand_smiles*, converts the SMILES to
    an OpenFF ``Molecule``, and registers a ``GAFFTemplateGenerator`` on *ff*.
    A warning is emitted (not an exception) for any ``LIG`` chain that has no
    SMILES entry; those chains will cause ``ff.createSystem()`` to fail with
    ``"No template found"`` unless they have been stripped beforehand.

    Parameters
    ----------
    ff:
        ``openmm.app.ForceField`` instance to extend.
    topology:
        OpenMM topology whose residues may include ``LIG`` records.
    ligand_smiles:
        Maps PDB chain ID (e.g. ``"B"``, ``"C"``) to SMILES string.
        Single-atom ions should use standard SMILES (e.g. ``"[Mg+2]"``).

    Raises
    ------
    ImportError
        When ``openff-toolkit`` or ``openmmforcefields`` is not installed.
    """
    try:
        from openff.toolkit.topology import Molecule as OpenFFMolecule  # noqa: PLC0415
        from openmmforcefields.generators import GAFFTemplateGenerator  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openff-toolkit and openmmforcefields are required for ligand "
            "parameterisation.  Install with:\n"
            "  conda install -c conda-forge openmmforcefields\n"
            f"Original error: {exc}"
        ) from exc

    lig_chains: dict[str, list] = {}
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == "LIG":
                lig_chains.setdefault(chain.id, [])
                break

    molecules: list = []
    for chain_id in sorted(lig_chains):
        smiles = ligand_smiles.get(chain_id)
        if smiles is None:
            logger.warning(
                "_register_ligand_params: chain '%s' contains LIG residue(s) "
                "but no SMILES was provided; ff.createSystem() will fail for "
                "this chain unless it is removed from the topology.",
                chain_id,
            )
            continue
        try:
            mol = OpenFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
            molecules.append(mol)
            logger.info(
                "_register_ligand_params: registered GAFF2 for chain '%s' "
                "SMILES='%.40s'",
                chain_id, smiles,
            )
        except Exception as exc:
            logger.warning(
                "_register_ligand_params: could not build OpenFF Molecule for "
                "chain '%s' SMILES='%s': %s",
                chain_id, smiles, exc,
            )

    if molecules:
        gaff = GAFFTemplateGenerator(molecules=molecules)
        ff.registerTemplateGenerator(gaff.generator)


# ---------------------------------------------------------------------------
# Core minimisation function
# ---------------------------------------------------------------------------

def minimize_pdb(
    pdb_path: Path,
    *,
    max_iter: int = 1000,
    platform_name: str = "CUDA",
    temperature_k: float = 300.0,
    ligand_smiles: Optional[dict[str, str]] = None,
) -> dict:
    """Load a PDB, add hydrogens, minimise on GPU, return Cα-RMSD and energy.

    The function:

    1. Loads *pdb_path* with ``openmm.app.PDBFile``.
    2. Selects Cα atoms from the **original** (heavy-atom) topology and records
       their coordinates.
    3. Prepares the structure for simulation (adds H via PDBFixer or Modeller).
    4. Re-selects Cα atoms from the hydrogen-added topology (atom indices shift
       after ``addHydrogens``).
    5. Builds an AMBER14 / implicit-GBSA-OBC2 system and minimises.
    6. Returns Kabsch-aligned Cα-RMSD (Å) between pre- and post-minimisation
       coordinates, plus the final potential energy.

    Ligand handling
    ---------------
    When *ligand_smiles* is provided, AMBER14 is extended with ``GAFFTemplateGenerator``
    (GAFF2) for each chain ID in the mapping.  PDBFixer is **not** used for
    ligand-containing structures (it cannot add atoms for unknown ``LIG``
    residues and may strip HETATM records); Modeller is used directly.

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
    ligand_smiles:
        Optional mapping of PDB chain ID (``"B"``, ``"C"``, …) to SMILES.
        Required for structures with ``HETATM LIG`` residues; omitting it for
        such structures will result in ``"No template found"`` failure.

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

        # Resolve platform before PDBFixer / Modeller: both accept ``platform=``.
        # Without it, OpenMM picks the fastest platform (often CUDA), which can
        # fail at runtime with PTX/driver mismatch while ``getPlatformByName``
        # still "succeeds".
        platform, actual_platform = _get_platform(platform_name)
        result["platform_used"] = actual_platform

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

        # 2. Build the shared ForceField (with GAFF2 extension if ligands present).
        #
        #    A single ForceField instance is used for both addHydrogens and
        #    createSystem so that the GAFF2 template generator is registered once
        #    and available at both stages.
        ff = openmm.app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        has_ligand = bool(ligand_smiles)
        if has_ligand:
            _register_ligand_params(ff, orig_topology, ligand_smiles)

        # 3. Prepare the structure (add terminal caps + hydrogens).
        #
        #    Strategy A — PDBFixer (preferred, protein-only structures):
        #      Boltz outputs heavy atoms only; AMBER14 expects terminal residue
        #      templates (NMET, CALA …).  PDBFixer handles all of that via
        #      findMissingAtoms / addMissingHydrogens.
        #      NOTE: PDBFixer must NOT be given the GPU platform — it creates its
        #      own OpenMM Context internally, which collides with the Boltz GPU
        #      model and causes clCreateContext (-6).  CPU is always sufficient
        #      for hydrogen addition.
        #      Skipped when ligand_smiles is provided because PDBFixer does not
        #      know the LIG residue template and may strip HETATM atoms.
        #
        #    Strategy B — Modeller.addHydrogens (ligand structures or fallback):
        #      Used when PDBFixer is skipped (has_ligand=True) or when PDBFixer
        #      fails.  May warn/fail for uncapped chains (terminal residues
        #      missing backbone atoms).
        #
        #    Strategy C — heavy-atoms-only (last resort):
        #      Sufficient for a Cα-RMSD comparison even if the absolute energy
        #      is physically unreliable (force field will still relax geometry).
        h_topology: openmm.app.Topology
        h_positions: object

        _pdbfixer_ok = False
        if not has_ligand:
            try:
                from pdbfixer import PDBFixer  # noqa: PLC0415
                # Do NOT pass the GPU platform to PDBFixer — it only adds hydrogens
                # and terminal caps (CPU work), but internally creates an OpenMM
                # Context.  Passing OpenCL here causes clCreateContext (-6) because
                # the GPU is already occupied by the Boltz diffusion model.
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
            # Strategy B: Modeller fallback (or primary path for ligands).
            # Use the same ff instance that has GAFF2 registered (if applicable).
            modeller = openmm.app.Modeller(orig_topology, orig_positions)
            try:
                modeller.addHydrogens(forcefield=ff, platform=platform)
            except Exception as h_exc:
                warnings.warn(
                    f"{pdb_path.name}: addHydrogens also failed ({h_exc}); "
                    "proceeding with heavy atoms only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            h_topology = modeller.topology
            h_positions = modeller.positions

        # 4. Cα indices in the fully protonated topology.
        ca_h_idx = select_ca_indices(h_topology)
        result["n_ca_atoms"] = len(ca_h_idx)

        # 5. Build AMBER14 + GBn2 implicit solvent system (NoCutoff, no periodic box).
        #    Force field: amber14-all.xml + implicit/gbn2.xml (GBn2, igb=8).
        #    GBn2 is the highest-accuracy GB model in OpenMM and the one recommended
        #    in the OpenMM user guide for AMBER14 force fields.
        #    Ref: docs.openmm.org §3.2 Implicit Solvent.
        #    NOTE: do NOT pass implicitSolvent= to createSystem() — that kwarg is
        #    only for AMBER prmtop files.  The GB parameters come from the XML.
        #    NOTE: ff already has GAFF2 registered if has_ligand=True.
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

        sim = openmm.app.Simulation(h_topology, system, integrator, platform)
        sim.context.setPositions(h_positions)

        # 6. Minimise
        sim.minimizeEnergy(maxIterations=max_iter)

        state = sim.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        min_positions = state.getPositions(asNumpy=True)  # nm

        # 7. Kabsch-aligned Cα-RMSD
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
        res = minimize_pdb(
            pdb_path,
            max_iter=args.max_iter,
            platform_name=args.platform,
            temperature_k=args.temperature,
            ligand_smiles=ligand_smiles,
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

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
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Boltz/OpenMM PDB: protein uses standard 3-letter codes; cofactors are HETATM
# with CCD names (ATP, MG, …), not necessarily ``LIG``.  Anything outside this
# set is treated as non-polymer for the ligand+protein protonation pipeline.
_CANONICAL_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",  # occasional in deposited structures / Boltz
})

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
# Monoatomic ions: AMBER14 tip3p.xml (Joung–Cheatham–like), not GAFF2
# ---------------------------------------------------------------------------
# GAFFTemplateGenerator always runs AM1-BCC unless partial charges are preset,
# and antechamber cannot charge/type single-atom species reliably.  OpenMM's
# ``amber14/tip3p.xml`` ships one-residue ion templates (MG, NA, CL, …) with
# Lennard-Jones and charges consistent with the TIP3P ion parameter set.
#
# Keys: (atomic_number, formal_charge) → (OpenMM residue name, PDB atom name).
_TIP3P_MONOATOMIC_BY_ELEMENT_CHARGE: dict[tuple[int, int], tuple[str, str]] = {
    (3, 1): ("LI", "LI"),
    (4, 2): ("Be", "Be"),
    (9, -1): ("F", "F"),
    (11, 1): ("NA", "NA"),
    (12, 2): ("MG", "MG"),
    (13, 3): ("AL", "AL"),
    (17, -1): ("CL", "CL"),
    (19, 1): ("K", "K"),
    (20, 2): ("CA", "CA"),
    (23, 2): ("V2+", "V2+"),
    (24, 2): ("Cr", "Cr"),
    (24, 3): ("CR", "CR"),
    (25, 2): ("MN", "MN"),
    (26, 2): ("FE2", "FE2"),
    (26, 3): ("FE", "FE"),
    (27, 2): ("CO", "CO"),
    (28, 2): ("NI", "NI"),
    (29, 2): ("CU", "CU"),
    (30, 2): ("ZN", "ZN"),
    (35, -1): ("BR", "BR"),
    (37, 1): ("RB", "RB"),
    (38, 2): ("SR", "SR"),
    (39, 3): ("Y", "Y"),
    (40, 4): ("Zr", "Zr"),
    (46, 2): ("PD", "PD"),
    (47, 2): ("Ag", "Ag"),
    (48, 2): ("CD", "CD"),
    (49, 3): ("IN", "IN"),
    (50, 2): ("Sn", "Sn"),
    (53, -1): ("IOD", "I"),
    (55, 1): ("CS", "CS"),
    (56, 2): ("BA", "BA"),
    (57, 3): ("LA", "LA"),
    (58, 3): ("CE", "CE"),
    (58, 4): ("Ce", "Ce"),
    (59, 3): ("PR", "PR"),
    (60, 3): ("Nd", "Nd"),
    (62, 2): ("Sm", "Sm"),
    (62, 3): ("SM", "SM"),
    (63, 2): ("EU", "EU"),
    (63, 3): ("EU3", "EU3"),
    (64, 3): ("GD3", "GD"),
    (65, 3): ("TB", "TB"),
    (66, 3): ("Dy", "Dy"),
    (68, 3): ("Er", "Er"),
    (69, 3): ("Tm", "Tm"),
    (70, 2): ("YB2", "YB2"),
    (71, 3): ("LU", "LU"),
    (72, 4): ("Hf", "Hf"),
    (78, 2): ("PT", "PT"),
    (80, 2): ("HG", "HG"),
    (81, 3): ("Tl", "Tl"),
    (82, 2): ("PB", "PB"),
    (88, 2): ("Ra", "Ra"),
    (90, 4): ("Th", "Th"),
    (92, 4): ("U4+", "U"),
    (94, 4): ("Pu", "Pu"),
}


def tip3p_monoatomic_template(
    smiles: str,
) -> tuple[str, str, int] | None:
    """If *smiles* is a single atom with a tip3p.xml entry, return template info.

    Returns
    -------
    tuple[str, str, int] | None
        ``(residue_name, atom_name, atomic_number)`` for OpenMM's
        ``amber14/tip3p.xml`` ion residue, or ``None`` if the species should use
        GAFF2 instead (multi-atom ligand or ion not in the bundled tip3p set).

    Notes
    -----
    Uses RDKit when available (lightweight); falls back to OpenFF ``Molecule``
    so environments without RDKit still classify ions if the toolkit imports.
    """
    if not smiles or not smiles.strip():
        return None
    s = smiles.strip()
    z: int | None = None
    fc: int | None = None
    try:
        from rdkit import Chem  # noqa: PLC0415

        rdm = Chem.MolFromSmiles(s)
        if rdm is None or rdm.GetNumAtoms() != 1:
            return None
        ra = rdm.GetAtomWithIdx(0)
        z = int(ra.GetAtomicNum())
        fc = int(ra.GetFormalCharge())
    except ImportError:
        try:
            from openff.toolkit.topology import (  # noqa: PLC0415
                Molecule as OpenFFMolecule,
            )

            mol = OpenFFMolecule.from_smiles(s, allow_undefined_stereo=True)
        except ImportError:
            return None
        except Exception:
            return None
        if len(mol.atoms) != 1:
            return None
        atom = mol.atoms[0]
        z = int(atom.atomic_number)
        fc = int(atom.formal_charge)
    hit = _TIP3P_MONOATOMIC_BY_ELEMENT_CHARGE.get((z, fc))
    if hit is None:
        return None
    res_name, atom_name = hit
    return (res_name, atom_name, z)


def _ligand_smiles_needs_tip3p_xml(ligand_smiles: dict[str, str]) -> bool:
    """True if any SMILES should be parameterised via ``amber14/tip3p.xml``."""
    for smi in ligand_smiles.values():
        if smi and tip3p_monoatomic_template(smi):
            return True
    return False


def _ligand_positions_by_chain(
    topology,
    positions,
    ligand_smiles: dict[str, str],
) -> dict[str, list]:
    """Map chain ID → list of ``Quantity`` positions for atoms in that chain."""
    out: dict[str, list] = {}
    for chain in topology.chains():
        chain_id = chain.id.strip()
        if chain_id not in ligand_smiles:
            continue
        atoms = list(chain.atoms())
        if not atoms:
            continue
        out[chain_id] = [positions[a.index] for a in atoms]
    return out


# ---------------------------------------------------------------------------
# Ligand force-field helpers (GAFF2 via openmmforcefields)
# ---------------------------------------------------------------------------

def _register_ligand_params(
    ff,  # openmm.app.ForceField
    topology,  # openmm.app.Topology
    ligand_smiles: dict[str, str],
) -> None:
    """Register GAFF2 parameters for cofactor chains listed in *ligand_smiles*.

    Registers one OpenFF ``Molecule`` per chain ID in *ligand_smiles* (Boltz
    writes ATP/MG/… as HETATM with CCD residue names, not necessarily ``LIG``).
    Chains present in the PDB but omitted from *ligand_smiles* are not covered
    here — they should be stripped before ``createSystem`` if non-protein.

    Parameters
    ----------
    ff:
        ``openmm.app.ForceField`` instance to extend.
    topology:
        Unused except for future validation; kept for API compatibility.
    ligand_smiles:
        Maps PDB chain ID (e.g. ``"B"``, ``"C"``) to SMILES string.
        Single-atom ions should use standard SMILES (e.g. ``"[Mg+2]"``).

    Raises
    ------
    ImportError
        When ``openff-toolkit`` or ``openmmforcefields`` is not installed and at
        least one ligand chain requires GAFF2 (non–tip3p monoatomic species).
    """
    del topology  # reserved for future chain-existence checks
    gaff_targets: dict[str, str] = {}
    for chain_id, smiles in ligand_smiles.items():
        if not smiles:
            continue
        if tip3p_monoatomic_template(smiles):
            logger.info(
                "_register_ligand_params: chain '%s' is a tip3p monoatomic ion "
                "(skip GAFF2); SMILES='%.40s'",
                chain_id,
                smiles,
            )
            continue
        gaff_targets[chain_id] = smiles

    if not gaff_targets:
        return

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

    molecules: list = []
    for chain_id in sorted(gaff_targets):
        smiles = gaff_targets[chain_id]
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
# Persistent MD simulation (no minimisation) — shared prep with ``minimize_pdb``
# ---------------------------------------------------------------------------


def build_md_simulation_from_pdb(
    pdb_path: Path,
    *,
    platform_name: str = "CUDA",
    temperature_k: float = 300.0,
    ligand_smiles: Optional[dict[str, str]] = None,
) -> tuple[object, dict]:
    """Build AMBER14 + implicit GBn2 Langevin :class:`openmm.app.Simulation` without minimising.

    Uses the same protonation / GAFF2 / PDBFixer logic as :func:`minimize_pdb`.
    Intended for persistent OpenMM contexts in FES-guided RL (MD bursts on the
    hot path avoid re-building the system).

    Parameters
    ----------
    pdb_path:
        Heavy-atom PDB (e.g. Boltz ``npz_to_pdb`` export).
    platform_name:
        Preferred OpenMM platform (CUDA / OpenCL / CPU); may fall back.
    temperature_k:
        Langevin bath temperature (K).
    ligand_smiles:
        Optional chain ID → SMILES mapping for cofactors (same as ``minimize_pdb``).

    Returns
    -------
    sim :
        Configured ``openmm.app.Simulation`` with positions already set.
    meta :
        ``platform_used``, ``n_residues``, ``n_ca_atoms``, ``ca_orig_coords``,
        ``ca_h_idx`` for optional downstream analysis (e.g. Cα RMSD after MD).
    """
    import openmm
    import openmm.app
    from openmm import unit

    meta: dict = {}

    platform, actual_platform = _get_platform(platform_name)
    meta["platform_used"] = actual_platform

    pdb_orig = openmm.app.PDBFile(str(pdb_path))
    orig_topology = pdb_orig.topology
    orig_positions = pdb_orig.positions

    n_residues = orig_topology.getNumResidues()
    meta["n_residues"] = n_residues

    ca_orig_idx = select_ca_indices(orig_topology)
    if not ca_orig_idx:
        raise ValueError("No Cα atoms found in the original topology.")
    ca_orig_coords = get_ca_coords_angstrom(orig_positions, ca_orig_idx)

    has_ligand = bool(ligand_smiles)
    ligand_chain_positions: dict[str, list] = {}
    if has_ligand and ligand_smiles is not None:
        ligand_chain_positions = _ligand_positions_by_chain(
            orig_topology, orig_positions, ligand_smiles
        )

    ff_paths = ["amber14-all.xml", "implicit/gbn2.xml"]
    if has_ligand and ligand_smiles and _ligand_smiles_needs_tip3p_xml(
        ligand_smiles
    ):
        ff_paths.append("amber14/tip3p.xml")
    ff = openmm.app.ForceField(*ff_paths)
    if has_ligand and ligand_smiles is not None:
        _register_ligand_params(ff, orig_topology, ligand_smiles)

    h_topology: openmm.app.Topology
    h_positions: object

    _pdbfixer_ok = False
    if not has_ligand:
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
        modeller = openmm.app.Modeller(orig_topology, orig_positions)
        if has_ligand:
            non_protein = [
                atom
                for atom in modeller.topology.atoms()
                if atom.residue.name.strip() not in _CANONICAL_AMINO_ACIDS
            ]
            if non_protein:
                modeller.delete(non_protein)
            _protein_h_ok = False
            if non_protein:
                prot_pdb: Path | None = None
                try:
                    fd, prot_name = tempfile.mkstemp(
                        suffix=".pdb", prefix="cv_rmsd_prot_"
                    )
                    prot_pdb = Path(prot_name)
                    os.close(fd)
                    with prot_pdb.open("w", encoding="utf-8") as tmp_f:
                        openmm.app.PDBFile.writeFile(
                            modeller.topology,
                            modeller.positions,
                            tmp_f,
                        )
                    from pdbfixer import PDBFixer  # noqa: PLC0415

                    fixer_lig = PDBFixer(filename=str(prot_pdb))
                    fixer_lig.findMissingResidues()
                    fixer_lig.findMissingAtoms()
                    fixer_lig.addMissingAtoms()
                    fixer_lig.addMissingHydrogens(pH=7.0)
                    modeller = openmm.app.Modeller(
                        fixer_lig.topology, fixer_lig.positions
                    )
                    _protein_h_ok = True
                except Exception as fixer_lig_exc:
                    warnings.warn(
                        f"{pdb_path.name}: PDBFixer on protein-only structure "
                        f"failed ({fixer_lig_exc}); falling back to "
                        "Modeller.addHydrogens.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                finally:
                    if prot_pdb is not None:
                        prot_pdb.unlink(missing_ok=True)
                if not _protein_h_ok:
                    try:
                        modeller.addHydrogens(forcefield=ff, platform=platform)
                    except Exception as h_exc:
                        warnings.warn(
                            f"{pdb_path.name}: addHydrogens also failed ({h_exc}); "
                            "proceeding with heavy atoms only.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                needs_openff_conformer = False
                if ligand_smiles:
                    for _cid, _smi in ligand_smiles.items():
                        if _smi and tip3p_monoatomic_template(_smi) is None:
                            needs_openff_conformer = True
                            break
                OpenFFMolecule = None
                if needs_openff_conformer:
                    try:
                        from openff.toolkit.topology import (  # noqa: PLC0415
                            Molecule as OpenFFMolecule,
                        )
                    except ImportError as exc:
                        raise ImportError(
                            "openff-toolkit is required for ligand protonation after "
                            "cofactor strip. Install with:\n"
                            "  conda install -c conda-forge openmmforcefields\n"
                            f"Original error: {exc}"
                        ) from exc
                for chain_id in sorted(ligand_smiles or {}):
                    smiles = ligand_smiles.get(chain_id) if ligand_smiles else None
                    if not smiles:
                        continue
                    ion_tpl = tip3p_monoatomic_template(smiles)
                    if ion_tpl is not None:
                        res_name, atom_name, atomic_number = ion_tpl
                        saved = ligand_chain_positions.get(chain_id)
                        if not saved:
                            warnings.warn(
                                f"{pdb_path.name}: no PDB positions for tip3p ion "
                                f"chain '{chain_id}'; placing ion at origin.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            lig_pos = [openmm.Vec3(0.0, 0.0, 0.0) * unit.nanometer]
                        else:
                            lig_pos = [saved[0]]
                        ion_topo = openmm.app.Topology()
                        ion_chain = ion_topo.addChain()
                        ion_res = ion_topo.addResidue(res_name, ion_chain)
                        ion_el = openmm.app.Element.getByAtomicNumber(atomic_number)
                        ion_topo.addAtom(atom_name, ion_el, ion_res)
                        modeller.add(ion_topo, lig_pos)
                        continue
                    assert OpenFFMolecule is not None
                    mol = OpenFFMolecule.from_smiles(
                        smiles, allow_undefined_stereo=True
                    )
                    mol.generate_conformers(n_conformers=1, rms_cutoff=None)
                    lig_topo = mol.to_topology().to_openmm()
                    conf_ang = mol.conformers[0].magnitude
                    conf_nm = conf_ang / 10.0
                    lig_pos = [
                        openmm.Vec3(
                            float(conf_nm[i, 0]),
                            float(conf_nm[i, 1]),
                            float(conf_nm[i, 2]),
                        )
                        * unit.nanometer
                        for i in range(int(conf_nm.shape[0]))
                    ]
                    modeller.add(lig_topo, lig_pos)
        else:
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

    ca_h_idx = select_ca_indices(h_topology)
    meta["n_ca_atoms"] = len(ca_h_idx)

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

    meta["ca_orig_coords"] = ca_orig_coords
    meta["ca_h_idx"] = ca_h_idx
    return sim, meta


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
    When *ligand_smiles* is provided, AMBER14 is extended with
    ``GAFFTemplateGenerator`` (GAFF2) for multi-atom cofactors, and with
    ``amber14/tip3p.xml`` for monoatomic ions recognised from SMILES (e.g.
    ``[Mg+2]`` → TIP3P-compatible Mg²⁺ parameters, no AM1-BCC).  PDBFixer is
    **not** used for ligand-containing structures (it cannot add atoms for
    unknown ``LIG`` residues and may strip HETATM records).

    ``GAFFTemplateGenerator`` matches residues to OpenFF molecules by graph
    isomorphism on **all** atoms.  A heavy-atom-only cofactor in the PDB
    therefore does not match a protonated SMILES molecule.  The implementation
    strips non–protein residues, runs PDBFixer on the protein-only structure,
    then re-inserts each ligand.  Multi-atom ligands use an OpenFF conformer
    (coordinates not taken from the PDB).  Monoatomic tip3p ions use the
    original PDB position of that chain.  Cα-RMSD compares original vs
    minimised **protein** Cα only.

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
        from openmm import unit
        from genai_tps.backends.boltz.collective_variables import kabsch_rmsd_aligned

        sim, meta = build_md_simulation_from_pdb(
            pdb_path,
            platform_name=platform_name,
            temperature_k=temperature_k,
            ligand_smiles=ligand_smiles,
        )
        result["platform_used"] = meta["platform_used"]
        result["n_residues"] = meta["n_residues"]
        result["n_ca_atoms"] = meta["n_ca_atoms"]
        ca_orig_coords = meta["ca_orig_coords"]
        ca_h_idx = meta["ca_h_idx"]

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

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
from typing import Any, Optional

import numpy as np

from genai_tps.simulation.openmm_boltz_bridge import (
    LigandPosePolicy,
    ligand_topology_relabeled_chain,
    try_ligand_pose_from_boltz_ccd,
)
from genai_tps.utils.compute_device import openmm_device_index_properties

logger = logging.getLogger(__name__)

# Tests import this symbol from ``compute_cv_rmsd``; keep alias to package helper.
_ligand_topology_relabeled_chain = ligand_topology_relabeled_chain

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


def _platform_runtime_smoke_test(
    platform,
    platform_properties: dict[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return whether *platform* can run a minimal Context (energy evaluation).

    ``Platform.getPlatformByName('CUDA')`` can succeed while the first real
    kernel fails with ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` (driver/toolkit
    mismatch).  PDBFixer and ``Modeller.addHydrogens`` then break before our
    main :class:`Simulation` is built.  This test catches that case so we fall
    back to OpenCL or CPU.

    When *platform_properties* is non-empty, it is passed to :class:`openmm.Context`
    so the smoke test targets the same GPU as the eventual :class:`Simulation`.
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
        props = dict(platform_properties or {})
        if props:
            ctx = mm.Context(system, integrator, platform, props)
        else:
            ctx = mm.Context(system, integrator, platform)
        ctx.setPositions([[0, 0, 0], [0.15, 0, 0]] * u.nanometers)
        ctx.getState(getEnergy=True)
        del ctx
        return True, None
    except Exception as exc:
        return False, str(exc)


def _get_platform(
    requested: str,
    *,
    runtime_smoke_properties: dict[str, str] | None = None,
):
    """Return the best available OpenMM Platform, falling back gracefully.

    CUDA is accepted only if a minimal Context runs on it; otherwise the next
    candidate (OpenCL, then CPU) is used.

    Parameters
    ----------
    requested:
        Preferred platform name (``"CUDA"``, ``"OpenCL"``, or ``"CPU"``).
    runtime_smoke_properties:
        Optional OpenMM property map (e.g. ``DeviceIndex``) passed into the CUDA
        runtime smoke test so the probe uses the same GPU as production.

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
                smoke_props: dict[str, str] | None = None
                if runtime_smoke_properties:
                    smoke_props = dict(runtime_smoke_properties)
                    try:
                        prop_names = set(platform.getPropertyNames())
                        if "Precision" in prop_names and "Precision" not in smoke_props:
                            smoke_props["Precision"] = "mixed"
                    except Exception:
                        pass
                ok, err = _platform_runtime_smoke_test(platform, smoke_props)
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
            from openff.toolkit import Molecule as OpenFFMolecule  # noqa: PLC0415

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


def _extract_pdb_chain_block(pdb_path: Path, chain_id: str) -> str:
    """Return ATOM/HETATM lines for *chain_id* plus CONECT lines linking only those atoms."""
    text = pdb_path.read_text(encoding="utf-8", errors="replace").splitlines()
    chain_id = chain_id.strip()
    atom_lines: list[str] = []
    serials: set[int] = set()
    for line in text:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 22:
            continue
        ch = line[21:22].strip() or " "
        if ch != chain_id:
            continue
        atom_lines.append(line)
        try:
            serials.add(int(line[6:11]))
        except ValueError:
            continue
    conect_lines: list[str] = []
    for line in text:
        if not line.startswith("CONECT"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            involved = [int(parts[k]) for k in range(1, len(parts))]
        except ValueError:
            continue
        if involved and all(s in serials for s in involved):
            conect_lines.append(line)
    return "\n".join(atom_lines + conect_lines) + "\n"


def _rmsd_heavy_aligned_angstrom(p_xyz: np.ndarray, q_xyz: np.ndarray) -> float:
    """Centroid-removed RMSD (Å) for two N×3 coordinate sets of equal length."""
    p = np.asarray(p_xyz, dtype=np.float64)
    q = np.asarray(q_xyz, dtype=np.float64)
    p0 = p - p.mean(axis=0, keepdims=True)
    q0 = q - q.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean((p0 - q0) ** 2) + 1e-24))


def _rdkit_ligand_positions_nm_from_pdb(
    pdb_path: Path,
    chain_id: str,
    mol_off: object,
    lig_topo: object,
    conf_nm: np.ndarray,
) -> list | None:
    """Map OpenFF/OpenMM ligand atoms to PDB coordinates via RDKit substructure match.

    Returns a list of ``openmm.Vec3 * nanometer`` in ``lig_topo`` atom order, or
    ``None`` if RDKit cannot align the PDB ligand fragment to the OpenFF graph.
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        return None

    import openmm
    from openmm import unit as mm_unit

    block = _extract_pdb_chain_block(pdb_path, chain_id)
    if not block.strip():
        return None

    pdb_mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if pdb_mol is None or pdb_mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(pdb_mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                pdb_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            return None

    rd_ref = mol_off.to_rdkit()
    try:
        Chem.SanitizeMol(rd_ref)
    except Exception:
        pass

    pdb_nh = Chem.RemoveHs(pdb_mol, implicitOnly=False)
    rd_nh = Chem.RemoveHs(rd_ref, implicitOnly=False)
    if pdb_nh.GetNumAtoms() == 0 or rd_nh.GetNumAtoms() == 0:
        return None
    if pdb_nh.GetNumAtoms() != rd_nh.GetNumAtoms():
        logger.warning(
            "RDKit ligand map: heavy-atom count mismatch PDB=%d OpenFF=%d "
            "(chain '%s'); skipping graph map.",
            pdb_nh.GetNumAtoms(),
            rd_nh.GetNumAtoms(),
            chain_id,
        )
        return None

    matches = list(
        pdb_nh.GetSubstructMatches(
            rd_nh,
            uniquify=True,
            useChirality=True,
        )
    )
    if not matches:
        matches = list(
            pdb_nh.GetSubstructMatches(rd_nh, uniquify=True, useChirality=False)
        )
    if not matches:
        return None

    pdb_conf = pdb_nh.GetConformer()
    rd_conf = rd_nh.GetConformer()
    best_m: tuple[int, ...] | None = None
    best_score = float("inf")
    for m in matches:
        p_xyz = np.zeros((len(m), 3), dtype=np.float64)
        q_xyz = np.zeros((len(m), 3), dtype=np.float64)
        for k in range(len(m)):
            pp = pdb_conf.GetAtomPosition(int(m[k]))
            qq = rd_conf.GetAtomPosition(k)
            p_xyz[k] = (float(pp.x), float(pp.y), float(pp.z))
            q_xyz[k] = (float(qq.x), float(qq.y), float(qq.z))
        score = _rmsd_heavy_aligned_angstrom(p_xyz, q_xyz)
        if score < best_score:
            best_score = score
            best_m = m
    if best_m is None:
        return None

    nh_to_rd_full: list[int] = []
    for a in rd_ref.GetAtoms():
        if a.GetAtomicNum() > 1:
            nh_to_rd_full.append(int(a.GetIdx()))
    if len(nh_to_rd_full) != rd_nh.GetNumAtoms():
        return None

    atoms_omm = list(lig_topo.atoms())
    conf_nm = np.asarray(conf_nm, dtype=np.float64)
    out_nm = np.zeros((len(atoms_omm), 3), dtype=np.float64)
    pdb_conf = pdb_nh.GetConformer()
    for i, atom in enumerate(atoms_omm):
        el = atom.element
        if el is None or el.symbol == "H":
            out_nm[i] = conf_nm[i]
            continue
        # Ligand-local atom index *i* matches OpenFF / RDKit parent mol atom index.
        try:
            j = nh_to_rd_full.index(i)
        except ValueError:
            out_nm[i] = conf_nm[i]
            continue
        p_at = pdb_conf.GetAtomPosition(int(best_m[j]))
        out_nm[i] = (float(p_at.x) * 0.1, float(p_at.y) * 0.1, float(p_at.z) * 0.1)

    lig_pos = [
        openmm.Vec3(float(out_nm[i, 0]), float(out_nm[i, 1]), float(out_nm[i, 2]))
        * mm_unit.nanometer
        for i in range(len(atoms_omm))
    ]
    logger.info(
        "RDKit ligand pose: chain '%s' mapped %d heavy atoms via substructure "
        "(alignment RMSD≈%.4f Å on matched heavy atoms).",
        chain_id,
        rd_nh.GetNumAtoms(),
        best_score,
    )
    return lig_pos


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
        from openff.toolkit import Molecule as OpenFFMolecule  # noqa: PLC0415
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
        gaff = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.11")
        ff.registerTemplateGenerator(gaff.generator)


# ---------------------------------------------------------------------------
# Persistent MD simulation (no minimisation) — shared prep with ``minimize_pdb``
# ---------------------------------------------------------------------------


def _ligand_openmm_positions_from_pdb_heavy_atoms(
    chain_id: str,
    orig_topology: object,
    orig_positions: object,
    lig_topo: object,
    conf_nm: np.ndarray,
    *,
    pdb_path: Path | None = None,
    mol_off: object | None = None,
    allow_gas_phase_fallback: bool = True,
) -> tuple[list, bool]:
    """Build OpenMM position list for a GAFF ligand, preferring PDB heavy-atom coords.

    OpenFF ``from_smiles`` + ``generate_conformers`` places the ligand in an
    arbitrary gas-phase pose (near the origin).  For protein–ligand complexes we
    first merge **heavy-atom** coordinates from the original PDB when atom
    **names** match OpenMM/OpenFF names.  If too few names match (Boltz/CCD
    labels vs RDKit labels), we fall back to an **RDKit** substructure map between
    the PDB ligand fragment (with ``CONECT``) and the OpenFF molecule graph;
    hydrogens always keep the OpenFF conformer coordinates.

    Parameters
    ----------
    chain_id
        PDB chain identifier for the ligand.
    orig_topology, orig_positions
        Topology and positions from the initial ``PDBFile`` read (before strip).
    lig_topo
        OpenMM ``Topology`` from ``OpenFFMolecule.to_topology().to_openmm()``.
    conf_nm
        Conformer coordinates in **nanometres**, shape ``(n_atoms, 3)``, same atom
        order as *lig_topo*.
    pdb_path, mol_off
        When both are given and name-based matching is insufficient, RDKit graph
        alignment is attempted against the ligand ``HETATM``/``CONECT`` block for
        *chain_id* in *pdb_path*.
    allow_gas_phase_fallback
        If ``False``, raises ``RuntimeError`` when PDB name merge and RDKit graph
        mapping both fail, instead of falling back to the OpenFF gas-phase pose.

    Returns
    -------
    positions, used_pdb_merge
        ``positions`` is a list of ``openmm.Vec3 * nanometer`` in lig_topo atom
        order.  *used_pdb_merge* is True when the pose is trusted from PDB-heavy
        placement (name merge or RDKit map); otherwise a pure gas-phase conformer
        list is returned.
    """
    import openmm
    from openmm import unit as mm_unit

    conf_nm = np.asarray(conf_nm, dtype=np.float64)
    atoms_omm = list(lig_topo.atoms())
    if conf_nm.shape[0] != len(atoms_omm):
        raise ValueError(
            "Conformer atom count does not match OpenMM ligand topology: "
            f"{conf_nm.shape[0]} vs {len(atoms_omm)}."
        )

    pdb_by_name: dict[str, object] = {}
    for chain in orig_topology.chains():
        if chain.id.strip() != chain_id.strip():
            continue
        for atom in chain.atoms():
            if atom.element is None or atom.element.symbol == "H":
                continue
            pdb_by_name[atom.name.strip().upper()] = orig_positions[atom.index]
        break

    n_omm_heavy = sum(
        1
        for a in atoms_omm
        if a.element is not None and a.element.symbol != "H"
    )
    lig_pos: list = []
    n_matched = 0
    for i, atom in enumerate(atoms_omm):
        el = atom.element
        is_h = el is None or el.symbol == "H"
        name_key = atom.name.strip().upper()
        if (not is_h) and name_key in pdb_by_name:
            lig_pos.append(pdb_by_name[name_key])
            n_matched += 1
        else:
            lig_pos.append(
                openmm.Vec3(
                    float(conf_nm[i, 0]),
                    float(conf_nm[i, 1]),
                    float(conf_nm[i, 2]),
                )
                * mm_unit.nanometer
            )

    min_match = max(1, (min(len(pdb_by_name), n_omm_heavy) + 1) // 2)
    used_pdb_merge = n_matched >= min_match
    if used_pdb_merge:
        return lig_pos, used_pdb_merge

    if pdb_path is not None and mol_off is not None:
        rdk_pos = _rdkit_ligand_positions_nm_from_pdb(
            pdb_path, chain_id, mol_off, lig_topo, conf_nm
        )
        if rdk_pos is not None:
            return rdk_pos, True

    if not used_pdb_merge and n_matched > 0:
        merge_msg = (
            f"{chain_id}: only {n_matched}/{n_omm_heavy} heavy atoms matched PDB "
            f"names (need >= {min_match}); would use gas-phase conformer for all atoms."
        )
        if not allow_gas_phase_fallback:
            raise RuntimeError(merge_msg)
        warnings.warn(merge_msg, RuntimeWarning, stacklevel=3)
        lig_pos = [
            openmm.Vec3(
                float(conf_nm[i, 0]),
                float(conf_nm[i, 1]),
                float(conf_nm[i, 2]),
            )
            * mm_unit.nanometer
            for i in range(len(atoms_omm))
        ]
    elif not used_pdb_merge:
        if len(pdb_by_name) == 0:
            merge_msg = (
                f"{chain_id}: no PDB heavy-atom rows for this chain in the input "
                "topology; using gas-phase conformer."
            )
        else:
            merge_msg = (
                f"{chain_id}: PDB has {len(pdb_by_name)} heavy-atom names but none "
                "matched OpenFF/OpenMM ligand atom names; using gas-phase conformer."
            )
        if not allow_gas_phase_fallback:
            raise RuntimeError(merge_msg)
        warnings.warn(merge_msg, RuntimeWarning, stacklevel=3)
        lig_pos = [
            openmm.Vec3(
                float(conf_nm[i, 0]),
                float(conf_nm[i, 1]),
                float(conf_nm[i, 2]),
            )
            * mm_unit.nanometer
            for i in range(len(atoms_omm))
        ]

    return lig_pos, False


def build_md_simulation_from_pdb(
    pdb_path: Path,
    *,
    platform_name: str = "CUDA",
    temperature_k: float = 300.0,
    ligand_smiles: Optional[dict[str, str]] = None,
    extra_forces: Optional[Any] = None,
    boltz_structure: Any = None,
    boltz_coords_angstrom: np.ndarray | None = None,
    boltz_mol_dir: Path | None = None,
    ligand_pose_policy: LigandPosePolicy = "pdb_only",
    ligand_pose_debug_sdf_dir: Path | None = None,
    platform_properties: dict[str, str] | None = None,
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
    platform_properties:
        Optional map passed as ``platformProperties`` to :class:`openmm.app.Simulation``
        (e.g. ``{"DeviceIndex": "1"}`` for CUDA/OpenCL). CUDA runs also merge
        ``Precision: mixed`` when any property is set.
    temperature_k:
        Langevin bath temperature (K).
    ligand_smiles:
        Optional chain ID → SMILES mapping for cofactors (same as ``minimize_pdb``).
    extra_forces:
        Optional OpenMM ``Force`` objects to add before the ``Simulation`` is
        created.  May be an iterable of forces or a callable
        ``f(system, topology, positions, meta)`` that either adds forces itself
        and returns ``None`` or returns an iterable of forces to add.
    boltz_structure, boltz_coords_angstrom, boltz_mol_dir:
        When all three are set, :func:`try_ligand_pose_from_boltz_ccd` can place
        each ligand chain from Boltz NPZ coordinates and the CCD ``.pkl`` cache.
    ligand_pose_policy:
        ``boltz_first`` tries Boltz+CCD then PDB merge / RDKit-PDB (no gas-phase
        fallback); ``strict`` uses only Boltz+CCD; ``pdb_only`` keeps the legacy
        path including gas-phase fallback.
    ligand_pose_debug_sdf_dir:
        Optional directory for per-chain debug SDF files after Boltz placement.

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
    meta["ligand_pose_policy"] = ligand_pose_policy

    sim_props = dict(platform_properties or {})
    platform, actual_platform = _get_platform(
        platform_name,
        runtime_smoke_properties=sim_props if sim_props else None,
    )
    meta["platform_used"] = actual_platform
    if actual_platform == "CUDA":
        sim_props.setdefault("Precision", "mixed")
    elif actual_platform not in ("CUDA", "OpenCL"):
        # Avoid passing GPU-only properties to Reference/CPU platforms.
        sim_props.pop("DeviceIndex", None)
        sim_props.pop("Precision", None)

    boltz_ctx_ok = (
        boltz_structure is not None
        and boltz_coords_angstrom is not None
        and boltz_mol_dir is not None
    )
    allow_gas_phase = ligand_pose_policy == "pdb_only"

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
                        from openff.toolkit import Molecule as OpenFFMolecule  # noqa: PLC0415
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
                        ion_chain = ion_topo.addChain(id=chain_id.strip())
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
                    lig_topo = _ligand_topology_relabeled_chain(
                        mol.to_topology().to_openmm(), chain_id
                    )
                    conf_ang = mol.conformers[0].magnitude
                    conf_nm = conf_ang / 10.0
                    lig_pos: list | None = None
                    if boltz_ctx_ok and ligand_pose_policy in (
                        "boltz_first",
                        "strict",
                    ):
                        try:
                            bolted = try_ligand_pose_from_boltz_ccd(
                                structure=boltz_structure,
                                coords_angstrom=np.asarray(
                                    boltz_coords_angstrom, dtype=np.float64
                                ),
                                mol_dir=Path(boltz_mol_dir),
                                chain_id=chain_id,
                                smiles=smiles,
                                debug_sdf_dir=ligand_pose_debug_sdf_dir,
                            )
                            if bolted is not None:
                                lig_pos, lig_topo = bolted
                        except Exception as exc:
                            if ligand_pose_policy == "strict":
                                raise
                            logger.warning(
                                "Boltz+CCD ligand pose failed (chain=%s): %s — "
                                "falling back to PDB merge / RDKit.",
                                chain_id,
                                exc,
                            )
                    if lig_pos is None:
                        if ligand_pose_policy == "strict":
                            raise RuntimeError(
                                f"Chain '{chain_id}': ligand_pose_policy=strict requires "
                                "Boltz+CCD placement; chain missing from NONPOLYMER "
                                "topology or pose failed."
                            )
                        lig_pos, _pdb_pose = _ligand_openmm_positions_from_pdb_heavy_atoms(
                            chain_id,
                            orig_topology,
                            orig_positions,
                            lig_topo,
                            conf_nm,
                            pdb_path=pdb_path,
                            mol_off=mol,
                            allow_gas_phase_fallback=allow_gas_phase,
                        )
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
    if extra_forces is not None:
        if callable(extra_forces):
            maybe_forces = extra_forces(system, h_topology, h_positions, meta)
        else:
            maybe_forces = extra_forces
        if maybe_forces is not None:
            for force in maybe_forces:
                system.addForce(force)

    integrator = openmm.LangevinIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )

    if sim_props:
        sim = openmm.app.Simulation(
            h_topology,
            system,
            integrator,
            platform,
            platformProperties=sim_props,
        )
    else:
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
    platform_properties: dict[str, str] | None = None,
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
    then re-inserts each ligand.  Multi-atom ligands use OpenFF for topology and
    GAFF parameters, but **heavy-atom coordinates are merged from the original
    PDB** when atom names match, or via **RDKit** substructure mapping when names
    differ (hydrogens use the OpenFF conformer).  Monoatomic
    tip3p ions use the
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
    platform_properties:
        Optional OpenMM ``platformProperties`` (e.g. ``DeviceIndex`` for CUDA).
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
            platform_properties=platform_properties,
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

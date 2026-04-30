"""Map Boltz ``StructureV2`` atoms to OpenMM topologies built from Boltz-exported PDBs.

Includes ligand placement from Boltz NPZ coordinates plus CCD ``.pkl`` RDKit
templates (``try_ligand_pose_from_boltz_ccd``), used when building systems from
:func:`scripts.compute_cv_rmsd.build_md_simulation_from_pdb`.

Used by :mod:`genai_tps.simulation.openmm_md_runner` without optional RL-only dependencies.
"""

from __future__ import annotations

import importlib.util
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

__all__ = [
    "LigandPosePolicy",
    "boltz_to_plumed_indices",
    "build_openmm_indices_for_boltz_atoms",
    "ligand_topology_relabeled_chain",
    "load_build_md_simulation_from_pdb",
    "try_ligand_pose_from_boltz_ccd",
]

LigandPosePolicy = Literal["boltz_first", "pdb_only", "strict"]


def load_build_md_simulation_from_pdb() -> Callable[..., tuple[Any, dict]]:
    """Return :func:`build_md_simulation_from_pdb` from ``scripts/compute_cv_rmsd.py``."""

    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    script_path = repo_root / "scripts" / "compute_cv_rmsd.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Expected OpenMM helper script at {script_path}")
    spec = importlib.util.spec_from_file_location("compute_cv_rmsd", script_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load compute_cv_rmsd module spec.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "build_md_simulation_from_pdb", None)
    if fn is None:
        raise AttributeError("compute_cv_rmsd.build_md_simulation_from_pdb missing.")
    return fn


def _openmm_residue_sequence_number(residue: Any) -> int:
    rid = residue.id
    if isinstance(rid, tuple):
        return int(rid[0])
    return int(rid)


def _boltz_atom_pdb_key(structure: Any, atom_idx: int) -> tuple[str, int, str]:
    """Return ``(chain_name, pdb_residue_number, atom_name)`` for a Boltz atom index."""
    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains
    for chain in chains:
        a0 = int(chain["atom_idx"])
        a1 = a0 + int(chain["atom_num"])
        if not (a0 <= atom_idx < a1):
            continue
        chain_name = str(chain["name"]).strip()
        rs = int(chain["res_idx"])
        re_end = rs + int(chain["res_num"])
        for ri in range(rs, re_end):
            res = residues[ri]
            r0 = int(res["atom_idx"])
            r1 = r0 + int(res["atom_num"])
            if r0 <= atom_idx < r1:
                pdb_resnum = int(res["res_idx"]) + 1  # matches Boltz ``to_pdb``
                aname = str(atoms[atom_idx]["name"]).strip().upper()
                return chain_name, pdb_resnum, aname
    raise ValueError(f"atom_idx {atom_idx} not found in Boltz structure.")


def boltz_to_plumed_indices(
    boltz_idx_array: Any,
    omm_idx_map: np.ndarray,
) -> list[int]:
    """Convert Boltz-order atom indices to PLUMED 1-based OpenMM atom indices.

    Parameters
    ----------
    boltz_idx_array:
        Iterable of Boltz atom indices (0-based, heavy-atom order).
    omm_idx_map:
        Array mapping every Boltz atom index to the corresponding OpenMM
        particle index (0-based).

    Returns
    -------
    list[int]
        PLUMED atom indices, i.e. OpenMM particle indices plus one.
    """
    boltz_indices = np.asarray(boltz_idx_array, dtype=np.int64)
    omm_indices = np.asarray(omm_idx_map, dtype=np.int64)[boltz_indices]
    if np.any(omm_indices < 0):
        bad = boltz_indices[np.where(omm_indices < 0)[0]]
        raise ValueError(
            "Cannot convert Boltz atoms with negative OpenMM atom index "
            f"to PLUMED indices; bad Boltz indices: {bad.tolist()}"
        )
    return [int(i) + 1 for i in omm_indices]


def build_openmm_indices_for_boltz_atoms(
    structure: Any,
    h_topology: Any,
    *,
    ref_coords_angstrom: np.ndarray | None = None,
    omm_positions_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Map each Boltz heavy-atom index to an OpenMM atom index in *h_topology*.

    Primary matching uses PDB semantics ``(chain_id, residue_sequence_number, atom_name)``
    for non-hydrogen atoms.

    When names disagree (e.g. Boltz CCD names vs OpenFF ligand atom names after
    :func:`build_md_simulation_from_pdb`), pass *ref_coords_angstrom* (Boltz-order
    Å) and *omm_positions_nm* (OpenMM ``N × 3`` positions in nm).  Unmatched
    Boltz atoms are paired with leftover OpenMM heavy atoms by **minimum total
    Euclidean distance** via :func:`scipy.optimize.linear_sum_assignment`
    (Hungarian / optimal linear assignment).
    """
    n = int(structure.atoms.shape[0])
    openmm_keys_to_idx: dict[tuple[str, int, str], int] = {}
    for atom in h_topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        chain_id = atom.residue.chain.id.strip()
        seq = _openmm_residue_sequence_number(atom.residue)
        name = atom.name.strip().upper()
        key = (chain_id, seq, name)
        if key in openmm_keys_to_idx:
            raise ValueError(f"Duplicate OpenMM heavy-atom key: {key}")
        openmm_keys_to_idx[key] = int(atom.index)

    out = np.empty(n, dtype=np.int64)
    missing: list[tuple[int, tuple[str, int, str]]] = []
    for i in range(n):
        key = _boltz_atom_pdb_key(structure, i)
        j = openmm_keys_to_idx.get(key)
        if j is None:
            missing.append((i, key))
            out[i] = -1
        else:
            out[i] = j
    if not missing:
        return out

    if ref_coords_angstrom is None or omm_positions_nm is None:
        raise ValueError(
            "Could not map some Boltz atoms to OpenMM by (chain, resnum, name); "
            f"first misses: {missing[:5]!r}. "
            "Pass ref_coords_angstrom and omm_positions_nm to enable geometric "
            "fallback (Hungarian assignment on 3D distances)."
        )

    assigned_omm = {int(out[i]) for i in range(n) if out[i] >= 0}
    miss_chains = {str(m[1][0]).strip() for m in missing}
    omm_pool: list[int] = []
    for atom in h_topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        j = int(atom.index)
        if j in assigned_omm:
            continue
        if atom.residue.chain.id.strip() not in miss_chains:
            continue
        omm_pool.append(j)

    # Boltz PDB keys use the YAML chain id (e.g. ``B``), but OpenFF→OpenMM may
    # emit a different chain id for the GAFF ligand (sometimes empty).  A strict
    # chain filter then yields an empty pool and geometric matching cannot run.
    if not omm_pool:
        omm_pool = [
            int(atom.index)
            for atom in h_topology.atoms()
            if atom.element is not None
            and atom.element.symbol != "H"
            and int(atom.index) not in assigned_omm
        ]
        print(
            "[Boltz-OmmMap] Chain-filtered pool empty for geometric fallback; "
            f"using all {len(omm_pool)} unassigned heavy atoms (Boltz chains "
            f"{sorted(miss_chains)!r}).",
            flush=True,
        )

    miss_idx = np.array([m[0] for m in missing], dtype=np.int64)
    bolt_pts = np.asarray(ref_coords_angstrom, dtype=np.float64)[miss_idx]
    omm_ang = np.asarray(omm_positions_nm, dtype=np.float64) * 10.0
    pool_idx = np.array(omm_pool, dtype=np.int64)
    cand = omm_ang[pool_idx]

    n_bolt_m = bolt_pts.shape[0]
    n_omm_m = cand.shape[0]
    if n_bolt_m > n_omm_m:
        raise ValueError(
            "Geometric fallback: more unmatched Boltz atoms than OpenMM pool "
            f"({n_bolt_m} vs {n_omm_m}); first name misses: {missing[:5]!r}."
        )

    diff = bolt_pts[:, None, :] - cand[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.shape[0] != n_bolt_m:
        raise ValueError(
            f"Hungarian matching returned {row_ind.shape[0]} pairs, expected {n_bolt_m}."
        )
    assigned_dists: list[float] = []
    for ri, ci in zip(row_ind, col_ind, strict=True):
        assigned_dists.append(float(cost[ri, ci]))
        out[miss_idx[ri]] = int(pool_idx[ci])

    if assigned_dists:
        print(
            "[Boltz-OmmMap] Hungarian fallback: "
            f"n={len(assigned_dists)} pairs | "
            f"dist_Å mean={float(np.mean(assigned_dists)):.3f} "
            f"max={float(np.max(assigned_dists)):.3f}",
            flush=True,
        )

    return out


# ---------------------------------------------------------------------------
# Ligand pose: Boltz StructureV2 + CCD RDKit pickles (Boltz mol cache)
# ---------------------------------------------------------------------------


def ligand_topology_relabeled_chain(lig_topo: object, chain_id: str) -> object:
    """Copy ligand ``Topology`` so the sole chain id matches YAML/PDB *chain_id*."""
    import openmm.app as app

    cid = chain_id.strip()
    new_top = app.Topology()
    nch = new_top.addChain(id=cid)
    old_to_new: dict = {}
    for old_ch in lig_topo.chains():
        for old_res in old_ch.residues():
            new_res = new_top.addResidue(old_res.name, nch)
            for old_atom in old_res.atoms():
                na = new_top.addAtom(old_atom.name, old_atom.element, new_res)
                old_to_new[old_atom] = na
    for bond in lig_topo.bonds():
        a1, a2 = bond
        if a1 in old_to_new and a2 in old_to_new:
            new_top.addBond(old_to_new[a1], old_to_new[a2])
    return new_top


def _find_nonpolymer_chain(structure: Any, chain_id: str) -> Any | None:
    """Return the ``chains`` row for *chain_id* if it is NONPOLYMER, else ``None``."""
    from boltz.data import const  # noqa: PLC0415

    want = chain_id.strip()
    nonpoly = int(const.chain_type_ids["NONPOLYMER"])
    for ch in structure.chains:
        if int(ch["mol_type"]) != nonpoly:
            continue
        if str(ch["name"]).strip() == want:
            return ch
    return None


def try_ligand_pose_from_boltz_ccd(
    *,
    structure: Any,
    coords_angstrom: np.ndarray,
    mol_dir: Path,
    chain_id: str,
    smiles: str,
    debug_sdf_dir: Path | None = None,
) -> tuple[list, Any] | None:
    """Place ligand atoms from Boltz coords + CCD RDKit template.

    Coordinates come from the same NPZ frame used to write ``initial_structure.pdb``.
    Chemistry for name matching follows Boltz featurisation: RDKit atoms in
    ``{mol_dir}/{CCD}.pkl`` often expose ``GetProp("name")`` aligned with
    ``structure.atoms["name"]``.

    OpenFF :class:`~openff.toolkit.Molecule` is built via ``from_rdkit`` so the
    ligand graph and 3D pose share one RDKit object, avoiding fragile PDB atom-name
    alignment against SMILES-generated OpenMM atom names.

    Parameters
    ----------
    structure
        Boltz ``StructureV2`` (``load_topo`` return value).
    coords_angstrom
        Shape ``(N_atom, 3)`` Å, same global indexing as *structure.atoms*.
    mol_dir
        Boltz CCD cache directory (``~/.boltz/mols``).
    chain_id
        PDB / YAML chain identifier (e.g. ``"B"``).
    smiles
        SMILES string used for GAFF / OpenFF registration (must be isomorphic
        with the CCD RDKit template after coordinate assignment).
    debug_sdf_dir
        If set, writes ``ligand_{chain_id}.sdf`` for inspection.

    Returns
    -------
    (positions, lig_openmm_topology) | None
        ``None`` if *chain_id* is not a NONPOLYMER chain in *structure*.
        Otherwise returns OpenMM position list (nm) and hydrogenated ligand
        topology for :class:`openmm.app.Modeller.add`.

    Raises
    ------
    FileNotFoundError
        If the CCD pickle is missing.
    ValueError
        If the chain has multiple residues (unsupported in v1).
    RuntimeError
        If heavy-atom names cannot be matched or OpenFF graphs disagree.
    """
    import openmm
    from openmm import unit as mm_unit
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    chain = _find_nonpolymer_chain(structure, chain_id)
    if chain is None:
        return None

    if int(chain["res_num"]) != 1:
        raise ValueError(
            f"Ligand chain '{chain_id}': res_num={int(chain['res_num'])} — "
            "multi-residue NONPOLYMER chains are not supported for Boltz CCD pose."
        )

    res_idx = int(chain["res_idx"])
    ccd = str(structure.residues[res_idx]["name"]).strip()
    pkl_path = mol_dir / f"{ccd}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Boltz CCD pickle not found for chain '{chain_id}' (CCD={ccd!r}): {pkl_path}"
        )

    with pkl_path.open("rb") as fh:
        template_rd: Chem.Mol = pickle.load(fh)  # noqa: S301 — trusted local cache

    rd = Chem.Mol(template_rd)
    start = int(chain["atom_idx"])
    end = start + int(chain["atom_num"])
    rd_heavy_idx = sorted(
        int(a.GetIdx()) for a in rd.GetAtoms() if int(a.GetAtomicNum()) > 1
    )
    if len(rd_heavy_idx) != end - start:
        raise RuntimeError(
            f"Boltz CCD pose: heavy-atom count mismatch RDKit={len(rd_heavy_idx)} "
            f"Boltz chain={end - start} (CCD={ccd}, chain={chain_id})."
        )
    need_name_bootstrap = any(
        not rd.GetAtomWithIdx(int(i)).HasProp("name") for i in rd_heavy_idx
    )
    if need_name_bootstrap:
        for g, ri in zip(range(start, end), rd_heavy_idx, strict=True):
            nm = str(structure.atoms[g]["name"]).strip()
            rd.GetAtomWithIdx(int(ri)).SetProp("name", nm)
        logger.info(
            "Boltz CCD pose: restored per-atom 'name' props on RDKit mol after load "
            "(pickle dropped them); mapped Boltz chain order → RDKit heavy indices "
            "%s (CCD=%s).",
            rd_heavy_idx,
            ccd,
        )

    if rd.GetNumConformers() == 0:
        h = AllChem.EmbedMolecule(rd, randomSeed=0xF00D)
        if h < 0:
            AllChem.Compute2DCoords(rd)

    conf = rd.GetConformer(0)
    coords = np.asarray(coords_angstrom, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_angstrom must be (N, 3), got {coords.shape}")

    boltz_names: list[str] = []
    for g in range(start, end):
        boltz_names.append(str(structure.atoms[g]["name"]).strip().upper())

    # Map PDB/CCD atom name -> global Boltz index (first occurrence in chain slice)
    name_to_global: dict[str, int] = {}
    for g, nm in enumerate(boltz_names, start=start):
        if nm not in name_to_global:
            name_to_global[nm] = g

    n_assigned_heavy = 0
    n_heavy_rd = 0
    for atom in rd.GetAtoms():
        z = int(atom.GetAtomicNum())
        if z <= 1:
            continue
        n_heavy_rd += 1
        key = None
        if atom.HasProp("name"):
            key = str(atom.GetProp("name")).strip().upper()
        if not key:
            raise RuntimeError(
                f"Boltz CCD pose: RDKit heavy atom index {atom.GetIdx()} has no "
                "'name' property; cannot map to Boltz atoms (CCD={ccd})."
            )
        if key not in name_to_global:
            raise RuntimeError(
                f"Boltz CCD pose: RDKit atom name {key!r} not found in Boltz chain "
                f"'{chain_id}' (CCD={ccd}). Boltz names (sample): {boltz_names[:8]}…"
            )
        gi = name_to_global[key]
        x, y, zc = (float(coords[gi, j]) for j in range(3))
        conf.SetAtomPosition(atom.GetIdx(), (x, y, zc))
        n_assigned_heavy += 1

    if n_heavy_rd == 0:
        raise RuntimeError(f"Boltz CCD pose: RDKit template has no heavy atoms (CCD={ccd}).")

    if n_assigned_heavy != n_heavy_rd:
        raise RuntimeError(
            f"Boltz CCD pose: internal mismatch assigned_heavy={n_assigned_heavy} "
            f"rd_heavy={n_heavy_rd} (CCD={ccd})."
        )

    try:
        from openff.toolkit import Molecule as OpenFFMolecule  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openff-toolkit is required for Boltz CCD ligand placement."
        ) from exc

    mol_placed = OpenFFMolecule.from_rdkit(rd, allow_undefined_stereo=True)
    mol_ref = OpenFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
    if not mol_placed.is_isomorphic_with(mol_ref):
        raise RuntimeError(
            f"Boltz CCD pose: OpenFF Molecule from RDKit (CCD={ccd}) is not "
            f"isomorphic with Molecule.from_smiles for chain '{chain_id}'. "
            "Check CCD pickle vs SMILES protonation/tautomer."
        )

    if debug_sdf_dir is not None:
        debug_sdf_dir.mkdir(parents=True, exist_ok=True)
        out_sdf = debug_sdf_dir / f"ligand_{chain_id.strip()}.sdf"
        try:
            w = Chem.SDWriter(str(out_sdf))
            w.write(rd)
            w.close()
            logger.info("Wrote ligand debug SDF: %s", out_sdf)
        except Exception as exc:
            logger.warning("Could not write debug SDF %s: %s", out_sdf, exc)

    lig_omm = mol_placed.to_topology().to_openmm()
    lig_topo = ligand_topology_relabeled_chain(lig_omm, chain_id)
    conf_q = mol_placed.conformers[0]
    conf_nm = np.asarray(conf_q.magnitude, dtype=np.float64) / 10.0

    lig_pos = [
        openmm.Vec3(float(conf_nm[i, 0]), float(conf_nm[i, 1]), float(conf_nm[i, 2]))
        * mm_unit.nanometer
        for i in range(conf_nm.shape[0])
    ]
    logger.info(
        "Boltz CCD ligand pose: chain=%s CCD=%s heavy_atoms=%d (from_rdkit + isomorphic)",
        chain_id,
        ccd,
        n_heavy_rd,
    )
    return lig_pos, lig_topo

"""Map Boltz ``StructureV2`` atoms to OpenMM topologies built from Boltz-exported PDBs.

Used by :mod:`genai_tps.simulation.openmm_md_runner` without depending on the
optional ``genai_tps.rl`` package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "boltz_to_plumed_indices",
    "build_openmm_indices_for_boltz_atoms",
    "load_build_md_simulation_from_pdb",
]


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

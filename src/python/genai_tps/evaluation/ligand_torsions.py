"""Ligand rotatable torsion angle analysis for ensemble quality evaluation.

Uses RDKit to identify rotatable bonds in a ligand and compute dihedral angles
from a trajectory of conformers.  The resulting per-bond distributions are
compared via the Jensen-Shannon distance (see ``distribution_metrics.py``).

Matches the torsion analysis protocol of Wang et al. 2026 (AnewSampling,
bioRxiv 10.64898/2026.03.10.710952, Appendix A.1).

Usage example
-------------
>>> from rdkit import Chem
>>> from genai_tps.evaluation.ligand_torsions import ligand_torsion_js_summary
>>> mol = Chem.MolFromSmiles("CCC(=O)c1ccccc1")
>>> # coords_pred and coords_ref: (T_pred, N_heavy, 3) and (T_ref, N_heavy, 3)
>>> summary = ligand_torsion_js_summary(mol, coords_pred, coords_ref)
>>> print(summary["mean_js"])

Dependencies
------------
- ``rdkit`` — install via conda-forge: ``conda install -c conda-forge rdkit``
- ``numpy``
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from genai_tps.evaluation.distribution_metrics import torsion_js_distance

__all__ = [
    "enumerate_rotatable_dihedrals",
    "extract_dihedral_trajectory",
    "ligand_torsion_js_summary",
]

# SMARTS for rotatable bonds following the AnewSampling / Lipinski convention:
# single non-ring bond between two heavy atoms that each have > 1 heavy neighbour
_ROTATABLE_BOND_SMARTS = "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"


def enumerate_rotatable_dihedrals(
    mol: Chem.Mol,
) -> list[tuple[int, int, int, int]]:
    """Return all unique rotatable dihedral atom quartets (i, j, k, l).

    Identifies rotatable bonds via SMARTS and selects one dihedral quartet
    per bond by choosing the first heavy-atom neighbour outside the bond
    for each end.

    Parameters
    ----------
    mol:
        RDKit molecule.  Must have a conformer (for 3-D use), though this
        function only needs topology (not coordinates).

    Returns
    -------
    list of (i, j, k, l) tuples
        Each tuple is a 4-atom path defining a torsion angle.  The central
        bond is j-k.  Indices are 0-based RDKit atom indices.
    """
    mol = Chem.RemoveHs(mol)                # work on heavy atoms only
    pattern = Chem.MolFromSmarts(_ROTATABLE_BOND_SMARTS)
    matches = mol.GetSubstructMatches(pattern)

    quartets: list[tuple[int, int, int, int]] = []
    seen_bonds: set[frozenset] = set()

    for j, k in matches:
        bond_key = frozenset((j, k))
        if bond_key in seen_bonds:
            continue
        seen_bonds.add(bond_key)

        # Choose first heavy-atom neighbour of j that is not k
        i = _first_neighbor_except(mol, j, k)
        # Choose first heavy-atom neighbour of k that is not j
        l = _first_neighbor_except(mol, k, j)

        if i is not None and l is not None:
            quartets.append((i, j, k, l))

    return quartets


def _first_neighbor_except(
    mol: Chem.Mol,
    atom_idx: int,
    exclude_idx: int,
) -> int | None:
    """Return the first heavy-atom neighbour of atom_idx excluding exclude_idx."""
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != exclude_idx:
            return nbr.GetIdx()
    return None


def extract_dihedral_trajectory(
    coords: np.ndarray,
    quartet: tuple[int, int, int, int],
) -> np.ndarray:
    """Compute dihedral angles for a single bond quartet across a trajectory.

    Uses the IUPAC definition (Prelog convention): the angle is defined by
    the planes formed by (i, j, k) and (j, k, l).

    Parameters
    ----------
    coords:
        (T, N, 3) array of Cartesian coordinates.  Atoms must be indexed
        consistently with the RDKit molecule (heavy atoms only).
    quartet:
        (i, j, k, l) atom indices defining the torsion.

    Returns
    -------
    angles : (T,) array in degrees, range (-180, 180].
    """
    i, j, k, l = quartet
    r1 = coords[:, i, :] - coords[:, j, :]  # (T, 3)
    r2 = coords[:, j, :] - coords[:, k, :]  # (T, 3)
    r3 = coords[:, k, :] - coords[:, l, :]  # (T, 3) -- note: inward

    # Normal vectors to the two planes
    n1 = np.cross(r1, r2)   # (T, 3)
    n2 = np.cross(r2, r3)   # (T, 3) -- note sign matches r3 direction

    norm_n1 = np.linalg.norm(n1, axis=-1, keepdims=True)
    norm_n2 = np.linalg.norm(n2, axis=-1, keepdims=True)

    # Avoid division by zero for degenerate geometries
    valid = (norm_n1 > 1e-8) & (norm_n2 > 1e-8)
    norm_n1 = np.where(valid, norm_n1, 1.0)
    norm_n2 = np.where(valid, norm_n2, 1.0)

    n1_hat = n1 / norm_n1
    n2_hat = n2 / norm_n2

    cos_theta = (n1_hat * n2_hat).sum(axis=-1).clip(-1.0, 1.0)

    # Sign convention: positive if n1 x n2 is aligned with b2 (the central bond j→k)
    # Note: r2 = atom_j - atom_k = -b2, so b2_hat = -r2_hat
    r2_hat = r2 / np.linalg.norm(r2, axis=-1, keepdims=True).clip(min=1e-8)
    cross_n = np.cross(n1_hat, n2_hat)
    sign = np.sign(-(cross_n * r2_hat).sum(axis=-1))  # negate because r2 = -b2
    sign = np.where(sign == 0, 1.0, sign)

    angles_rad = sign * np.arccos(cos_theta)
    return np.degrees(angles_rad)


def ligand_torsion_js_summary(
    mol: Chem.Mol,
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    n_bins: int = 36,
) -> dict:
    """Compute per-bond JS distances between predicted and reference torsion distributions.

    Parameters
    ----------
    mol:
        RDKit molecule defining the bond topology.  Should be the same
        heavy-atom topology as the coordinates.
    coords_pred:
        (T_pred, N, 3) predicted ensemble (e.g., from generative sampling).
    coords_ref:
        (T_ref, N, 3) reference ensemble (e.g., from MD simulation).
    n_bins:
        Histogram bins for the JS calculation (default 36 = 10° resolution).

    Returns
    -------
    dict with keys:
        - ``"per_bond_js"`` : list[float] -- JS distance for each rotatable bond
        - ``"n_bonds"``     : int         -- number of rotatable bonds found
        - ``"mean_js"``     : float       -- mean over all bonds (NaN if no bonds)
        - ``"median_js"``   : float       -- median over all bonds (NaN if no bonds)
        - ``"success_rate"``  : float     -- fraction of bonds with JS < 0.2
        - ``"quartets"``    : list of (i,j,k,l) tuples

    Raises
    ------
    ValueError
        If the coordinate arrays have incompatible atom counts.
    """
    if coords_pred.ndim != 3 or coords_ref.ndim != 3:
        raise ValueError("coords_pred and coords_ref must be (T, N, 3) arrays")
    if coords_pred.shape[1] != coords_ref.shape[1]:
        raise ValueError(
            f"Atom count mismatch: coords_pred has {coords_pred.shape[1]}, "
            f"coords_ref has {coords_ref.shape[1]}"
        )

    quartets = enumerate_rotatable_dihedrals(mol)

    if not quartets:
        return {
            "per_bond_js": [],
            "n_bonds": 0,
            "mean_js": float("nan"),
            "median_js": float("nan"),
            "success_rate": float("nan"),
            "quartets": [],
        }

    per_bond_js = []
    for q in quartets:
        angles_pred = extract_dihedral_trajectory(coords_pred, q)
        angles_ref = extract_dihedral_trajectory(coords_ref, q)
        js = torsion_js_distance(angles_pred, angles_ref, n_bins=n_bins)
        per_bond_js.append(js)

    arr = np.array(per_bond_js)
    return {
        "per_bond_js": per_bond_js,
        "n_bonds": len(quartets),
        "mean_js": float(arr.mean()),
        "median_js": float(np.median(arr)),
        "success_rate": float((arr < 0.2).mean()),
        "quartets": quartets,
    }

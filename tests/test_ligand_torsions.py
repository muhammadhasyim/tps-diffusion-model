"""Tests for genai_tps.evaluation.ligand_torsions.

Verified properties:
1. enumerate_rotatable_dihedrals: known molecule has expected bond count.
2. extract_dihedral_trajectory: known geometry gives exact dihedral angle.
3. ligand_torsion_js_summary: output keys, identical ensembles give JS~0,
   disjoint ensembles give JS~1, no-rotatable-bond molecule returns NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from genai_tps.evaluation.ligand_torsions import (
    enumerate_rotatable_dihedrals,
    extract_dihedral_trajectory,
    ligand_torsion_js_summary,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol(smiles: str) -> "Chem.Mol":
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return Chem.RemoveHs(mol)


def _random_coords(mol, T: int) -> np.ndarray:
    """Random (T, N_heavy, 3) coordinate array for mol."""
    n = mol.GetNumAtoms()
    return RNG.normal(0, 1, (T, n, 3))


# ---------------------------------------------------------------------------
# enumerate_rotatable_dihedrals
# ---------------------------------------------------------------------------

def test_enumerate_butane_has_one_rotatable_bond():
    """Butane (CC-CC) has exactly 1 rotatable bond."""
    mol = _mol("CCCC")
    quartets = enumerate_rotatable_dihedrals(mol)
    assert len(quartets) == 1, f"Expected 1 quartet, got {len(quartets)}"


def test_enumerate_ethane_has_zero_rotatable_bonds():
    """Ethane: both C atoms have degree 1 after RemoveHs, excluded by [!D1] SMARTS."""
    mol = _mol("CC")
    quartets = enumerate_rotatable_dihedrals(mol)
    assert len(quartets) == 0


def test_enumerate_benzene_has_no_rotatable_bonds():
    """Benzene has no rotatable bonds (all are ring bonds)."""
    mol = _mol("c1ccccc1")
    quartets = enumerate_rotatable_dihedrals(mol)
    assert len(quartets) == 0


def test_enumerate_returns_valid_atom_indices():
    """All quartet indices must be valid atom indices."""
    mol = _mol("CCCCCC")  # hexane -- 4 rotatable bonds
    quartets = enumerate_rotatable_dihedrals(mol)
    n_atoms = mol.GetNumAtoms()
    for q in quartets:
        assert len(q) == 4
        for idx in q:
            assert 0 <= idx < n_atoms, f"Atom index {idx} out of range [0, {n_atoms})"


def test_enumerate_no_duplicate_bonds():
    """Each central bond should appear at most once in the quartet list."""
    mol = _mol("CCCCCCCC")  # octane
    quartets = enumerate_rotatable_dihedrals(mol)
    central_bonds = {frozenset((q[1], q[2])) for q in quartets}
    assert len(central_bonds) == len(quartets), "Duplicate central bond in quartet list"


# ---------------------------------------------------------------------------
# extract_dihedral_trajectory
# ---------------------------------------------------------------------------

def _make_dihedral_coords(angle_deg: float) -> np.ndarray:
    """Create a single frame (1, 4, 3) with atoms 0-3 forming a specific dihedral."""
    phi = np.radians(angle_deg)
    # atom 1 and 2 on the x-axis; atom 0 in the xy-plane; atom 3 rotated by phi
    coords = np.array([
        [[-1.0,  1.0, 0.0],   # atom 0
         [ 0.0,  0.0, 0.0],   # atom 1
         [ 1.0,  0.0, 0.0],   # atom 2
         [ 2.0,  np.cos(phi), np.sin(phi)]],  # atom 3
    ])
    return coords  # (1, 4, 3)


@pytest.mark.parametrize("angle", [0.0, 60.0, 90.0, 120.0, 180.0, -90.0, -120.0])
def test_dihedral_trajectory_known_angles(angle):
    """extract_dihedral_trajectory recovers the correct angle for known geometries."""
    coords = _make_dihedral_coords(angle)
    quartet = (0, 1, 2, 3)
    computed = extract_dihedral_trajectory(coords, quartet)
    assert computed.shape == (1,)
    assert abs(computed[0] - angle) < 0.1, (
        f"Expected {angle}°, got {computed[0]:.2f}°"
    )


def test_dihedral_trajectory_shape():
    """Output shape is (T,)."""
    T, N = 50, 6
    coords = RNG.normal(0, 1, (T, N, 3))
    angles = extract_dihedral_trajectory(coords, (0, 1, 2, 3))
    assert angles.shape == (T,)


def test_dihedral_trajectory_range():
    """Angles must lie in (-180, 180]."""
    T, N = 100, 6
    coords = RNG.normal(0, 1, (T, N, 3))
    angles = extract_dihedral_trajectory(coords, (0, 1, 2, 3))
    assert np.all(angles > -180 - 1e-6)
    assert np.all(angles <= 180 + 1e-6)


# ---------------------------------------------------------------------------
# ligand_torsion_js_summary
# ---------------------------------------------------------------------------

def test_summary_output_keys():
    """Summary dict contains all expected keys."""
    mol = _mol("CCCC")
    coords = _random_coords(mol, T=50)
    result = ligand_torsion_js_summary(mol, coords, coords)
    expected_keys = {"per_bond_js", "n_bonds", "mean_js", "median_js", "success_rate", "quartets"}
    assert expected_keys.issubset(set(result.keys()))


def test_summary_identical_ensembles_low_js():
    """Identical predicted and reference ensembles give low JS distance."""
    mol = _mol("CCCCCC")
    coords = _random_coords(mol, T=200)
    result = ligand_torsion_js_summary(mol, coords, coords)
    assert result["mean_js"] < 0.1, f"Expected low JS for identical input, got {result['mean_js']:.3f}"


def test_summary_n_bonds_matches_quartets():
    """n_bonds equals len(per_bond_js) and len(quartets)."""
    mol = _mol("CCC(=O)c1ccccc1")  # butyrophenone: a few rotatable bonds
    coords = _random_coords(mol, T=50)
    result = ligand_torsion_js_summary(mol, coords, coords)
    assert result["n_bonds"] == len(result["per_bond_js"])
    assert result["n_bonds"] == len(result["quartets"])


def test_summary_success_rate_range():
    """success_rate must be in [0, 1]."""
    mol = _mol("CCCCCC")
    coords_p = _random_coords(mol, T=100)
    coords_q = _random_coords(mol, T=100)
    result = ligand_torsion_js_summary(mol, coords_p, coords_q)
    assert 0.0 <= result["success_rate"] <= 1.0


def test_summary_no_rotatable_bonds_returns_nan():
    """Benzene (no rotatable bonds) returns NaN metrics."""
    mol = _mol("c1ccccc1")
    coords = _random_coords(mol, T=50)
    result = ligand_torsion_js_summary(mol, coords, coords)
    assert result["n_bonds"] == 0
    assert np.isnan(result["mean_js"])
    assert np.isnan(result["median_js"])


def test_summary_atom_count_mismatch_raises():
    """Mismatched atom counts between pred and ref raise ValueError."""
    mol = _mol("CCCC")
    n = mol.GetNumAtoms()
    coords_pred = _random_coords(mol, T=20)
    coords_ref = RNG.normal(0, 1, (20, n + 2, 3))  # wrong atom count
    with pytest.raises(ValueError, match="Atom count mismatch"):
        ligand_torsion_js_summary(mol, coords_pred, coords_ref)

"""Tests for residue-aware binding-site indexing (ligand–protein contact shells)."""

from __future__ import annotations

import numpy as np
import pytest

from genai_tps.backends.boltz.collective_variables import _binding_site_touching_protein_atoms


def test_contact_defines_pocket_when_com_sphere_is_empty():
    """Docking-style contact: sidechain may sit near ligand while ligand COM stays far from Cα."""
    # Two protein "atoms" far from ligand COM cluster; one ligand atom hugs protein 0.
    prot = np.array(
        [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float64
    )  # residues 0 & 1 along x
    lig = np.array([[0.4, 0.0, 0.0]], dtype=np.float64)  # touching residue 0 within 0.5 Å
    prot_idx = np.array([0, 1], dtype=np.int64)
    lig_idx = np.array([10], dtype=np.int64)  # indices arbitrary; coords from arrays

    ref = np.zeros((11, 3), dtype=np.float64)
    ref[0] = prot[0]
    ref[1] = prot[1]
    ref[10] = lig[0]

    hit = _binding_site_touching_protein_atoms(
        ref,
        lig_idx,
        prot_idx,
        cutoff_angstrom=1.5,
    )

    assert np.array_equal(hit, np.array([0], dtype=np.int64))

    lig_com = ref[lig_idx].mean(axis=0)
    d_com_to_ca1 = float(np.linalg.norm(prot[1] - lig_com))
    assert d_com_to_ca1 > 5.0


def test_touching_requires_finite_cutoff():
    with pytest.raises(ValueError, match="positive"):
        _binding_site_touching_protein_atoms(
            np.zeros((2, 3)),
            np.array([0]),
            np.array([1]),
            cutoff_angstrom=0.0,
        )

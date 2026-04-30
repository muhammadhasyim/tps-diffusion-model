"""Tests for Boltz NPZ + CCD pickle ligand placement (``boltz_ligand_pose``)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from boltz.data.types import AtomV2, Chain, Residue


def _fake_structure_for_ethanol() -> object:
    """Minimal ``StructureV2``-like object: one NONPOLYMER chain ``B``, CCD ``TST``."""
    chains = np.array(
        [
            (
                "B",
                3,
                0,
                0,
                0,
                0,
                3,
                0,
                1,
                0,
            )
        ],
        dtype=Chain,
    )
    residues = np.array(
        [
            (
                "TST",
                0,
                0,
                0,
                3,
                0,
                0,
                False,
                True,
            )
        ],
        dtype=Residue,
    )
    atoms = np.array(
        [
            ("C1 ", (0.0, 0.0, 0.0), True, 0.0, 0.0),
            ("C2 ", (0.0, 0.0, 0.0), True, 0.0, 0.0),
            ("O1 ", (0.0, 0.0, 0.0), True, 0.0, 0.0),
        ],
        dtype=AtomV2,
    )

    class _S:
        pass

    s = _S()
    s.chains = chains
    s.residues = residues
    s.atoms = atoms
    return s


def _ethanol_rdkit_template_with_names():
    pytest.importorskip("rdkit", reason="RDKit required")
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    heavy = [(a.GetIdx(), a) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    assert len(heavy) == 3
    names = ["C1", "C2", "O1"]
    for (_, a), nm in zip(heavy, names, strict=True):
        a.SetProp("name", nm)
    return mol


def test_try_ligand_pose_from_boltz_ccd_places_heavy_atoms(tmp_path: Path):
    pytest.importorskip("openff.toolkit", reason="OpenFF required")
    from genai_tps.simulation.boltz_ligand_pose import try_ligand_pose_from_boltz_ccd

    mol_dir = tmp_path
    with (mol_dir / "TST.pkl").open("wb") as fh:
        pickle.dump(_ethanol_rdkit_template_with_names(), fh)

    structure = _fake_structure_for_ethanol()
    coords = np.array(
        [
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [11.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    out = try_ligand_pose_from_boltz_ccd(
        structure=structure,
        coords_angstrom=coords,
        mol_dir=mol_dir,
        chain_id="B",
        smiles="CCO",
        debug_sdf_dir=tmp_path / "dbg",
    )
    assert out is not None
    lig_pos, lig_topo = out
    atoms = list(lig_topo.atoms())
    expected = [(1.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.1, 0.1, 0.0)]
    hi = 0
    for i, atom in enumerate(atoms):
        if atom.element is None or atom.element.symbol == "H":
            continue
        v = lig_pos[i]
        ex, ey, ez = expected[hi]
        assert abs(float(v.x) - ex) < 1e-5
        assert abs(float(v.y) - ey) < 1e-5
        assert abs(float(v.z) - ez) < 1e-5
        hi += 1
    assert hi == 3
    assert (tmp_path / "dbg" / "ligand_B.sdf").is_file()


def test_try_ligand_pose_raises_on_smiles_graph_mismatch(tmp_path: Path):
    pytest.importorskip("openff.toolkit", reason="OpenFF required")
    from genai_tps.simulation.boltz_ligand_pose import try_ligand_pose_from_boltz_ccd

    mol_dir = tmp_path
    with (mol_dir / "TST.pkl").open("wb") as fh:
        pickle.dump(_ethanol_rdkit_template_with_names(), fh)
    structure = _fake_structure_for_ethanol()
    coords = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="isomorphic"):
        try_ligand_pose_from_boltz_ccd(
            structure=structure,
            coords_angstrom=coords,
            mol_dir=mol_dir,
            chain_id="B",
            smiles="CCC",
        )


def test_pdb_heavy_atoms_raises_without_gas_fallback(tmp_path: Path):
    """When PDB+RDKit cannot place the ligand, ``allow_gas_phase_fallback=False`` raises."""
    pytest.importorskip("rdkit", reason="RDKit required")
    pytest.importorskip("openff.toolkit", reason="OpenFF required")
    from unittest.mock import patch

    import openmm.app

    from openff.toolkit import Molecule as OpenFFMolecule  # noqa: PLC0415

    from compute_cv_rmsd import (  # type: ignore[import]
        _ligand_openmm_positions_from_pdb_heavy_atoms,
        _ligand_topology_relabeled_chain,
    )

    pdb_text = """\
ATOM      1  N   ALA A   1       0.0     0.0     0.0   1.00  0.00           N
ATOM      2  CA  ALA A   1       1.0     0.0     0.0   1.00  0.00           C
TER
HETATM   10  XX  LIG B   1      10.0    0.0     0.0   1.00  0.00           C
HETATM   11  YY  LIG B   1      12.0    0.0     0.0   1.00  0.00           C
HETATM   12  ZZ  LIG B   1      11.0    1.0     0.0   1.00  0.00           O
CONECT   10   11
CONECT   11   10   12
CONECT   12   11
END
"""
    pdb_path = tmp_path / "mismatch_names.pdb"
    pdb_path.write_text(pdb_text)
    pdb = openmm.app.PDBFile(str(pdb_path))
    mol = OpenFFMolecule.from_smiles("CCO", allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1, rms_cutoff=None)
    lig_topo = _ligand_topology_relabeled_chain(mol.to_topology().to_openmm(), "B")
    conf_nm = np.asarray(mol.conformers[0].magnitude) / 10.0
    with patch(
        "compute_cv_rmsd._rdkit_ligand_positions_nm_from_pdb", return_value=None
    ):
        with pytest.raises(RuntimeError, match="PDB|matched|gas"):
            _ligand_openmm_positions_from_pdb_heavy_atoms(
                "B",
                pdb.topology,
                pdb.positions,
                lig_topo,
                conf_nm,
                pdb_path=pdb_path,
                mol_off=mol,
                allow_gas_phase_fallback=False,
            )

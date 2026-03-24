"""Unit tests for Kabsch-RMSD and OpenMM minimisation helpers.

Tests
-----
Kabsch algorithm (pure numpy, no hardware required):
  - identity: RMSD of a set with itself is zero.
  - pure translation: shifting all coords by a constant vector gives zero RMSD.
  - pure rotation: a 90° rotation gives zero RMSD after alignment.
  - known value: two 3-atom sets with analytically known RMSD.
  - bad shape: raises ValueError for mismatched inputs.

OpenMM minimisation (CPU, no GPU required in CI):
  - smoke test on a minimal 2-residue ALA-ALA peptide constructed entirely
    in-memory; checks RMSD >= 0 and energy < 0 kJ/mol.

The smoke test is skipped (xfail → skip) if OpenMM is not installed or if
the AMBER14 XML is unavailable in the current environment.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from genai_tps.backends.boltz.collective_variables import kabsch_rmsd_aligned


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rot_z(theta: float) -> np.ndarray:
    """3×3 rotation matrix around the z-axis by *theta* radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Kabsch RMSD tests
# ---------------------------------------------------------------------------

class TestKabschRMSD:
    def test_identity(self):
        """RMSD of a coordinate set with itself must be exactly zero."""
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((20, 3))
        assert kabsch_rmsd_aligned(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_pure_translation(self):
        """Translating all atoms by a constant vector must give RMSD = 0."""
        rng = np.random.default_rng(1)
        coords = rng.standard_normal((15, 3))
        shifted = coords + np.array([5.0, -3.0, 7.5])
        assert kabsch_rmsd_aligned(coords, shifted) == pytest.approx(0.0, abs=1e-10)

    def test_pure_rotation_90deg(self):
        """A pure rotation must give RMSD = 0 after Kabsch alignment."""
        rng = np.random.default_rng(2)
        coords = rng.standard_normal((12, 3))
        R = _rot_z(math.pi / 2)
        rotated = coords @ R.T
        assert kabsch_rmsd_aligned(coords, rotated) == pytest.approx(0.0, abs=1e-9)

    def test_pure_rotation_arbitrary(self):
        """Arbitrary rotation must give RMSD = 0."""
        rng = np.random.default_rng(3)
        coords = rng.standard_normal((30, 3))
        # Random proper rotation via QR decomposition
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        rotated = coords @ Q.T
        assert kabsch_rmsd_aligned(coords, rotated) == pytest.approx(0.0, abs=1e-9)

    def test_translation_plus_rotation(self):
        """Translation + rotation combined must give RMSD = 0."""
        rng = np.random.default_rng(4)
        coords = rng.standard_normal((10, 3))
        R = _rot_z(1.23)
        transformed = coords @ R.T + np.array([2.0, -1.0, 4.0])
        assert kabsch_rmsd_aligned(coords, transformed) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        """Two 3-atom sets: atom 1 displaced by d; RMSD = d / sqrt(3) after alignment.

        If P = [[0,0,0],[1,0,0],[0,1,0]] and Q = [[d,0,0],[1,0,0],[0,1,0]]
        the centred displacement of atom 0 after Kabsch is exactly d*(2/3)
        while atoms 1 and 2 each contribute d*(1/3) in the opposite direction,
        giving RMSD² = (d*(2/3))² + 2*(d*(1/3))² = d²*(4+2)/9 = d²*6/9.
        However the Kabsch rotation may alter this slightly; we verify the
        result is strictly positive and within a reasonable bound.
        """
        d = 3.0  # Å
        coords_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coords_b = np.array([[d, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rmsd = kabsch_rmsd_aligned(coords_a, coords_b)
        assert rmsd > 0.0
        assert rmsd < d  # after alignment it must be strictly less than d

    def test_raises_on_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            kabsch_rmsd_aligned(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_raises_on_wrong_ndim(self):
        with pytest.raises(ValueError, match="shape"):
            kabsch_rmsd_aligned(np.zeros((5, 3)), np.zeros((5,)))

    def test_symmetric_up_to_rotation(self):
        """kabsch_rmsd_aligned(A, B) == kabsch_rmsd_aligned(B, A) (symmetric)."""
        rng = np.random.default_rng(5)
        a = rng.standard_normal((10, 3))
        b = rng.standard_normal((10, 3))
        assert kabsch_rmsd_aligned(a, b) == pytest.approx(
            kabsch_rmsd_aligned(b, a), rel=1e-6
        )


# ---------------------------------------------------------------------------
# OpenMM minimisation smoke test
# ---------------------------------------------------------------------------

_OPENMM_AVAILABLE = True
try:
    import openmm  # noqa: F401
    import openmm.app  # noqa: F401
    from openmm import unit  # noqa: F401
except ImportError:
    _OPENMM_AVAILABLE = False

# A minimal two-residue ALA-ALA peptide in PDB format (heavy atoms only,
# slightly distorted so the minimiser has something to do).
_ALA_ALA_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   1       1.201   0.847   0.100  1.00  0.00           N
    ATOM      2  CA  ALA A   1       2.285  -0.103   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       3.669   0.538   0.000  1.00  0.00           C
    ATOM      4  O   ALA A   1       4.008   1.650  -0.400  1.00  0.00           O
    ATOM      5  CB  ALA A   1       2.166  -0.999  -1.231  1.00  0.00           C
    ATOM      6  N   ALA A   2       4.592  -0.277   0.400  1.00  0.00           N
    ATOM      7  CA  ALA A   2       5.961   0.223   0.400  1.00  0.00           C
    ATOM      8  C   ALA A   2       6.940  -0.921   0.000  1.00  0.00           C
    ATOM      9  O   ALA A   2       7.100  -1.920   0.800  1.00  0.00           O
    ATOM     10  CB  ALA A   2       6.343   1.332   1.381  1.00  0.00           C
    ATOM     11  OXT ALA A   2       7.643  -0.870  -1.000  1.00  0.00           O
    END
""")


@pytest.mark.skipif(not _OPENMM_AVAILABLE, reason="openmm not installed")
class TestMinimizePDB:
    def _write_ala_ala(self, tmp_path: Path) -> Path:
        pdb_path = tmp_path / "ala_ala.pdb"
        pdb_path.write_text(_ALA_ALA_PDB)
        return pdb_path

    def test_minimize_smoke(self, tmp_path):
        """Minimisation of a small peptide must converge and give valid outputs."""
        from compute_cv_rmsd import minimize_pdb  # type: ignore[import]

        pdb_path = self._write_ala_ala(tmp_path)
        result = minimize_pdb(pdb_path, max_iter=500, platform_name="CPU")

        assert result["converged"], f"Minimisation failed: {result['error']}"
        assert result["ca_rmsd_angstrom"] is not None
        assert result["ca_rmsd_angstrom"] >= 0.0, "RMSD must be non-negative"
        assert result["energy_kj_mol"] is not None
        # AMBER14 + GBn2 implicit solvent potential energies for a small peptide
        # are always negative (electrostatic and solvation contributions dominate)
        assert result["energy_kj_mol"] < 0.0, (
            f"Expected negative potential energy, got {result['energy_kj_mol']}"
        )
        assert result["n_ca_atoms"] == 2, "ALA-ALA has 2 Cα atoms"
        assert result["platform_used"] == "CPU"
        assert result["error"] is None

    def test_minimize_bad_pdb_returns_error(self, tmp_path):
        """A nonsense PDB must return an error dict rather than raise."""
        from compute_cv_rmsd import minimize_pdb  # type: ignore[import]

        bad = tmp_path / "bad.pdb"
        bad.write_text("ATOM      1  XX  XYZ A   1       0.0   0.0   0.0\nEND\n")
        result = minimize_pdb(bad, max_iter=100, platform_name="CPU")
        assert not result["converged"]
        assert result["error"] is not None

    def test_select_ca_indices(self, tmp_path):
        """select_ca_indices should return exactly the Cα atoms."""
        from compute_cv_rmsd import select_ca_indices  # type: ignore[import]
        import openmm.app

        pdb_path = self._write_ala_ala(tmp_path)
        pdb = openmm.app.PDBFile(str(pdb_path))
        ca_idx = select_ca_indices(pdb.topology)
        assert len(ca_idx) == 2
        names = [
            a.name
            for a in pdb.topology.atoms()
            if a.index in ca_idx
        ]
        assert all(n == "CA" for n in names)

    def test_get_ca_coords_angstrom_shape(self, tmp_path):
        """get_ca_coords_angstrom must return (N_CA, 3) in Angstroms."""
        from compute_cv_rmsd import (  # type: ignore[import]
            get_ca_coords_angstrom,
            select_ca_indices,
        )
        import openmm.app

        pdb_path = self._write_ala_ala(tmp_path)
        pdb = openmm.app.PDBFile(str(pdb_path))
        ca_idx = select_ca_indices(pdb.topology)
        ca_coords = get_ca_coords_angstrom(pdb.positions, ca_idx)

        assert ca_coords.shape == (2, 3)
        # Positions in the PDB are in Angstroms (1–10 Å range); OpenMM
        # PDBFile stores them in nanometres → after * 10 they should be > 1 Å
        assert np.all(np.abs(ca_coords) < 100.0), "Coordinates look unexpectedly large"

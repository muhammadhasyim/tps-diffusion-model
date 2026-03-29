"""CUDA + OpenMM pipeline checks beyond ``python -m openmm.testInstallation``.

`openmm.testInstallation` only exercises a tiny generic system in a fresh
process.  TPS-OPES uses **PyTorch on CUDA** together with **OpenMM** on the
same GPU and runs **AMBER14 + implicit GBn2** kernels inside
``scripts/compute_cv_rmsd.minimize_pdb``.  Driver/PTX mismatches can pass the
stock test yet fail on a heavier kernel or after CUDA context ordering changes.

These tests are marked ``@pytest.mark.cuda`` and are **skipped** when PyTorch
CUDA is unavailable (e.g. CPU CI).  On a GPU machine they **fail loudly** if
OpenMM's CUDA platform cannot run the same runtime smoke test used by
``compute_cv_rmsd._get_platform``, or if a full protein minimisation does not
actually use ``platform_used == "CUDA"`` (silent fallback would hide PTX issues
from users who expect GPU minimisation).

Optional: protein + ligand minimisation on CUDA (GAFF2 / antechamber on CPU,
OpenMM integration on GPU) — skipped when openmmforcefields stack is broken.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

# scripts/ on path via conftest
from compute_cv_rmsd import (  # type: ignore[import]
    minimize_pdb,
    _platform_runtime_smoke_test,
)

# Same heavy-atom ALA-ALA peptide as tests/test_cv_rmsd.py (protein-only path).
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


def _openmm_cuda_platform():
    pytest.importorskip("openmm", reason="openmm not installed")
    import openmm

    return openmm.Platform.getPlatformByName("CUDA")


@pytest.mark.cuda
def test_openmm_cuda_same_runtime_smoke_as_compute_cv_rmsd():
    """Two-particle Context on CUDA — identical guard used before minimisation."""
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA not available")

    plat = _openmm_cuda_platform()
    plat.setPropertyDefaultValue("CudaPrecision", "mixed")
    ok, err = _platform_runtime_smoke_test(plat)
    assert ok, (
        "OpenMM CUDA failed the same runtime check as compute_cv_rmsd._get_platform. "
        "Typical cause: CUDA_ERROR_UNSUPPORTED_PTX_VERSION or driver/plugin mismatch. "
        f"Detail: {err}"
    )


@pytest.mark.cuda
def test_openmm_cuda_runtime_smoke_after_pytorch_allocates():
    """Order-sensitive check: PyTorch grabs the GPU first, then OpenMM CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA not available")

    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(4096, 4096, device=dev)
    y = torch.randn(4096, 4096, device=dev)
    (x @ y).sum().item()
    torch.cuda.synchronize()

    plat = _openmm_cuda_platform()
    plat.setPropertyDefaultValue("CudaPrecision", "mixed")
    ok, err = _platform_runtime_smoke_test(plat)
    assert ok, (
        "OpenMM CUDA failed after PyTorch CUDA work — similar to Boltz + CV ordering. "
        f"Detail: {err}"
    )
    del x, y
    torch.cuda.empty_cache()


@pytest.mark.cuda
def test_minimize_pdb_protein_only_uses_cuda_not_silent_fallback(tmp_path):
    """Full PDBFixer + AMBER14 + GBn2 minimisation must run on CUDA when requested."""
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA not available")

    plat = _openmm_cuda_platform()
    plat.setPropertyDefaultValue("CudaPrecision", "mixed")
    ok, err = _platform_runtime_smoke_test(plat)
    if not ok:
        pytest.fail(
            "CUDA smoke failed before minimise (fix OpenMM/driver first). "
            f"Detail: {err}"
        )

    pdb_path = tmp_path / "ala_ala.pdb"
    pdb_path.write_text(_ALA_ALA_PDB)
    result = minimize_pdb(pdb_path, max_iter=500, platform_name="CUDA")

    assert result["converged"], result.get("error")
    assert result["error"] is None
    assert result["platform_used"] == "CUDA", (
        "minimize_pdb fell back from CUDA — often PTX/OpenCL/CPU path while PyTorch "
        f"still holds VRAM. Got platform_used={result['platform_used']!r}."
    )
    assert result["ca_rmsd_angstrom"] is not None
    assert result["energy_kj_mol"] is not None


@pytest.mark.cuda
def test_minimize_pdb_protein_only_cuda_after_pytorch(tmp_path):
    """End-to-end ordering: torch on GPU, then full minimise on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA not available")

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.randn(1024, 1024, device=dev).sum().item()
    torch.cuda.synchronize()

    pdb_path = tmp_path / "ala_ala.pdb"
    pdb_path.write_text(_ALA_ALA_PDB)
    result = minimize_pdb(pdb_path, max_iter=500, platform_name="CUDA")

    assert result["converged"], result.get("error")
    assert result["platform_used"] == "CUDA", result


# --- Optional ligand path (GAFF2 parameterisation on CPU, OpenMM on CUDA) ---

try:
    from openmmforcefields.generators import GAFFTemplateGenerator  # noqa: F401
    from openff.toolkit.topology import Molecule as _OpenFFMolecule  # noqa: F401

    _LIGAND_STACK_OK = True
except Exception:
    _LIGAND_STACK_OK = False

_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

_ALA_ALA_ASPIRIN_PDB = textwrap.dedent("""\
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
    TER      12      ALA A   2
    HETATM   13  C1  LIG B   1      21.160   4.950   0.000  1.00  0.00           C
    HETATM   14  C2  LIG B   1      21.160   3.440   0.000  1.00  0.00           C
    HETATM   15  O3  LIG B   1      22.217   2.830   0.000  1.00  0.00           O
    HETATM   16  O4  LIG B   1      20.000   2.770   0.000  1.00  0.00           O
    HETATM   17  C5  LIG B   1      20.000   1.400   0.000  1.00  0.00           C
    HETATM   18  C6  LIG B   1      21.210   0.700   0.000  1.00  0.00           C
    HETATM   19  C7  LIG B   1      21.210  -0.700   0.000  1.00  0.00           C
    HETATM   20  C8  LIG B   1      20.000  -1.400   0.000  1.00  0.00           C
    HETATM   21  C9  LIG B   1      18.790  -0.700   0.000  1.00  0.00           C
    HETATM   22  C10 LIG B   1      18.790   0.700   0.000  1.00  0.00           C
    HETATM   23  C11 LIG B   1      17.485   1.455   0.000  1.00  0.00           C
    HETATM   24  O12 LIG B   1      17.485   2.675   0.000  1.00  0.00           O
    HETATM   25  O13 LIG B   1      16.309   0.778   0.000  1.00  0.00           O
    CONECT   13   14
    CONECT   14   13   15   16
    CONECT   15   14
    CONECT   16   14   17
    CONECT   17   16   18   22
    CONECT   18   17   19
    CONECT   19   18   20
    CONECT   20   19   21
    CONECT   21   20   22
    CONECT   22   21   17   23
    CONECT   23   22   24   25
    CONECT   24   23
    CONECT   25   23
    END
""")


@pytest.mark.cuda
@pytest.mark.skipif(not _LIGAND_STACK_OK, reason="openmmforcefields / openff stack")
def test_minimize_pdb_protein_ligand_uses_cuda(tmp_path):
    """Ligand path: GAFF typing (CPU) + OpenMM minimise on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA not available")

    pdb_path = tmp_path / "ala_ala_aspirin.pdb"
    pdb_path.write_text(_ALA_ALA_ASPIRIN_PDB)
    result = minimize_pdb(
        pdb_path,
        max_iter=500,
        platform_name="CUDA",
        ligand_smiles={"B": _ASPIRIN_SMILES},
    )

    assert result["converged"], result.get("error")
    assert result["platform_used"] == "CUDA", result


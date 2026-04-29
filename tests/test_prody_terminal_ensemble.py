"""Tests for genai_tps.evaluation.terminal_ensemble_prody — ProDy terminal-structure ensemble analysis.

ProDy is a required analysis dependency (pyproject.toml [analysis]).

The tests use minimal synthetic PDB fixtures (5 atoms per structure, 3–5
structures) to keep runtime under a second.
"""

from __future__ import annotations

import json
import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import prody  # noqa: F401
import pytest

# ---------------------------------------------------------------------------
# Minimal PDB fixture helpers
# ---------------------------------------------------------------------------
#
# Three-residue poly-Cα PDB with chain A.  Each call perturbs the coordinates
# by a small random offset so the ensemble has some variance.

_CA_TEMPLATE = """\
ATOM      1  CA  ALA A   1    {x0:8.3f}{y0:8.3f}{z0:8.3f}  1.00  0.00           C
ATOM      2  CA  GLY A   2    {x1:8.3f}{y1:8.3f}{z1:8.3f}  1.00  0.00           C
ATOM      3  CA  VAL A   3    {x2:8.3f}{y2:8.3f}{z2:8.3f}  1.00  0.00           C
END
"""

# Base coordinates (three Cα positions in Å)
_BASE_COORDS = np.array([
    [1.0, 0.0, 0.0],
    [4.8, 0.0, 0.0],
    [8.6, 0.0, 0.0],
], dtype=np.float64)


def _write_pdb(path: Path, coords: np.ndarray) -> None:
    text = _CA_TEMPLATE.format(
        x0=coords[0, 0], y0=coords[0, 1], z0=coords[0, 2],
        x1=coords[1, 0], y1=coords[1, 1], z1=coords[1, 2],
        x2=coords[2, 0], y2=coords[2, 1], z2=coords[2, 2],
    )
    path.write_text(text)


def _make_ensemble_pdb_dir(tmp_path: Path, n: int = 5, seed: int = 42) -> Path:
    """Create *n* PDB files with perturbed coordinates in *tmp_path/pdbs/*."""
    rng = np.random.default_rng(seed)
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for i in range(n):
        noise = rng.normal(scale=1.5, size=(3, 3))
        coords = _BASE_COORDS + noise
        _write_pdb(pdb_dir / f"tps_mc_step_{(i + 1) * 10:08d}_last.pdb", coords)
    return pdb_dir


# ---------------------------------------------------------------------------
# Unit tests for EnsembleAnalysisResult.to_dict()
# ---------------------------------------------------------------------------

class TestEnsembleAnalysisResultToDict:
    def test_empty_result_serialisable(self):
        from genai_tps.evaluation.terminal_ensemble_prody import EnsembleAnalysisResult

        r = EnsembleAnalysisResult()
        d = r.to_dict()
        json_str = json.dumps(d)
        back = json.loads(json_str)
        assert back["n_structures"] == 0
        assert back["eigenvalues"] == []
        assert back["anm_eigenvalues"] is None

    def test_filled_result_serialisable(self):
        from genai_tps.evaluation.terminal_ensemble_prody import EnsembleAnalysisResult

        r = EnsembleAnalysisResult(
            n_structures=3,
            n_ca_atoms=5,
            eigenvalues=np.array([10.0, 5.0]),
            explained_variance_ratio=np.array([0.67, 0.33]),
            cumulative_variance_ratio=np.array([0.67, 1.0]),
            projections=np.zeros((3, 2)),
            labels=["a", "b", "c"],
            mean_coords=np.zeros((5, 3)),
            anm_eigenvalues=np.array([1.0, 2.0]),
            gnm_eigenvalues=None,
        )
        d = r.to_dict()
        json_str = json.dumps(d)
        back = json.loads(json_str)
        assert back["n_structures"] == 3
        assert back["anm_eigenvalues"] == pytest.approx([1.0, 2.0])
        assert back["gnm_eigenvalues"] is None


# ---------------------------------------------------------------------------
# Unit tests for run_ensemble_analysis
# ---------------------------------------------------------------------------

class TestRunEnsembleAnalysis:
    def test_basic_pca(self, tmp_path):
        """PCA should run and return correct shapes for a minimal ensemble."""
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=5)
        result = run_ensemble_analysis(pdb_dir, n_pcs=3)

        assert result.n_structures == 5
        assert result.n_ca_atoms == 3
        # n_pcs is capped at min(n_pcs, n_structures - 1) = 4
        n_pcs_actual = min(3, 5 - 1)
        assert len(result.eigenvalues) == n_pcs_actual
        assert result.projections.shape == (5, n_pcs_actual)
        assert result.labels == sorted(result.labels)  # sorted by filename
        assert result.failed_pdbs == []

    def test_eigenvalues_non_negative(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        result = run_ensemble_analysis(pdb_dir, n_pcs=3)
        assert np.all(result.eigenvalues >= 0)

    def test_explained_variance_sums_to_one(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=5)
        # 3 Cα atoms → at most 3*3-6=3 non-trivial modes; request 4 so capping kicks in
        result = run_ensemble_analysis(pdb_dir, n_pcs=4)
        # cumulative variance of all returned PCs must sum to 1
        assert result.cumulative_variance_ratio[-1] == pytest.approx(1.0, abs=1e-6)

    def test_mean_coords_shape(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        result = run_ensemble_analysis(pdb_dir, n_pcs=2)
        assert result.mean_coords.shape == (3, 3)  # (n_ca_atoms=3, 3D)

    def test_anm_runs_and_gives_eigenvalues(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        result = run_ensemble_analysis(pdb_dir, n_pcs=2, run_anm=True, n_enm_modes=3)
        assert result.anm_eigenvalues is not None
        assert len(result.anm_eigenvalues) > 0
        assert np.all(result.anm_eigenvalues >= 0)

    def test_gnm_runs_and_gives_eigenvalues(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        result = run_ensemble_analysis(pdb_dir, n_pcs=2, run_gnm=True, n_enm_modes=3)
        assert result.gnm_eigenvalues is not None
        assert len(result.gnm_eigenvalues) > 0

    def test_warns_on_fewer_than_two_structures(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=1)
        with pytest.warns(RuntimeWarning, match="at least 2"):
            result = run_ensemble_analysis(pdb_dir, n_pcs=2)
        assert result.n_structures < 2
        assert len(result.eigenvalues) == 0

    def test_empty_dir_raises(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        with pytest.raises(FileNotFoundError, match="No PDB files"):
            run_ensemble_analysis(tmp_path / "nonexistent")

    def test_topology_mismatch_skips_bad_structure(self, tmp_path):
        """A PDB with a different number of Cα atoms is skipped, not fatal."""
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        # Write a 2-residue PDB (wrong atom count)
        bad_pdb = pdb_dir / "tps_mc_step_00000999_last.pdb"
        bad_pdb.write_text(textwrap.dedent("""\
            ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
            ATOM      2  CA  GLY A   2       4.800   0.000   0.000  1.00  0.00           C
            END
        """))
        result = run_ensemble_analysis(pdb_dir, n_pcs=2)
        assert bad_pdb.stem in result.failed_pdbs
        assert result.n_structures == 4  # the 4 good ones still processed

    def test_result_to_dict_roundtrip(self, tmp_path):
        from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

        pdb_dir = _make_ensemble_pdb_dir(tmp_path, n=4)
        result = run_ensemble_analysis(pdb_dir, n_pcs=2)
        d = result.to_dict()
        json_str = json.dumps(d)
        back = json.loads(json_str)
        assert back["n_structures"] == result.n_structures
        assert len(back["eigenvalues"]) == len(result.eigenvalues)
        assert len(back["labels"]) == result.n_structures


# ---------------------------------------------------------------------------
# Tests for boltz_npz_export (no Boltz dependency — test only pure-NumPy parts)
# ---------------------------------------------------------------------------

class TestBoltzNpzExportPaths:
    def test_load_topo_missing_file_raises(self, tmp_path):
        from genai_tps.io.boltz_npz_export import load_topo

        with pytest.raises(FileNotFoundError):
            load_topo(tmp_path / "nonexistent.npz")

    def test_batch_export_empty_dir_returns_empty_list(self, tmp_path):
        """batch_export with no matching NPZ files should return empty list."""
        from boltz.data.types import StructureV2  # noqa: F401

        from genai_tps.io.boltz_npz_export import batch_export

        # Supply a dummy topo_npz — the function will fail at load_topo before
        # iterating, but we want it to iterate on an empty dir instead.
        # Easiest: pass a real topo NPZ if present, else skip.
        topo_candidates = sorted(
            Path("artifacts/cofolding/cofolding_tps_out").glob(
                "boltz_results_*/processed/structures/*.npz"
            )
        )
        if not topo_candidates:
            pytest.skip(
                "No Boltz topology NPZ found in artifacts/cofolding/cofolding_tps_out/"
            )

        result = batch_export(tmp_path, topo_candidates[0], tmp_path / "out")
        assert result == []

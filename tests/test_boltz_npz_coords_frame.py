"""Tests for ``coords_frame_from_npz`` without importing ``genai_tps`` package root.

``genai_tps.__init__`` pulls OpenMM-dependent stacks that can fail pytest on some
NumPy ABI combinations; loading the I/O module in isolation keeps this logic
tested everywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src/python/genai_tps/io/boltz_npz_export.py"


def _load_boltz_npz_export_module():
    spec = importlib.util.spec_from_file_location(
        "boltz_npz_export_standalone",
        _MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coords_frame_from_npz():
    return _load_boltz_npz_export_module().coords_frame_from_npz


@pytest.fixture(scope="module")
def coords_frame_from_npz():
    return _coords_frame_from_npz()


@pytest.fixture(scope="module")
def coords_centered_for_pdb_export():
    return _load_boltz_npz_export_module().coords_centered_for_pdb_export


class TestCoordsFrameFromNpz:
    def test_trajectory_coords_last_frame(self, tmp_path, coords_frame_from_npz):
        traj = np.zeros((4, 5, 3), dtype=np.float32)
        traj[3, :, 0] = 1.0
        path = tmp_path / "t.npz"
        np.savez(path, coords=traj)
        with np.load(path) as d:
            fc = coords_frame_from_npz(d, frame_idx=-1, n_struct=5)
        assert fc.shape == (5, 3)
        assert np.allclose(fc[:, 0], 1.0)

    def test_trajectory_atom_coords_middle_index(self, tmp_path, coords_frame_from_npz):
        traj = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
        path = tmp_path / "t.npz"
        np.savez(path, atom_coords=traj)
        with np.load(path) as d:
            fc = coords_frame_from_npz(d, frame_idx=1, n_struct=4)
        assert np.allclose(fc, traj[1])

    def test_boltz_structured_coords(self, tmp_path, coords_frame_from_npz):
        dt = np.dtype([("coords", np.float32, (3,))])
        structured = np.zeros(3, dtype=dt)
        structured["coords"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        path = tmp_path / "t.npz"
        np.savez(path, coords=structured)
        with np.load(path) as d:
            fc = coords_frame_from_npz(d, frame_idx=99, n_struct=3)
        assert fc.dtype == np.float32
        assert fc.shape == (3, 3)
        assert np.allclose(fc[0], [1, 2, 3])

    def test_snapshot_plain_matrix(self, tmp_path, coords_frame_from_npz):
        xy = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        path = tmp_path / "t.npz"
        np.savez(path, coords=xy)
        with np.load(path) as d:
            fc = coords_frame_from_npz(d, frame_idx=0, n_struct=2)
        assert fc.dtype == np.float32
        assert fc.shape == (2, 3)

    def test_truncate_n_struct(self, tmp_path, coords_frame_from_npz):
        xy = np.ones((10, 3), dtype=np.float32)
        path = tmp_path / "t.npz"
        np.savez(path, coords=xy)
        with np.load(path) as d:
            fc = coords_frame_from_npz(d, frame_idx=0, n_struct=4)
        assert fc.shape == (4, 3)

    def test_missing_coords_raises_keyerror(self, tmp_path, coords_frame_from_npz):
        path = tmp_path / "t.npz"
        np.savez(path, other=np.array([1]))
        with np.load(path) as d, pytest.raises(KeyError):
            coords_frame_from_npz(d, frame_idx=0, n_struct=1)

    def test_unsupported_layout_raises(self, tmp_path, coords_frame_from_npz):
        path = tmp_path / "t.npz"
        np.savez(path, coords=np.zeros((2, 2, 2, 3), dtype=np.float32))
        with np.load(path) as d, pytest.raises(ValueError, match="Unsupported"):
            coords_frame_from_npz(d, frame_idx=0, n_struct=1)


class TestCoordsCenteredForPdbExport:
    """COM shift must make Boltz-style %8.3f PDB coordinate fields parseable."""

    def test_far_from_origin_fits_after_centering(self, coords_centered_for_pdb_export):
        # Far from origin: raw %8.3f would overflow; centered COM is near zero.
        fc = np.array(
            [
                [10_000.0, 0.0, 0.0],
                [10_001.0, 1.0, 0.0],
                [10_000.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        shifted, com = coords_centered_for_pdb_export(fc)
        assert shifted.shape == fc.shape
        np.testing.assert_allclose(
            com,
            np.array([10_000 + 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float64),
        )
        for v in shifted.ravel():
            assert len(f"{float(v):8.3f}") <= 8

    def test_raises_when_spread_too_large(self, coords_centered_for_pdb_export):
        fc = np.array(
            [
                [-20_000.0, 0.0, 0.0],
                [20_000.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        with pytest.raises(ValueError, match="PDB 8.3f"):
            coords_centered_for_pdb_export(fc)

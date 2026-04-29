"""Tests for ATLAS-to-WDSM conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from genai_tps.data.atlas_convert import (
    AtomMapError,
    BoltzAtomRecord,
    TrajectoryAtomRecord,
    build_atom_index_map,
    sample_frame_indices,
    wdsm_arrays_from_trajectory,
    write_wdsm_npz,
)


def _boltz_records() -> list[BoltzAtomRecord]:
    return [
        BoltzAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="N"),
        BoltzAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="CA"),
        BoltzAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="C"),
    ]


def _traj_records() -> list[TrajectoryAtomRecord]:
    return [
        TrajectoryAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="C"),
        TrajectoryAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="N"),
        TrajectoryAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="CA"),
    ]


def test_build_atom_index_map_reorders_to_boltz_order():
    index_map = build_atom_index_map(_traj_records(), _boltz_records())

    assert index_map.tolist() == [1, 2, 0]


def test_build_atom_index_map_fails_on_missing_atom():
    boltz = _boltz_records() + [
        BoltzAtomRecord(chain_id="A", residue_index=2, residue_name="GLY", atom_name="CA")
    ]

    with pytest.raises(AtomMapError, match="missing"):
        build_atom_index_map(_traj_records(), boltz)


def test_build_atom_index_map_fails_on_duplicate_trajectory_key():
    traj = _traj_records() + [
        TrajectoryAtomRecord(chain_id="A", residue_index=1, residue_name="ALA", atom_name="CA")
    ]

    with pytest.raises(AtomMapError, match="duplicate"):
        build_atom_index_map(traj, _boltz_records())


def test_sample_frame_indices_stride_and_max_frames_are_deterministic():
    assert sample_frame_indices(10, stride=3, max_frames=None, seed=0).tolist() == [0, 3, 6, 9]
    assert sample_frame_indices(10, stride=1, max_frames=4, seed=7).tolist() == [5, 6, 8, 9]


def test_wdsm_arrays_from_trajectory_reorders_coords_and_sets_uniform_logw():
    coords = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    index_map = np.array([1, 2, 0])

    wdsm = wdsm_arrays_from_trajectory(coords, index_map)

    assert wdsm.coords.shape == (2, 3, 3)
    np.testing.assert_array_equal(wdsm.coords[:, 0, :], coords[:, 1, :])
    np.testing.assert_array_equal(wdsm.logw, np.zeros(2, dtype=np.float64))
    np.testing.assert_array_equal(wdsm.atom_mask, np.ones((2, 3), dtype=np.float32))


def test_write_wdsm_npz_schema(tmp_path):
    coords = np.zeros((2, 3, 3), dtype=np.float32)
    logw = np.zeros(2, dtype=np.float64)
    atom_mask = np.ones((2, 3), dtype=np.float32)

    out = tmp_path / "training_dataset.npz"
    write_wdsm_npz(out, coords=coords, logw=logw, atom_mask=atom_mask)

    with np.load(out) as data:
        assert set(data.files) == {"coords", "logw", "atom_mask"}
        assert data["coords"].dtype == np.float32
        assert data["logw"].dtype == np.float64
        assert data["atom_mask"].dtype == np.float32

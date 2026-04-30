"""Tests for OneOPES hydration spot inference (geometric + mapping helpers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class _FakeIndexer:
    """Minimal stand-in for PoseCVIndexer fields used by inference."""

    def __init__(self) -> None:
        self.ligand_idx = np.array([0, 1], dtype=np.int64)
        self.pocket_heavy_idx = np.array([10, 11, 12], dtype=np.int64)
        self.ligand_no_idx = np.array([0], dtype=np.int64)
        self.pocket_no_idx = np.array([10], dtype=np.int64)


def test_geometric_hydration_spot_boltz_indices_interface() -> None:
    from genai_tps.simulation.hydration_site_inference import (
        geometric_hydration_spot_boltz_indices,
    )

    ref = np.zeros((20, 3), dtype=np.float64)
    ref[0] = (0.0, 0.0, 0.0)
    ref[1] = (1.0, 0.0, 0.0)
    ref[10] = (0.5, 0.5, 0.0)
    ref[11] = (5.0, 0.0, 0.0)
    ref[12] = (6.0, 0.0, 0.0)
    idx = _FakeIndexer()
    spots = geometric_hydration_spot_boltz_indices(
        ref, idx, max_sites=3, interface_cutoff_angstrom=2.0
    )
    assert 0 in spots
    assert 10 in spots


def test_map_water_centroids_to_boltz_no_indices() -> None:
    from genai_tps.simulation.hydration_site_inference import (
        map_water_centroids_to_boltz_no_indices,
    )

    ref = np.zeros((20, 3), dtype=np.float64)
    ref[0] = (0.0, 0.0, 0.0)
    ref[10] = (1.0, 0.0, 0.0)
    cents = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)
    idx = _FakeIndexer()
    out = map_water_centroids_to_boltz_no_indices(
        cents, ref, idx, max_sites=2, max_anchor_distance_angstrom=2.0
    )
    assert out == [0]


def test_default_oneopes_hydration_skips_rism_when_disabled(tmp_path: Path) -> None:
    from genai_tps.simulation.hydration_site_inference import (
        default_oneopes_hydration_boltz_indices,
    )

    ref = np.zeros((20, 3), dtype=np.float64)
    ref[0] = (0.0, 0.0, 0.0)
    ref[10] = (0.5, 0.0, 0.0)
    idx = _FakeIndexer()
    out = default_oneopes_hydration_boltz_indices(
        ref,
        idx,
        pdb_path=tmp_path / "missing.pdb",
        max_sites=5,
        use_3drism=False,
    )
    assert isinstance(out, list)
    assert len(out) >= 1


def test_try_infer_hydration_site_coords_3drism_returns_none_for_missing_pdb() -> None:
    from genai_tps.simulation.hydration_site_inference import (
        try_infer_hydration_site_coords_3drism,
    )

    assert try_infer_hydration_site_coords_3drism(Path("/nonexistent/no.pdb")) is None


def test_oneopes_plumed_script_contains_hydr_when_spots_provided(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    out_dir = tmp_path / "opes"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[1, 2],
        pocket_ca_plumed_idx=[3, 4],
        rmsd_reference_pdb=tmp_path / "ref.pdb",
        sigma=(0.3, 0.5),
        pace=500,
        barrier=40.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=1000,
        progress_every=100,
        out_dir=out_dir,
        cv_mode="oneopes",
        pocket_heavy_plumed_idx=[5, 6],
        coordination_r0=4.5,
        oneopes_axis_p0_plumed_idx=[1],
        oneopes_axis_p1_plumed_idx=[2],
        oneopes_contactmap_pairs_plumed=[(5, 1)],
        oneopes_hydration_spot_plumed_idx=[2],
        water_oxygen_plumed_idx=[100, 101],
    )
    assert "hydr_0" in script
    assert "opes_hydr_0" in script

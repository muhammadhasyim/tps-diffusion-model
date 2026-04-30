"""Regression tests for :func:`compute_cvs` atom-mask handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _find_any_topo_npz(repo_root: Path) -> Path | None:
    cands = sorted(repo_root.glob("**/processed/structures/*.npz"))
    return cands[0] if cands else None


def test_compute_cvs_atom_mask_1d_length_n_struct() -> None:
    """1-D mask of length *n_struct* (legacy callers) must not collapse to shape (1,)."""
    from genai_tps.backends.boltz.collective_variables import compute_cvs
    from genai_tps.io.boltz_npz_export import load_topo

    repo = Path(__file__).resolve().parents[1]
    topo_npz = _find_any_topo_npz(repo)
    if topo_npz is None:
        pytest.skip("No Boltz processed/structures/*.npz under repository checkout")

    structure, n_struct = load_topo(topo_npz)
    n_s = int(n_struct)
    ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)

    rng = np.random.default_rng(42)
    coords = ref_coords[np.newaxis, :, :].copy() + 0.01 * rng.standard_normal((2, n_s, 3))

    out = compute_cvs(
        coords,
        reference_coords=ref_coords,
        atom_mask_np=np.ones(n_s, dtype=np.float32),
        topo_npz=topo_npz,
        pocket_radius=6.0,
    )
    assert len(out["rmsd"]) == 2
    assert np.isfinite(out["rmsd"]).all()
    assert np.isfinite(out["rg"]).all()
    assert np.isfinite(out["contact_order"]).all()
    assert np.isfinite(out["clash_count"]).all()


def test_compute_cvs_atom_mask_2d_first_row() -> None:
    """2-D ``(1, n_atoms)`` mask remains supported."""
    from genai_tps.backends.boltz.collective_variables import compute_cvs
    from genai_tps.io.boltz_npz_export import load_topo

    repo = Path(__file__).resolve().parents[1]
    topo_npz = _find_any_topo_npz(repo)
    if topo_npz is None:
        pytest.skip("No Boltz processed/structures/*.npz under repository checkout")

    structure, n_struct = load_topo(topo_npz)
    n_s = int(n_struct)
    ref_coords = np.asarray(structure.atoms["coords"], dtype=np.float64)
    rng = np.random.default_rng(43)
    coords = ref_coords[np.newaxis, :, :].copy() + 0.01 * rng.standard_normal((1, n_s, 3))
    mask2 = np.ones((1, n_s), dtype=np.float32)

    out = compute_cvs(
        coords,
        reference_coords=ref_coords,
        atom_mask_np=mask2,
        topo_npz=topo_npz,
    )
    assert len(out["rmsd"]) == 1
    assert np.isfinite(out["rmsd"][0])


def test_normalize_atom_mask_rejects_bad_width() -> None:
    from genai_tps.backends.boltz.collective_variables import _normalize_atom_mask_for_compute_cvs

    with pytest.raises(ValueError, match="expected width"):
        _normalize_atom_mask_for_compute_cvs(
            np.ones((1, 5), dtype=np.float32),
            n_atoms_coords=10,
            n_struct_topo=5,
        )

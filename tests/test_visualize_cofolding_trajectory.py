"""Tests for ``scripts/visualize_cofolding_trajectory.py`` helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "scripts"))
from visualize_cofolding_trajectory import discover_structures_npz


def test_discover_structures_npz_single_match(tmp_path: Path) -> None:
    base = tmp_path / "cofolding_tps_out"
    target = base / "boltz_results_foo" / "processed" / "structures" / "rec.npz"
    target.parent.mkdir(parents=True)
    target.write_text("")

    found = discover_structures_npz(base / "coords_trajectory.npz", None)
    assert found == target


def test_discover_structures_npz_explicit(tmp_path: Path) -> None:
    p = tmp_path / "s.npz"
    p.write_text("")
    assert discover_structures_npz(tmp_path / "coords.npz", p) == p


def test_discover_structures_npz_ambiguous(tmp_path: Path) -> None:
    base = tmp_path / "out"
    for name in ("a", "b"):
        p = base / f"boltz_results_{name}" / "processed" / "structures" / "x.npz"
        p.parent.mkdir(parents=True)
        p.write_text("")
    assert discover_structures_npz(base / "coords.npz", None) is None


def test_render_cartoon_missing_pdb(tmp_path: Path) -> None:
    from visualize_cofolding_trajectory import render_cartoon_png_pymol

    missing = tmp_path / "nope.pdb"
    out = tmp_path / "x.png"
    try:
        render_cartoon_png_pymol(missing, out)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")

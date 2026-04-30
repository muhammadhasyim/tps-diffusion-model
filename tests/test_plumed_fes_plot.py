"""Tests for PLUMED FES ``.dat`` parsing and PNG plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest


def test_load_plumed_fes_dat_1d(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_fes_dat

    p = tmp_path / "fes1d.dat"
    p.write_text(
        "#! FIELDS rmsd file.free\n"
        "0.0 0.0\n"
        "0.5 1.5\n"
        "1.0 2.0\n",
        encoding="utf-8",
    )
    dim, names, x, y, fes = load_plumed_fes_dat(p)
    assert dim == 1
    assert names == ("rmsd",)
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    assert y.size == 0
    np.testing.assert_allclose(fes, [0.0, 1.5, 2.0])


def test_load_plumed_fes_dat_2d(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_fes_dat

    p = tmp_path / "fes2d.dat"
    p.write_text(
        "#! FIELDS cv_a cv_b file.free\n"
        "0.0 0.0 0.1\n"
        "0.0 1.0 0.2\n"
        "1.0 0.0 0.3\n",
        encoding="utf-8",
    )
    dim, names, x, y, fes = load_plumed_fes_dat(p)
    assert dim == 2
    assert names == ("cv_a", "cv_b")
    np.testing.assert_allclose(x, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(fes, [0.1, 0.2, 0.3])


def test_plot_fes_dat_to_png_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    from genai_tps.simulation.plumed_fes_plot import plot_fes_dat_to_png

    p = tmp_path / "fes2d.dat"
    p.write_text(
        "#! FIELDS x y file.free\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 1.0\n"
        "0.0 1.0 1.0\n"
        "1.0 1.0 2.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.png"
    plot_fes_dat_to_png(p, out)
    assert out.is_file()
    assert out.stat().st_size > 100

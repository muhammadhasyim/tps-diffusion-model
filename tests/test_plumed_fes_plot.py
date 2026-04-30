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
    dim, names, x, y, z, fes = load_plumed_fes_dat(p)
    assert dim == 1
    assert names == ("rmsd",)
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    assert y.size == 0
    assert z.size == 0
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
    dim, names, x, y, z, fes = load_plumed_fes_dat(p)
    assert dim == 2
    assert names == ("cv_a", "cv_b")
    assert z.size == 0
    np.testing.assert_allclose(x, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(fes, [0.1, 0.2, 0.3])


def test_load_plumed_fes_dat_3d(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_fes_dat

    p = tmp_path / "fes3d.dat"
    p.write_text(
        "#! FIELDS cv_a cv_b cv_c file.free\n"
        "#! SET genai_grid_bins 2 2 2\n"
        "0.0 0.0 0.0 0.0\n"
        "0.0 0.0 1.0 0.1\n"
        "0.0 1.0 0.0 0.2\n"
        "0.0 1.0 1.0 0.3\n"
        "1.0 0.0 0.0 0.4\n"
        "1.0 0.0 1.0 0.5\n"
        "1.0 1.0 0.0 0.6\n"
        "1.0 1.0 1.0 0.7\n",
        encoding="utf-8",
    )
    dim, names, x, y, z, fes = load_plumed_fes_dat(p)
    assert dim == 3
    assert names == ("cv_a", "cv_b", "cv_c")
    assert x.size == 8 and z.size == 8
    np.testing.assert_allclose(fes, np.linspace(0.0, 0.7, 8))


def test_plot_fes_dat_to_png_3d_marginals(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    from genai_tps.simulation.plumed_fes_plot import plot_fes_dat_to_png

    p = tmp_path / "fes3d.dat"
    p.write_text(
        "#! FIELDS cv_a cv_b cv_c file.free\n"
        "#! SET genai_grid_bins 2 2 2\n"
        "0.0 0.0 0.0 0.0\n"
        "0.0 0.0 1.0 0.1\n"
        "0.0 1.0 0.0 0.2\n"
        "0.0 1.0 1.0 0.3\n"
        "1.0 0.0 0.0 0.4\n"
        "1.0 0.0 1.0 0.5\n"
        "1.0 1.0 0.0 0.6\n"
        "1.0 1.0 1.0 0.7\n",
        encoding="utf-8",
    )
    out = tmp_path / "fes3d.png"
    plot_fes_dat_to_png(p, out)
    assert out.is_file()
    assert out.stat().st_size > 200


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


def test_load_plumed_kernels_2d_no_fields(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_kernels_2d

    kpath = tmp_path / "KERNELS"
    kpath.write_text(
        "0.0 0.1 0.2 0.15 0.15 0.01 -6.9\n"
        "1.0 0.3 0.4 0.15 0.15 0.02 -6.0\n",
        encoding="utf-8",
    )
    names, c, s, h = load_plumed_kernels_2d(kpath)
    assert names == ("cv1", "cv2")
    np.testing.assert_allclose(c[0], [0.1, 0.2])
    np.testing.assert_allclose(s[0], [0.15, 0.15])
    np.testing.assert_allclose(h, [0.01, 0.02])


def test_load_plumed_kernels_2d_plumed_regtest() -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_kernels_2d
    from genai_tps.subprocess_support import repository_root

    kpath = (
        repository_root()
        / "plumed2"
        / "regtest"
        / "opes"
        / "rt-opes_metad-restart"
        / "KERNELS"
    )
    if not kpath.is_file():
        pytest.skip("plumed2 submodule regtest KERNELS not present")
    names, c, s, h = load_plumed_kernels_2d(kpath)
    assert names == ("phi", "psi")
    assert c.shape[1] == 2 == s.shape[1]
    assert h.shape[0] == c.shape[0] > 0


def test_load_plumed_colvar_three_cvs(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_colvar_three_cvs

    cpath = tmp_path / "COLVAR"
    cpath.write_text(
        "#! FIELDS time lig_rmsd lig_dist lig_contacts opes.bias\n"
        "0 0.0 0.0 10.0 0\n"
        "1 0.5 0.5 11.0 1\n"
        "2 1.0 1.0 12.0 2\n",
        encoding="utf-8",
    )
    x, y, z = load_plumed_colvar_three_cvs(
        cpath, "lig_rmsd", "lig_dist", "lig_contacts"
    )
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(z, [10.0, 11.0, 12.0])


def test_load_plumed_colvar_two_cvs(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_colvar_two_cvs

    cpath = tmp_path / "COLVAR"
    cpath.write_text(
        "#! FIELDS time lig_rmsd lig_dist opes.bias\n"
        "0 0.0 0.0 0\n"
        "1 0.5 0.5 1\n"
        "2 1.0 1.0 2\n",
        encoding="utf-8",
    )
    x, y = load_plumed_colvar_two_cvs(cpath, "lig_rmsd", "lig_dist")
    np.testing.assert_allclose(x, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0])


def test_plot_opes_2d_fes_triptych_opes_style_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    from genai_tps.simulation.plumed_fes_plot import plot_opes_2d_fes_triptych_opes_style

    fes = tmp_path / "fes2d.dat"
    fes.write_text(
        "#! FIELDS lig_rmsd lig_dist file.free\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 1.0\n"
        "0.0 1.0 1.0\n"
        "1.0 1.0 2.0\n",
        encoding="utf-8",
    )
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time lig_rmsd lig_dist opes.bias\n"
        + "\n".join(f"{i} {0.1 * i} {0.1 * (5 - i)} {i}" for i in range(20))
        + "\n",
        encoding="utf-8",
    )
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "\n".join(
            f"{i} {0.2 * i} {0.15 * i} 0.2 0.2 0.01 -6.9" for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "trip_opes.png"
    plot_opes_2d_fes_triptych_opes_style(
        fes,
        kernels,
        colvar,
        out,
        grid_bins=32,
        hexbin_gridsize=20,
    )
    assert out.is_file()
    assert out.stat().st_size > 800


def test_plot_opes_2d_fes_triptych_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    from genai_tps.simulation.plumed_fes_plot import plot_opes_2d_fes_triptych

    fes = tmp_path / "fes2d.dat"
    fes.write_text(
        "#! FIELDS lig_rmsd lig_dist file.free\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 1.0\n"
        "0.0 1.0 1.0\n"
        "1.0 1.0 2.0\n",
        encoding="utf-8",
    )
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time lig_rmsd lig_dist opes.bias\n"
        + "\n".join(f"{i} {0.1 * i} {0.1 * (5 - i)} {i}" for i in range(20))
        + "\n",
        encoding="utf-8",
    )
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "\n".join(
            f"{i} {0.2 * i} {0.15 * i} 0.2 0.2 0.01 -6.9" for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "trip.png"
    plot_opes_2d_fes_triptych(
        fes,
        kernels,
        colvar,
        out,
        grid_bins=32,
        hist_bins=20,
    )
    assert out.is_file()
    assert out.stat().st_size > 500


def test_plot_opes_3d_fes_triptych_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    from genai_tps.simulation.plumed_fes_plot import plot_opes_3d_fes_triptych

    lines = ["#! FIELDS lig_rmsd lig_dist lig_contacts file.free"]
    lines.append("#! SET genai_grid_bins 2 2 2")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                lines.append(f"{float(i)} {float(j)} {float(k)} {float(i + j + k)}")
    fes = tmp_path / "fes3d.dat"
    fes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time lig_rmsd lig_dist lig_contacts opes.bias\n"
        + "\n".join(
            f"{i} {0.1 * i} {0.1 * (3 - i)} {50.0 + i} {i}" for i in range(15)
        )
        + "\n",
        encoding="utf-8",
    )
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "\n".join(
            f"{i} {0.2 * i} {0.15 * i} {50.0 + 0.1 * i} 0.2 0.2 0.2 0.01 -6.9"
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "trip3d.png"
    plot_opes_3d_fes_triptych(
        fes,
        kernels,
        colvar,
        out,
        grid_bins=24,
        hist_bins=12,
    )
    assert out.is_file()
    assert out.stat().st_size > 800


def test_plot_opes_3d_fes_triptych_opes_style_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    from genai_tps.simulation.plumed_fes_plot import plot_opes_3d_fes_triptych_opes_style

    lines = ["#! FIELDS lig_rmsd lig_dist lig_contacts file.free"]
    lines.append("#! SET genai_grid_bins 2 2 2")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                lines.append(f"{float(i)} {float(j)} {float(k)} {float(i + j + k)}")
    fes = tmp_path / "fes3d.dat"
    fes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time lig_rmsd lig_dist lig_contacts opes.bias\n"
        + "\n".join(
            f"{i} {0.1 * i} {0.1 * (3 - i)} {50.0 + i} {i}" for i in range(15)
        )
        + "\n",
        encoding="utf-8",
    )
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "\n".join(
            f"{i} {0.2 * i} {0.15 * i} {50.0 + 0.1 * i} 0.2 0.2 0.2 0.01 -6.9"
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "trip3d_opes.png"
    plot_opes_3d_fes_triptych_opes_style(fes, kernels, colvar, out, hexbin_gridsize=24)
    assert out.is_file()
    assert out.stat().st_size > 800


def test_plot_opes_3d_fes_slices_kde_deposits_writes_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    from genai_tps.simulation.plumed_fes_plot import plot_opes_3d_fes_slices_kde_deposits

    lines = ["#! FIELDS lig_rmsd lig_dist lig_contacts file.free"]
    lines.append("#! SET genai_grid_bins 2 2 2")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                lines.append(f"{float(i)} {float(j)} {float(k)} {float(i + j + k)}")
    fes = tmp_path / "fes3d.dat"
    fes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time lig_rmsd lig_dist lig_contacts opes.bias\n"
        + "\n".join(
            f"{i} {0.1 * i} {0.1 * (3 - i)} {50.0 + i} {i}" for i in range(15)
        )
        + "\n",
        encoding="utf-8",
    )
    kernels = tmp_path / "KERNELS"
    kernels.write_text(
        "\n".join(
            f"{i} {0.2 * i} {0.15 * i} {50.0 + 0.1 * i} 0.2 0.2 0.2 0.01 -6.9"
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "dash3d.png"
    plot_opes_3d_fes_slices_kde_deposits(fes, kernels, colvar, out)
    assert out.is_file()
    assert out.stat().st_size > 800


def test_mask_marginal_fes_far_from_colvar() -> None:
    from genai_tps.simulation.plumed_fes_plot import _mask_marginal_fes_far_from_colvar

    fm = np.ones((3, 3))
    gx = np.array([0.0, 1.0, 2.0])
    gy = np.array([0.0, 1.0, 2.0])
    samples_a = np.array([1.0])
    samples_b = np.array([1.0])
    out = _mask_marginal_fes_far_from_colvar(
        fm, gx, gy, samples_a, samples_b, 1.0, 1.0, nsigma=0.5
    )
    assert np.isfinite(out[1, 1])
    assert np.isnan(out[0, 0])
    assert np.isnan(out[2, 2])


def test_load_plumed_kernels_3d_roundtrip_fields(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_fes_plot import load_plumed_kernels_3d

    kpath = tmp_path / "K"
    kpath.write_text(
        "#! FIELDS time a b c sa sb sc h lw\n"
        "0 1 2 3 0.5 0.5 0.5 0.1 -1\n",
        encoding="utf-8",
    )
    names, c, s, h = load_plumed_kernels_3d(kpath)
    assert names == ("a", "b", "c")
    np.testing.assert_allclose(c[0], [1, 2, 3])
    np.testing.assert_allclose(s[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(h, [0.1])

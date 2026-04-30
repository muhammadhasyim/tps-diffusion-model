"""Tests for PLUMED COLVAR → FES subprocess wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest


def _minimal_colvar_text(*, n_rows: int = 48) -> str:
    lines = ["#! FIELDS time lig_rmsd lig_dist opes.bias"]
    for i in range(n_rows):
        t = float(i * 10)
        r = 1.0 + 0.02 * (i % 7)
        d = 5.0 + 0.03 * (i % 5)
        b = -2.0 + 0.1 * (i % 11)
        lines.append(f"{t:.6f} {r:.6f} {d:.6f} {b:.6f}")
    return "\n".join(lines) + "\n"


def test_run_fes_from_reweighting_script_synthetic_colvar(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_colvar_fes import (
        fes_from_reweighting_script_path,
        run_fes_from_reweighting_script,
    )
    from genai_tps.subprocess_support import repository_root

    script = fes_from_reweighting_script_path()
    if not script.is_file():
        pytest.skip(f"PLUMED tutorial script not present: {script}")

    opes_dir = tmp_path / "opes_states"
    opes_dir.mkdir(parents=True)
    colvar = opes_dir / "COLVAR"
    colvar.write_text(_minimal_colvar_text(), encoding="utf-8")
    outfile = opes_dir / "fes_reweighted_2d.dat"

    run_fes_from_reweighting_script(
        colvar_path=colvar,
        outfile=outfile,
        temperature_k=300.0,
        sigma="0.3,0.5",
        grid_bin="24,24",
        skiprows=0,
        blocks=1,
        repo_root=repository_root(),
    )

    assert outfile.is_file()
    text = outfile.read_text(encoding="utf-8").strip()
    assert len(text) > 0
    assert not text.startswith("--- ERROR")


def test_run_fes_rejects_empty_colvar(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_colvar_fes import (
        fes_from_reweighting_script_path,
        run_fes_from_reweighting_script,
    )

    if not fes_from_reweighting_script_path().is_file():
        pytest.skip("plumed2 tutorial not present")

    opes_dir = tmp_path / "opes_states"
    opes_dir.mkdir(parents=True)
    colvar = opes_dir / "COLVAR"
    colvar.write_text("", encoding="utf-8")
    outfile = opes_dir / "fes_reweighted_2d.dat"

    with pytest.raises(ValueError, match="empty or too small"):
        run_fes_from_reweighting_script(
            colvar_path=colvar,
            outfile=outfile,
            temperature_k=300.0,
            sigma="0.3,0.5",
            grid_bin="12,12",
        )


def test_run_fes_raises_if_outfile_wrong_parent(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_colvar_fes import (
        fes_from_reweighting_script_path,
        run_fes_from_reweighting_script,
    )

    if not fes_from_reweighting_script_path().is_file():
        pytest.skip("plumed2 tutorial not present")

    opes_dir = tmp_path / "opes_states"
    opes_dir.mkdir(parents=True)
    colvar = opes_dir / "COLVAR"
    colvar.write_text(_minimal_colvar_text(n_rows=10), encoding="utf-8")
    bad_out = tmp_path / "elsewhere" / "fes.dat"
    bad_out.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="outfile parent must match"):
        run_fes_from_reweighting_script(
            colvar_path=colvar,
            outfile=bad_out,
            temperature_k=300.0,
            sigma="0.3,0.5",
            grid_bin="12,12",
        )

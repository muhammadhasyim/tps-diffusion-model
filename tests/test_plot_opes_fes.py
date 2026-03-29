"""Tests for plot_opes_fes CV loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_load_cv_samples_jsonl(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "tps_steps.jsonl"
    p.write_text(
        json.dumps({"step": 1, "cv_value": 0.5}) + "\n"
        + json.dumps({"step": 2, "cv_value": None}) + "\n"
        + json.dumps({"step": 3, "cv_value": 1.25}) + "\n",
        encoding="utf-8",
    )
    arr = _load_cv_samples_jsonl(p, ndim=1)
    np.testing.assert_array_almost_equal(arr, np.array([0.5, 1.25]))


def test_load_cv_samples_jsonl_list_cv_value_first_component(tmp_path: Path) -> None:
    """Multi-D logs store cv_value as a list; plotting uses the first component."""
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "tps_steps.jsonl"
    p.write_text(
        json.dumps({"step": 1, "cv_value": [0.12, 3.4]}) + "\n",
        encoding="utf-8",
    )
    arr = _load_cv_samples_jsonl(p, ndim=1)
    np.testing.assert_array_almost_equal(arr, np.array([0.12]))


def test_load_cv_samples_jsonl_2d_two_columns(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "tps_steps.jsonl"
    p.write_text(
        json.dumps({"step": 1, "cv_value": [0.12, 3.4]}) + "\n"
        + json.dumps({"step": 2, "cv_value": [0.2, 3.5]}) + "\n",
        encoding="utf-8",
    )
    arr = _load_cv_samples_jsonl(p, ndim=2)
    assert arr.shape == (2, 2)
    np.testing.assert_array_almost_equal(arr[0], [0.12, 3.4])


def test_load_cv_samples_jsonl_empty_raises(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "empty.jsonl"
    p.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no finite"):
        _load_cv_samples_jsonl(p, ndim=1)


def test_plot_opes_fes_writes_png_explicit_x_range(tmp_path: Path) -> None:
    """plot_opes_fes respects x_min/x_max and writes a PNG."""
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    src = root / "src" / "python"
    for p in (scripts, src):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: PLC0415
    from plot_opes_fes import plot_opes_fes  # type: ignore[import]

    bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1)
    for step, cv in enumerate([0.4, 0.45, 0.5, 0.55, 0.48], start=1):
        bias.update(cv_accepted=cv, mc_step=step)

    cv_samples = np.array([0.4, 0.45, 0.5, 0.55, 0.48], dtype=np.float64)
    out = tmp_path / "fes.png"
    plot_opes_fes(
        bias,
        cv_samples,
        out,
        x_min=0.0,
        x_max=1.5,
        n_grid=80,
        n_hist_bins=12,
    )
    assert out.is_file()
    assert out.stat().st_size > 500


def test_plot_opes_fes_raises_when_x_range_excludes_all_samples(tmp_path: Path) -> None:
    """Fixed x window with no overlapping samples must error clearly (not hist 0/0)."""
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    src = root / "src" / "python"
    for p in (scripts, src):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: PLC0415
    from plot_opes_fes import plot_opes_fes  # type: ignore[import]

    bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1)
    for step, cv in enumerate([0.03, 0.04, 0.035], start=1):
        bias.update(cv_accepted=cv, mc_step=step)

    cv_samples = np.array([0.03, 0.04, 0.035], dtype=np.float64)
    out = tmp_path / "fes.png"
    with pytest.raises(ValueError, match="No CV samples fall"):
        plot_opes_fes(
            bias,
            cv_samples,
            out,
            x_min=0.1,
            x_max=0.8,
            n_grid=80,
            n_hist_bins=12,
        )


def test_resolve_state_path_prefers_latest_when_cv_newer_than_final(tmp_path: Path) -> None:
    import os
    import sys
    import time

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from plot_opes_fes import _resolve_state_path  # type: ignore[import]

    d = tmp_path / "run"
    (d / "opes_states").mkdir(parents=True)
    final = d / "opes_state_final.json"
    latest = d / "opes_states" / "opes_state_latest.json"
    cv = d / "cv_values.json"
    final.write_text("{}", encoding="utf-8")
    latest.write_text("{}", encoding="utf-8")
    cv.write_text("{}", encoding="utf-8")
    # Make final clearly older than cv
    old = time.time() - 100.0
    os.utime(final, (old, old))
    chosen = _resolve_state_path(d, cv_path=cv)
    assert chosen is not None and chosen.resolve() == latest.resolve()

    forced = _resolve_state_path(d, cv_path=cv, prefer_final=True)
    assert forced is not None and forced.resolve() == final.resolve()


def test_pdf_on_mesh_integrates_to_one() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from plot_opes_fes import (  # type: ignore[import]
        _integrate_trapezoid_2d,
        _pdf_on_mesh,
    )

    x = np.linspace(0.0, 1.0, 50)
    y = np.linspace(0.0, 1.0, 60)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.exp(-((X - 0.3) ** 2 + (Y - 0.7) ** 2) / 0.02)
    pdf = _pdf_on_mesh(Z, x, y)
    assert _integrate_trapezoid_2d(pdf, x, y) == pytest.approx(1.0, abs=2e-2)


def test_mathtext_cv_name_axis_label_escapes_underscore() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from plot_opes_fes import _mathtext_cv_name_axis_label  # type: ignore[import]

    assert _mathtext_cv_name_axis_label("contact_order") == r"$\mathrm{contact\_order}$"


def test_neg_ln_rho_relative_zero_at_mode() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from plot_opes_fes import _neg_ln_rho_relative  # type: ignore[import]

    p = np.array([0.1, 0.5, 0.2, 0.05], dtype=np.float64)
    z = _neg_ln_rho_relative(p)
    imax = int(np.argmax(p))
    assert z[imax] == pytest.approx(0.0, abs=1e-9)
    assert z[0] > 0.0 and z[-1] > z[imax]


def test_pdf_on_grid_integrates_to_one() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from plot_opes_fes import _integrate_trapezoid, _pdf_on_grid  # type: ignore[import]

    grid = np.linspace(0.0, 1.0, 200)
    y = np.exp(-((grid - 0.3) ** 2) / 0.01)
    pdf = _pdf_on_grid(grid, y)
    integral = _integrate_trapezoid(pdf, grid)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_opes_evaluation_grid_resolves_needle_kernel() -> None:
    """Uniform linspace can miss narrow OPES kernels; merged grid must not."""
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    src = root / "src" / "python"
    for p in (scripts, src):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: PLC0415
    from plot_opes_fes import (  # type: ignore[import]
        _integrate_trapezoid,
        _opes_evaluation_grid,
        _pdf_on_grid,
    )

    bias = OPESBias(
        kbt=1.0,
        barrier=5.0,
        biasfactor=10.0,
        pace=1,
        sigma_min=1e-5,
    )
    for step in range(1, 6):
        bias.update(cv_accepted=1.0, mc_step=step)

    lo, hi = 0.0, 3.0
    n_coarse = 50
    grid_uniform = np.linspace(lo, hi, n_coarse)
    p_u = np.array([bias.kde_probability(float(s)) for s in grid_uniform])
    integ_u = _integrate_trapezoid(p_u, grid_uniform)
    assert integ_u == 0.0, "regression guard: uniform grid should miss needle"

    grid_m = _opes_evaluation_grid(bias, lo, hi, n_coarse)
    p_m = np.array([bias.kde_probability(float(s)) for s in grid_m])
    assert np.max(p_m) > 0.0
    integ_m = _integrate_trapezoid(p_m, grid_m)
    assert integ_m > 0.0

    pdf_m = _pdf_on_grid(grid_m, p_m)
    assert np.max(pdf_m) > 0.0
    assert _integrate_trapezoid(pdf_m, grid_m) == pytest.approx(1.0, abs=1e-2)


def test_plot_opes_fes_2d_writes_png(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    src = root / "src" / "python"
    for p in (scripts, src):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: PLC0415
    from plot_opes_fes import plot_opes_fes_2d  # type: ignore[import]

    bias = OPESBias(
        ndim=2, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, sigma_min=1e-4,
    )
    rng = np.random.default_rng(0)
    pts = []
    for step in range(1, 40):
        cv = np.array([0.15 + 0.05 * np.sin(step / 3.0), 0.8 + 0.1 * rng.standard_normal()])
        bias.update(cv_accepted=cv, mc_step=step)
        pts.append(cv)
    cv_samples = np.stack(pts, axis=0)
    out = tmp_path / "fes2d.png"
    plot_opes_fes_2d(
        bias,
        cv_samples,
        out,
        n_per_axis=48,
        pad_fraction=0.2,
        x_min=0.0,
        x_max=0.6,
        y_min=0.4,
        y_max=1.2,
    )
    assert out.is_file()
    assert out.stat().st_size > 2_000

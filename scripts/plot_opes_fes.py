#!/usr/bin/env python3
"""Plot OPES-TPS diagnostics: kernel density, biased vs reweighted FES, histogram overlay.

**1-D** runs: three-panel figure (``opes_fes.png``) — OPES mixture, biased vs
reweighted FES, histogram vs OPES density.

**2-D** runs (comma-separated ``--bias-cv`` in ``run_opes_tps.py``): three-panel
row — OPES mixture surface, reweighted (unbiased) FES, hexbin + OPES contours.

The OPES mixture ``kde_probability`` is divided by ``sum_weights`` inside
``OPESBias`` (bias construction), which does **not** imply ∫P(s)ds = 1.  For
panels 1 and 3 the curve is therefore **renormalized by trapezoidal quadrature**
on the plotted ``grid`` so ∫ P(s) ds ≈ 1 on ``[lo, hi]``, matching Matplotlib's
``hist(..., density=True)``.  The grid is **not** a single uniform ``linspace``:
it merges that with dense samples on each kernel's truncated-Gaussian support
(± ``kernel_cutoff * σ``), so very narrow kernels are not missed.  Use
``--opes-raw-scale`` to plot the internal OPES scale instead (heights not
comparable to the histogram).

Usage::

    # Shortcut: use the TPS output folder (final state if present, else latest checkpoint)
    python scripts/plot_opes_fes.py \\
        --run-dir     opes_tps_out_case2 \\
        --out         opes_tps_out_case2/opes_fes.png

    # Wider x-axis (padding fraction) and fixed range:
    python scripts/plot_opes_fes.py \\
        --run-dir opes_tps_out_case2 --out opes_tps_out_case2/opes_fes.png \\
        --x-pad-fraction 0.25 --x-min 0 --x-max 2.0

    # Explicit paths (after a full run you get opes_state_final.json + cv_values.json)
    python scripts/plot_opes_fes.py \\
        --opes-state  opes_tps_out/opes_state_final.json \\
        --cv-json     opes_tps_out/cv_values.json \\
        --out         opes_fes.png

``--run-dir`` looks for, in order:

* State: ``opes_state_final.json`` if present, **unless** the resolved CV file is
  newer than that final (then ``opes_states/opes_state_latest.json`` automatically).
  Override with ``--prefer-latest-opes-state`` or ``--prefer-opes-final``.
* CVs: ``cv_values.json``, else ``opes_tps_summary.json`` (uses ``tps_steps[].cv_value``)

If the run was killed before the end, you still get checkpoints under ``opes_states/``,
but **``cv_values.json`` is only written when ``run_opes_tps.py`` finishes**.  Without it
(or a saved ``opes_tps_summary.json``), there is no CV time series to plot—re-run TPS
to completion or point ``--cv-json`` at another run's file.

No figure suptitle is drawn (axis labels and colorbars carry the science).

To copy trajectory checkpoints whose **logged endpoint CV** falls in a basin, see
``scripts/export_opes_basin_frames.py``.

"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def _ensure_genai_tps_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "python"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_genai_tps_path()

from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: E402

# Matplotlib mathtext on axes (explicit sizes override rcParams after _apply_plot_rcparams).
_MPL_LABEL_FS = 18
_MPL_TICK_FS = 15
_MPL_LEGEND_FS = 14
_MPL_CBAR_LABEL_FS = 17
_MPL_CONTOUR_LABEL_FS = 11


def _apply_matplotlib_style(*, usetex: bool = False) -> None:
    """Computer Modern for math; optional ``text.usetex`` for full LaTeX (true CM text + math)."""
    import matplotlib as mpl

    base = {
        "font.size": 14,
        "axes.labelsize": _MPL_LABEL_FS,
        "xtick.labelsize": _MPL_TICK_FS,
        "ytick.labelsize": _MPL_TICK_FS,
        "legend.fontsize": _MPL_LEGEND_FS,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # Scientific / contour tick and label numerics use mathtext (CM) when needed.
        "axes.formatter.use_mathtext": True,
    }
    if usetex:
        base.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )
    else:
        base["text.usetex"] = False
        base["font.family"] = "serif"
        # Body text approximates LaTeX serif; all math uses bundled CM via mathtext.fontset.
        base["font.serif"] = ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "serif"]

    mpl.rcParams.update(base)


def _add_panel_letters(axes: object, *, fontsize: int | None = None) -> None:
    """Place ``(a)``, ``(b)``, … as LaTeX/mathtext (Computer Modern) in the top-left."""
    fs = int(fontsize if fontsize is not None else _MPL_LABEL_FS)
    ax_list = np.ravel(np.asarray(axes, dtype=object))
    for idx, ax in enumerate(ax_list):
        ch = chr(ord("a") + idx)
        letter = rf"$\mathbf{{({ch})}}$"
        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            fontsize=fs,
            va="top",
            ha="left",
            zorder=100,
            clip_on=False,
        )


def _style_axis_math(ax: object) -> None:
    ax.tick_params(axis="both", which="major", labelsize=_MPL_TICK_FS)


def _cbar_with_label(fig: object, mappable, ax: object, *, label: str) -> object:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=_MPL_TICK_FS)
    if label:
        cbar.set_label(label, fontsize=_MPL_CBAR_LABEL_FS)
    return cbar


def _scalar_from_cv_value(v: object) -> float | None:
    """Extract one float from ``cv_value`` (scalar or multi-D list from TPS log)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        v = v[0]
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _vector_from_cv_value(v: object, ndim: int) -> np.ndarray | None:
    """Parse ``cv_value`` into shape ``(ndim,)`` for multi-D OPES (ndim >= 2)."""
    if v is None or ndim < 2:
        return None
    if not isinstance(v, (list, tuple)) or len(v) < ndim:
        return None
    row: list[float] = []
    for i in range(ndim):
        try:
            x = float(v[i])
        except (TypeError, ValueError, IndexError):
            return None
        if not np.isfinite(x):
            return None
        row.append(x)
    return np.asarray(row, dtype=np.float64)


def _load_cv_samples_jsonl(jsonl_path: Path, *, ndim: int) -> np.ndarray:
    """Load CVs from ``tps_steps.jsonl`` (one JSON object per line, ``cv_value`` field)."""
    text = jsonl_path.read_text(encoding="utf-8")
    if ndim == 1:
        raw: list[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            v = obj.get("cv_value")
            x = _scalar_from_cv_value(v)
            if x is not None:
                raw.append(x)
        arr = np.asarray(raw, dtype=np.float64)
    else:
        rows: list[np.ndarray] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            v = obj.get("cv_value")
            vec = _vector_from_cv_value(v, ndim)
            if vec is not None:
                rows.append(vec)
        if not rows:
            arr = np.empty((0, ndim), dtype=np.float64)
        else:
            arr = np.stack(rows, axis=0)
    if arr.size == 0:
        raise ValueError(f"{jsonl_path}: no finite cv_value lines")
    return arr


# Matplotlib mathtext labels for ``--bias-cv`` names written by ``run_opes_tps.py``.
_CV_LABEL_BY_NAME: dict[str, str] = {
    "openmm": r"C$\alpha$-RMSD to local min (\AA)",
    "openmm_energy": r"$U_\mathrm{AMBER14}$ (kJ/mol)",
    "rmsd": r"C$\alpha$-RMSD (\AA)",
    "rg": r"$R_\mathrm{g}$ (\AA)",
    "contact_order": r"C$\alpha$ contact order (fraction)",
    "clash_count": r"Steric clash count (C$\alpha$)",
    "ramachandran_outlier": r"Ramachandran outlier fraction",
    "lddt": r"lDDT to reference",
    "ligand_rmsd": r"Ligand pose RMSD (\AA)",
    "ligand_pocket_dist": r"Ligand--pocket distance (\AA)",
    "ligand_contacts": r"Protein--ligand contacts",
    "ligand_hbonds": r"H-bond count (proxy)",
}


def _infer_cv_label_from_run_json(data: dict) -> str | None:
    """Return axis label from ``cv_values.json`` / summary keys if recognized."""
    key = data.get("cv_type")
    if isinstance(key, str) and key in _CV_LABEL_BY_NAME:
        return _CV_LABEL_BY_NAME[key]
    names = data.get("cv_names")
    if isinstance(names, list) and len(names) == 1:
        n0 = names[0]
        if isinstance(n0, str) and n0 in _CV_LABEL_BY_NAME:
            return _CV_LABEL_BY_NAME[n0]
    if isinstance(key, str) and key:
        return key.replace("_", " ")
    return None


def _infer_cv_title_suffix(data: dict) -> str | None:
    """Short descriptor for figure title (e.g. ``contact_order``)."""
    key = data.get("cv_type")
    if isinstance(key, str) and key:
        return key.replace("_", " ")
    names = data.get("cv_names")
    if isinstance(names, list) and names:
        return ", ".join(str(n).replace("_", " ") for n in names)
    return None


def _mathtext_cv_name_axis_label(name: str) -> str:
    """CV column name as ``$\\mathrm{...}$`` (Computer Modern roman via mathtext)."""
    esc = str(name).strip().replace("_", r"\_")
    if not esc:
        esc = r"\,"
    return rf"$\mathrm{{{esc}}}$"


def _infer_cv_labels_2d(data: dict | None) -> tuple[str, str]:
    """Axis labels for the first two biased CVs (matplotlib mathtext)."""
    names = data.get("cv_names") if isinstance(data, dict) else None
    if isinstance(names, list) and len(names) >= 2:
        n0, n1 = str(names[0]), str(names[1])
        lx = _CV_LABEL_BY_NAME.get(n0, _mathtext_cv_name_axis_label(n0))
        ly = _CV_LABEL_BY_NAME.get(n1, _mathtext_cv_name_axis_label(n1))
        return lx, ly
    return r"$s_1$", r"$s_2$"


def _samples_from_json_data(
    data: dict, path_label: Path | str, *, ndim: int
) -> np.ndarray:
    """Build CV sample array ``(N,)`` or ``(N, ndim)`` from summary / ``cv_values.json``."""
    label = str(path_label)
    if "cv_values" in data:
        arr = np.asarray(data["cv_values"], dtype=np.float64)
        if arr.ndim == 1:
            if ndim != 1:
                raise ValueError(
                    f"{label}: bias is {ndim}-D but cv_values is 1-D with shape {arr.shape}."
                )
        elif arr.ndim == 2:
            if arr.shape[1] != ndim:
                raise ValueError(
                    f"{label}: cv_values has shape {arr.shape} but OPES state has ndim={ndim}."
                )
        else:
            raise ValueError(f"{label}: cv_values must be 1-D or 2-D array, got {arr.ndim}-D.")
    elif "tps_steps" in data:
        if ndim == 1:
            raw: list[float] = []
            for step in data["tps_steps"]:
                v = step.get("cv_value")
                x = _scalar_from_cv_value(v)
                if x is not None:
                    raw.append(x)
            arr = np.asarray(raw, dtype=np.float64)
        else:
            rows: list[np.ndarray] = []
            for step in data["tps_steps"]:
                v = step.get("cv_value")
                vec = _vector_from_cv_value(v, ndim)
                if vec is not None:
                    rows.append(vec)
            arr = np.stack(rows, axis=0) if rows else np.empty((0, ndim), dtype=np.float64)
    else:
        raise ValueError(
            f"{label}: need 'cv_values', 'tps_steps', or use a .jsonl step log"
        )
    if arr.size == 0:
        raise ValueError(f"{label}: no finite CV samples")
    return arr


def _load_cv_samples(cv_path: Path, *, ndim: int) -> np.ndarray:
    """Load CV trajectory from JSON summary, ``cv_values.json``, or ``tps_steps.jsonl``."""
    if cv_path.suffix == ".jsonl" or cv_path.name == "tps_steps.jsonl":
        return _load_cv_samples_jsonl(cv_path, ndim=ndim)
    data = json.loads(cv_path.read_text(encoding="utf-8"))
    return _samples_from_json_data(data, cv_path, ndim=ndim)


def _resolve_state_path(
    run_dir: Path,
    *,
    prefer_latest: bool = False,
    prefer_final: bool = False,
    cv_path: Path | None = None,
) -> Path | None:
    """Pick OPES JSON under *run_dir*.

    Default: ``opes_state_final.json`` if it exists, **unless** *cv_path* exists and
    is newer than final — then use ``opes_states/opes_state_latest.json`` so a new
    run (new ``cv_values.json``) is not paired with a stale final from an old run.

    *prefer_latest* / *prefer_final* force that file when present.
    """
    final = run_dir / "opes_state_final.json"
    latest = run_dir / "opes_states" / "opes_state_latest.json"
    have_f, have_l = final.is_file(), latest.is_file()

    if prefer_latest and have_l:
        return latest
    if prefer_final and have_f:
        return final

    if have_f and have_l and cv_path is not None and cv_path.is_file():
        try:
            if cv_path.stat().st_mtime > final.stat().st_mtime + 0.5:
                return latest
        except OSError:
            pass

    if have_f:
        return final
    if have_l:
        return latest
    return None


def _note_auto_latest_vs_stale_final(
    run_dir: Path,
    state_path: Path,
    cv_path: Path,
    *,
    prefer_latest: bool,
    prefer_final: bool,
) -> None:
    """Inform user when we auto-picked latest over an older final."""
    if prefer_latest or prefer_final:
        return
    latest = run_dir / "opes_states" / "opes_state_latest.json"
    final = run_dir / "opes_state_final.json"
    if not (latest.is_file() and final.is_file() and cv_path.is_file()):
        return
    if state_path.resolve() != latest.resolve():
        return
    try:
        if cv_path.stat().st_mtime <= final.stat().st_mtime + 0.5:
            return
    except OSError:
        return
    print(
        "[plot_opes_fes] Note: using opes_states/opes_state_latest.json because "
        f"{cv_path.name} is newer than opes_state_final.json (stale final would "
        "mismatch the current CV trajectory). Use --prefer-opes-final to force "
        "the old final anyway.",
        file=sys.stderr,
    )


def _warn_forced_final_is_stale(
    run_dir: Path,
    state_path: Path,
    cv_path: Path,
    *,
    prefer_final: bool,
) -> None:
    if not prefer_final:
        return
    final = run_dir / "opes_state_final.json"
    if not final.is_file() or state_path.resolve() != final.resolve():
        return
    if not cv_path.is_file():
        return
    if cv_path.stat().st_mtime <= final.stat().st_mtime + 0.5:
        return
    print(
        "[plot_opes_fes] WARNING: --prefer-opes-final selected but "
        f"{cv_path.name} is newer than {final.name}. "
        "OPES kernels may not match this CV — expect a flat / wrong blue curve.",
        file=sys.stderr,
    )


def _check_bias_cv_matches_json(bias: OPESBias, json_doc: dict | None) -> None:
    """Raise if saved OPES ``bias_cv`` disagrees with ``cv_values.json`` metadata."""
    if json_doc is None:
        return
    saved = getattr(bias, "saved_bias_cv", None)
    if not isinstance(saved, str) or not saved.strip():
        return
    doc_type = json_doc.get("cv_type")
    if not isinstance(doc_type, str):
        return
    if doc_type != saved:
        raise ValueError(
            f"OPES state file was built with bias_cv={saved!r} but "
            f"{doc_type!r} in the CV JSON does not match. "
            "The kernel centers live in a different CV space than the x-axis — "
            "OPES P(s) will look like zero. "
            "Use matching --opes-state and --cv-json from the same run, or delete "
            "the stale opes_state_final.json so the plot can use latest checkpoints."
        )


def _warn_kernel_vs_sample_mismatch(bias: OPESBias, cv_samples: np.ndarray) -> None:
    """Heuristic when state files omit ``bias_cv`` (old checkpoints)."""
    if not bias.kernels or cv_samples.size == 0:
        return
    if getattr(bias, "saved_bias_cv", None):
        return
    if cv_samples.ndim == 1:
        centers = np.array(
            [float(np.asarray(k.center, dtype=np.float64).ravel()[0]) for k in bias.kernels],
            dtype=np.float64,
        )
        s_med = float(np.median(cv_samples))
        mad = float(np.median(np.abs(cv_samples - s_med)))
        scale = max(mad, float(np.std(cv_samples)), 1e-12)
        min_d = float(np.min(np.abs(centers - s_med)))
        if min_d > 50.0 * scale:
            warnings.warn(
                f"Kernel centers (median {float(np.median(centers)):g}) look incompatible "
                f"with CV samples (median {s_med:g}); min |center - sample_median| = {min_d:g} "
                f"vs scale ~{scale:g}. OPES P(s) may be zero — check OPES state matches "
                f"this CV trajectory.",
                UserWarning,
                stacklevel=2,
            )
        return

    d = int(cv_samples.shape[1])
    med = np.median(cv_samples, axis=0)
    dev = np.linalg.norm(cv_samples - med, axis=1)
    scale = float(max(np.median(dev), np.std(dev), 1e-12))
    kcent = np.stack(
        [np.asarray(k.center, dtype=np.float64).ravel()[:d] for k in bias.kernels],
        axis=0,
    )
    dists = np.linalg.norm(kcent - med, axis=1)
    min_d = float(np.min(dists))
    if min_d > 50.0 * scale:
        warnings.warn(
            f"Kernel centers look far from CV cloud (min L2 dist to median ~{min_d:g} "
            f"vs scale ~{scale:g}). OPES P(s) may be zero — check OPES state matches "
            "this CV trajectory.",
            UserWarning,
            stacklevel=2,
        )


def _resolve_cv_path(run_dir: Path) -> Path | None:
    for name in ("cv_values.json", "tps_steps.jsonl", "opes_tps_summary.json"):
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """∫ y(x) dx with NumPy 1.x/2.x compatible API."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _pdf_on_grid(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Divide *density* by ∫ density dx on *grid* so the result integrates to ~1."""
    y = np.asarray(density, dtype=np.float64)
    integ = _integrate_trapezoid(y, grid)
    if integ <= 0.0 or not np.isfinite(integ):
        return np.zeros_like(y)
    return y / integ


def _integrate_trapezoid_2d(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """∬ Z(x,y) dx dy with Z shape ``(len(y), len(x))`` (``meshgrid(..., indexing='xy')``)."""
    zx = Z.astype(np.float64, copy=False)
    if hasattr(np, "trapezoid"):
        inner = np.trapezoid(zx, x, axis=1)
        return float(np.trapezoid(inner, y))
    inner = np.trapz(zx, x, axis=1)
    return float(np.trapz(inner, y))


def _pdf_on_mesh(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Divide *Z* by ∬ Z dx dy on the tensor-product grid so the integral is ~1."""
    zz = np.asarray(Z, dtype=np.float64)
    integ = _integrate_trapezoid_2d(zz, x, y)
    if integ <= 0.0 or not np.isfinite(integ):
        return np.zeros_like(zz)
    return zz / integ


def _neg_ln_rho_relative(density: np.ndarray) -> np.ndarray:
    """Dimensionless ``-ln(ρ̂)`` with ``ρ̂ = p / max(p)`` (minimum 0 at the mode)."""
    p = np.asarray(density, dtype=np.float64)
    p = np.clip(p, 1e-300, None)
    p = p / (np.nanmax(p) + 1e-300)
    return -np.log(p)


def _kernel_center_x0(kernel) -> float:
    """Scalar x-coordinate for vertical line at kernel center (1-D OPES)."""
    c = np.asarray(kernel.center, dtype=np.float64).ravel()
    if c.size == 0:
        return 0.0
    return float(c[0])


def _kernel_sigma0(kernel) -> float:
    """First CV-component σ (plot is 1-D along the stored trajectory)."""
    s = np.asarray(kernel.sigma, dtype=np.float64).ravel()
    if s.size == 0:
        return 1.0
    return float(s[0])


def _opes_evaluation_grid(
    bias: OPESBias,
    lo: float,
    hi: float,
    base_n: int,
    *,
    local_points: int = 128,
    max_points: int = 30_000,
) -> np.ndarray:
    """Grid points for ``kde_probability``: uniform span plus dense windows on each kernel.

    Truncated OPES Gaussians are **exactly zero** outside ~±``kernel_cutoff * σ``.
    A fixed ``linspace(lo, hi, n_grid)`` often places **no** sample inside that
    interval when σ is tiny relative to ``hi - lo``, so the trapezoid integral
    vanishes and the renormalized curve becomes all zeros.  Here we merge the
    coarse grid with ``local_points`` samples on each kernel's in-range support.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"Invalid plot range: lo={lo!r}, hi={hi!r}")

    n_base = max(2, int(base_n))
    chunks: list[np.ndarray] = [np.linspace(lo, hi, n_base)]

    nk = len(bias.kernels)
    n_local = int(local_points)
    if nk > 0:
        budget = max(max_points // (nk + 2), 24)
        n_local = max(24, min(n_local, budget))

    k_cut = float(bias.kernel_cutoff)
    for k in bias.kernels:
        c = _kernel_center_x0(k)
        sig0 = _kernel_sigma0(k)
        rad = k_cut * max(sig0, 1e-300)
        a = max(lo, c - rad)
        b = min(hi, c + rad)
        if b > a:
            chunks.append(np.linspace(a, b, n_local))

    grid = np.unique(np.concatenate(chunks))
    grid = grid[(grid >= lo) & (grid <= hi)]
    if grid.size == 0:
        grid = np.linspace(lo, hi, n_base)

    if grid.size > int(max_points):
        idx = np.linspace(0, grid.size - 1, int(max_points))
        idx = np.unique(np.round(idx).astype(np.intp))
        grid = grid[idx]
    return grid


def plot_opes_fes(
    bias: OPESBias,
    cv_samples: np.ndarray,
    out_path: Path,
    *,
    title: str | None = None,
    cv_label: str = r"C$\alpha$-RMSD (Å)",
    cv_descriptor: str | None = None,
    n_grid: int = 400,
    n_hist_bins: int = 36,
    unbiased_fes_scale: float = 1.0,
    x_pad_fraction: float = 0.15,
    x_min: float | None = None,
    x_max: float | None = None,
    fes_y_max: float | None = None,
    density_y_max: float | None = None,
    opes_raw_scale: bool = False,
    usetex: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    _apply_matplotlib_style(usetex=usetex)

    s_min_data = float(np.min(cv_samples))
    s_max_data = float(np.max(cv_samples))
    span_data = max(s_max_data - s_min_data, 0.1)
    pad_frac = float(x_pad_fraction)
    if pad_frac < 0:
        raise ValueError("x_pad_fraction must be non-negative")

    if x_min is not None and x_max is not None:
        lo, hi = float(x_min), float(x_max)
        if lo >= hi:
            raise ValueError(f"x_min ({lo}) must be < x_max ({hi})")
    elif x_min is not None:
        lo = float(x_min)
        hi = s_max_data + pad_frac * max(s_max_data - lo, 0.1)
    elif x_max is not None:
        hi = float(x_max)
        lo = s_min_data - pad_frac * max(hi - s_min_data, 0.1)
    else:
        pad = pad_frac * span_data
        lo = s_min_data - pad
        hi = s_max_data + pad

    n_in_window = int(np.sum((cv_samples >= lo) & (cv_samples <= hi)))
    if n_in_window == 0:
        raise ValueError(
            f"No CV samples fall in the plotted x-range [{lo:g}, {hi:g}]; "
            f"data span is [{s_min_data:g}, {s_max_data:g}]. "
            "Omit --x-min/--x-max to auto-pad from data, or set limits that "
            "include your samples (e.g. contact order often lies in [0, 0.2])."
        )

    grid = _opes_evaluation_grid(bias, lo, hi, n_grid)

    p_opes_internal = np.array([bias.kde_probability(float(s)) for s in grid])
    if opes_raw_scale:
        p_opes_plot = p_opes_internal
    else:
        p_opes_plot = _pdf_on_grid(grid, p_opes_internal)

    # Biased ensemble: unweighted KDE of CV samples along the chain
    try:
        kde_b = gaussian_kde(cv_samples)
        p_bias = kde_b(grid)
    except Exception:
        p_bias = np.histogram(cv_samples, bins=n_grid, range=(lo, hi), density=True)[0]
        p_bias = np.interp(grid, np.linspace(lo, hi, len(p_bias)), p_bias)

    w = bias.reweight_samples(cv_samples)
    try:
        kde_u = gaussian_kde(cv_samples, weights=w)
        p_unb = kde_u(grid)
    except TypeError:
        hist, edges = np.histogram(
            cv_samples, bins=n_grid, range=(lo, hi), weights=w, density=True
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        p_unb = np.interp(grid, centers, hist)

    f_bias = _neg_ln_rho_relative(p_bias)
    f_unb = _neg_ln_rho_relative(p_unb)
    scale_u = float(unbiased_fes_scale)
    f_unb_plot = scale_u * f_unb

    _ = (title, cv_descriptor)  # optional API; no figure suptitle

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- Panel 1: OPES kernel mixture density ---
    ax = axes[0]
    ax.fill_between(grid, p_opes_plot, alpha=0.35, color="#90CAF9", linewidth=0)
    ax.plot(grid, p_opes_plot, color="#1976D2", linewidth=2.0)
    for k in bias.kernels:
        ax.axvline(
            _kernel_center_x0(k),
            color="#FF9800",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
    ax.set_xlabel(cv_label, fontsize=_MPL_LABEL_FS)
    ypdf = (
        r"$\hat{P}_{\mathrm{OPES}}(s)$ (internal scale)"
        if opes_raw_scale
        else r"$\hat{P}_{\mathrm{OPES}}(s)$"
    )
    ax.set_ylabel(ypdf, fontsize=_MPL_LABEL_FS)
    _style_axis_math(ax)
    ax.set_xlim(lo, hi)
    if density_y_max is not None:
        ax.set_ylim(0.0, float(density_y_max))
    else:
        ax.set_ylim(bottom=0.0)

    # --- Panel 2: Biased vs reweighted FES ---
    ax = axes[1]
    ax.set_facecolor("#FCE4EC")
    ax.plot(
        grid,
        f_bias,
        color="#2E7D32",
        linewidth=2.0,
        label=r"Biased: $-\ln\bigl(\hat{\rho}/\hat{\rho}_{\mathrm{max}}\bigr)$",
    )
    unb_label = r"Unbiased: $-\ln\bigl(\hat{\rho}/\hat{\rho}_{\mathrm{max}}\bigr)$"
    if abs(scale_u - 1.0) > 1e-12:
        unb_label = rf"{unb_label} ($\times {scale_u:g}$)"
    ax.plot(grid, f_unb_plot, color="#C62828", linewidth=2.0, label=unb_label)
    ax.set_xlabel(cv_label, fontsize=_MPL_LABEL_FS)
    ax.set_ylabel(
        r"$-\ln\bigl(\hat{\rho}(s)/\hat{\rho}_{\mathrm{max}}\bigr)$",
        fontsize=_MPL_LABEL_FS,
    )
    _style_axis_math(ax)
    ax.legend(fontsize=_MPL_LEGEND_FS, loc="best")
    ax.set_xlim(lo, hi)
    if fes_y_max is not None:
        y0, _ = ax.get_ylim()
        ax.set_ylim(y0, float(fes_y_max))

    # --- Panel 3: histogram vs OPES P(s) ---
    ax = axes[2]
    ax.hist(
        cv_samples,
        bins=n_hist_bins,
        range=(lo, hi),
        density=True,
        alpha=0.55,
        color="#7B1FA2",
        edgecolor="white",
        linewidth=0.5,
        label=rf"$\{{s_i\}}$ ($N={len(cv_samples)}$)",
    )
    opes_label = (
        r"$\hat{P}_{\mathrm{OPES}}(s)$ (internal)"
        if opes_raw_scale
        else r"$\hat{P}_{\mathrm{OPES}}(s)$"
    )
    ax.plot(grid, p_opes_plot, color="#1976D2", linewidth=2.0, label=opes_label)
    ax.set_xlabel(cv_label, fontsize=_MPL_LABEL_FS)
    ax.set_ylabel(r"$\mathrm{PDF}(s)$", fontsize=_MPL_LABEL_FS)
    _style_axis_math(ax)
    ax.legend(fontsize=_MPL_LEGEND_FS, loc="upper right")
    ax.set_xlim(lo, hi)
    if density_y_max is not None:
        ax.set_ylim(0.0, float(density_y_max))
    else:
        ax.set_ylim(bottom=0.0)

    _add_panel_letters(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_opes_fes_2d(
    bias: OPESBias,
    cv_samples: np.ndarray,
    out_path: Path,
    *,
    title: str | None = None,
    label_x: str = r"$s_1$",
    label_y: str = r"$s_2$",
    cv_descriptor: str | None = None,
    n_per_axis: int = 96,
    pad_fraction: float = 0.12,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    opes_raw_scale: bool = False,
    ln_rho_n_levels: int = 42,
    ln_rho_cap_percentile: float = 92.0,
    usetex: bool = False,
) -> None:
    """Three-panel row for ``ndim==2`` OPES: mixture PDF, reweighted -ln(rho) map, hexbin + contours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    _apply_matplotlib_style(usetex=usetex)

    if bias.ndim != 2:
        raise ValueError(f"plot_opes_fes_2d requires bias.ndim==2, got {bias.ndim}")
    arr = np.asarray(cv_samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"cv_samples must have shape (N, 2), got {arr.shape}")

    n_axis = int(max(24, min(280, n_per_axis)))

    s0_min, s0_max = float(np.min(arr[:, 0])), float(np.max(arr[:, 0]))
    s1_min, s1_max = float(np.min(arr[:, 1])), float(np.max(arr[:, 1]))
    span0 = max(s0_max - s0_min, 1e-6)
    span1 = max(s1_max - s1_min, 1e-6)
    pf = float(pad_fraction)
    if pf < 0:
        raise ValueError("pad_fraction must be non-negative")

    def _lo_hi(
        smin: float, smax: float, span: float, lo: float | None, hi: float | None
    ) -> tuple[float, float]:
        if lo is not None and hi is not None:
            if lo >= hi:
                raise ValueError(f"axis limits invalid: {lo} >= {hi}")
            return float(lo), float(hi)
        if lo is not None:
            return float(lo), smax + pf * max(smax - lo, span * 0.1)
        if hi is not None:
            return smin - pf * max(hi - smin, span * 0.1), float(hi)
        pad0 = pf * span
        return smin - pad0, smax + pad0

    lo0, hi0 = _lo_hi(s0_min, s0_max, span0, x_min, x_max)
    lo1, hi1 = _lo_hi(s1_min, s1_max, span1, y_min, y_max)

    in_win = (
        (arr[:, 0] >= lo0)
        & (arr[:, 0] <= hi0)
        & (arr[:, 1] >= lo1)
        & (arr[:, 1] <= hi1)
    )
    if not np.any(in_win):
        raise ValueError(
            f"No CV samples in plot window "
            f"x∈[{lo0:g},{hi0:g}], y∈[{lo1:g},{hi1:g}]; "
            f"data span x∈[{s0_min:g},{s0_max:g}], y∈[{s1_min:g},{s1_max:g}]."
        )

    gx = np.linspace(lo0, hi0, n_axis)
    gy = np.linspace(lo1, hi1, n_axis)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    flat = np.vstack([X.ravel(), Y.ravel()])
    p_opes_internal = np.array(
        [bias.kde_probability(flat[:, i]) for i in range(flat.shape[1])],
        dtype=np.float64,
    ).reshape(X.shape)
    if opes_raw_scale:
        p_opes_plot = p_opes_internal
    else:
        p_opes_plot = _pdf_on_mesh(p_opes_internal, gx, gy)

    w = bias.reweight_samples(arr)
    try:
        kde_u = gaussian_kde(arr.T, weights=w)
        p_unb = kde_u(flat).reshape(X.shape)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        p_unb = np.full_like(p_opes_plot, np.nan)

    f_unb_m = _neg_ln_rho_relative(p_unb)

    _ = (title, cv_descriptor)  # optional API; no figure suptitle

    # Square axes boxes (physical aspect), three columns + colorbars: width >> height.
    try:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), layout="constrained")
    except TypeError:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), constrained_layout=True)
    kw = dict(levels=18, cmap="viridis", extend="both")

    ax = axes[0]
    cf0 = ax.contourf(X, Y, p_opes_plot, **kw)
    kc = np.array([np.asarray(k.center, dtype=np.float64)[:2] for k in bias.kernels])
    if kc.size > 0:
        ax.scatter(
            kc[:, 0], kc[:, 1], c="none", edgecolors="#FF9800", s=28, linewidths=1.0,
            label=r"$\mathrm{kernel\ } c_k$", zorder=5,
        )
    ax.set_xlabel(label_x, fontsize=_MPL_LABEL_FS)
    ax.set_ylabel(label_y, fontsize=_MPL_LABEL_FS)
    _style_axis_math(ax)
    ax.set_xlim(lo0, hi0)
    ax.set_ylim(lo1, hi1)
    ax.set_box_aspect(1.0)
    cbar0_lbl = (
        r"$\hat{P}_{\mathrm{OPES}}(s_1,s_2)$ (internal scale)"
        if opes_raw_scale
        else r"$\hat{P}_{\mathrm{OPES}}(s_1,s_2)$"
    )
    _cbar_with_label(fig, cf0, ax, label=cbar0_lbl)
    ax.legend(fontsize=_MPL_LEGEND_FS, loc="best")

    ax = axes[1]
    if np.all(np.isfinite(p_unb)):
        z_ln = f_unb_m
        fin = z_ln[np.isfinite(z_ln)]
        vmin_ln = float(np.nanmin(z_ln))
        vmax_full = float(np.nanmax(z_ln))
        cap_p = float(np.clip(ln_rho_cap_percentile, 50.0, 99.999))
        vmax_cap = float(np.percentile(fin, cap_p)) if fin.size else vmax_full
        vmax_plot = min(max(vmax_cap, vmin_ln + 1e-12), vmax_full)
        if vmax_plot <= vmin_ln + 1e-12:
            vmax_plot = vmax_full
        nlev = max(8, int(ln_rho_n_levels))
        levels_f = np.linspace(vmin_ln, vmax_plot, nlev)
        extend_ln = "max" if vmax_full > vmax_plot + 1e-6 * max(1.0, abs(vmax_full)) else "neither"
        cf1 = ax.contourf(
            X, Y, z_ln,
            levels=levels_f,
            cmap="viridis",
            extend=extend_ln,
        )
        n_line = max(14, min(36, nlev))
        levels_line = np.linspace(vmin_ln, vmax_plot, n_line)
        ax.contour(
            X, Y, z_ln,
            levels=levels_line,
            colors="k",
            linewidths=0.45,
            alpha=0.38,
        )
        ln_lbl = r"$-\ln\bigl(\hat{\rho}/\hat{\rho}_{\mathrm{max}}\bigr)$"
        ax.set_xlabel(label_x, fontsize=_MPL_LABEL_FS)
        ax.set_ylabel(label_y, fontsize=_MPL_LABEL_FS)
        _style_axis_math(ax)
        ax.set_xlim(lo0, hi0)
        ax.set_ylim(lo1, hi1)
        ax.set_box_aspect(1.0)
        _cbar_with_label(fig, cf1, ax, label=ln_lbl)
    else:
        ax.text(
            0.5,
            0.5,
            r"$\mathrm{Reweighted\ KDE\ failed}$",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=_MPL_LABEL_FS,
        )
        ax.set_xlabel(label_x, fontsize=_MPL_LABEL_FS)
        ax.set_ylabel(label_y, fontsize=_MPL_LABEL_FS)
        _style_axis_math(ax)
        ax.set_xlim(lo0, hi0)
        ax.set_ylim(lo1, hi1)
        ax.set_box_aspect(1.0)

    ax = axes[2]
    hb = ax.hexbin(
        arr[:, 0], arr[:, 1], gridsize=42, mincnt=1, cmap="Purples", bins="log",
    )
    cs = ax.contour(
        X, Y, p_opes_plot, levels=8, colors="#1976D2", linewidths=1.0, alpha=0.85,
    )
    ax.clabel(
        cs,
        inline=True,
        fontsize=_MPL_CONTOUR_LABEL_FS,
        fmt="%.2g",
        use_clabeltext=True,
    )
    ax.set_xlabel(label_x, fontsize=_MPL_LABEL_FS)
    ax.set_ylabel(label_y, fontsize=_MPL_LABEL_FS)
    _style_axis_math(ax)
    ax.set_xlim(lo0, hi0)
    ax.set_ylim(lo1, hi1)
    ax.set_box_aspect(1.0)
    _cbar_with_label(
        fig, hb, ax,
        label=r"$\mathrm{counts\ (log\ scale)}$",
    )
    _add_panel_letters(axes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "TPS output root: picks opes_state_final.json or "
            "opes_states/opes_state_latest.json, and cv_values.json or "
            "opes_tps_summary.json"
        ),
    )
    ap.add_argument(
        "--opes-state",
        type=Path,
        default=None,
        help="OPES state JSON (overrides --run-dir state resolution).",
    )
    ap.add_argument(
        "--cv-json",
        type=Path,
        default=None,
        help="cv_values.json or opes_tps_summary.json (overrides --run-dir).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("opes_fes.png"),
        help="Output PNG path (default: ./opes_fes.png)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Ignored (no figure title is drawn; kept for script compatibility).",
    )
    ap.add_argument(
        "--cv-label",
        type=str,
        default=None,
        help=(
            "1-D: x-axis label.  2-D: first CV axis label (matplotlib mathtext). "
            "Default: infer from ``cv_values.json`` metadata when available."
        ),
    )
    ap.add_argument(
        "--cv-label-y",
        type=str,
        default=None,
        help="2-D only: second CV axis label (default: infer from ``cv_names``[1]).",
    )
    ap.add_argument("--n-grid", type=int, default=400, help="1-D: grid points for curves.")
    ap.add_argument(
        "--n-per-axis-2d",
        type=int,
        default=96,
        help="2-D: evaluation grid points per axis (96 → 96×96 OPES/KDE surfaces).",
    )
    ap.add_argument(
        "--pad-fraction-2d",
        type=float,
        default=None,
        help=(
            "2-D: margin on each side as a fraction of span (default: same as "
            "--x-pad-fraction)."
        ),
    )
    ap.add_argument("--hist-bins", type=int, default=36, help="Histogram bins (panel 3).")
    ap.add_argument(
        "--unbiased-fes-scale",
        type=float,
        default=1.0,
        help=(
            "1-D only: multiply the reweighted $-\\ln(\\hat\\rho/\\hat\\rho_\\mathrm{max})$ "
            "curve in panel 2 (default: 1, no scaling). A value $\\neq 1$ appends "
            "$(\\times \\ldots)$ to the legend."
        ),
    )
    ap.add_argument(
        "--ln-rho-n-levels-2d",
        type=int,
        default=42,
        help=(
            "2-D middle panel: number of filled contour levels for reweighted "
            "$-\\ln(\\hat\\rho/\\hat\\rho_\\mathrm{max})$ (default: 42)."
        ),
    )
    ap.add_argument(
        "--ln-rho-cap-percentile-2d",
        type=float,
        default=92.0,
        help=(
            "2-D middle panel: cap color scale at this percentile of the field so "
            "outliers do not flatten the wells (default: 92)."
        ),
    )
    ap.add_argument(
        "--x-pad-fraction",
        type=float,
        default=0.15,
        help=(
            "Extra x-axis margin as a fraction of the CV span (each side), when "
            "--x-min/--x-max are not both set (default: 0.15)."
        ),
    )
    ap.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Fix left edge of all panels (Å or CV units); may combine with --x-max.",
    )
    ap.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Fix right edge of all panels (Å or CV units); may combine with --x-min.",
    )
    ap.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="2-D only: lower limit for the second CV axis.",
    )
    ap.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="2-D only: upper limit for the second CV axis.",
    )
    ap.add_argument(
        "--fes-y-max",
        type=float,
        default=None,
        help="Optional top limit for free-energy y-axis (panel 2 only, kBT units).",
    )
    ap.add_argument(
        "--density-y-max",
        type=float,
        default=None,
        help="Optional top limit for density y-axis (panels 1 and 3).",
    )
    ap.add_argument(
        "--opes-raw-scale",
        action="store_true",
        help=(
            "Plot OPES kde_probability without ∫=1 renormalization on the grid "
            "(internal sum_weights scale; not comparable to hist density=True)."
        ),
    )
    ap.add_argument(
        "--usetex",
        action="store_true",
        help=(
            "Use LaTeX (text.usetex) for text — full Computer Modern document fonts; "
            "requires a working ``latex`` on PATH."
        ),
    )
    ap.add_argument(
        "--prefer-latest-opes-state",
        action="store_true",
        help=(
            "With --run-dir, always load opes_states/opes_state_latest.json when "
            "it exists (even if opes_state_final.json is newer)."
        ),
    )
    ap.add_argument(
        "--prefer-opes-final",
        action="store_true",
        help=(
            "With --run-dir, always load opes_state_final.json when it exists, "
            "even if cv_values.json is newer (can mismatch after changing --bias-cv)."
        ),
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    if run_dir is not None and not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    state_path = args.opes_state.expanduser().resolve() if args.opes_state else None
    cv_path = args.cv_json.expanduser().resolve() if args.cv_json else None

    if run_dir is not None:
        if cv_path is None:
            cv_path = _resolve_cv_path(run_dir)
        if state_path is None:
            state_path = _resolve_state_path(
                run_dir,
                prefer_latest=bool(args.prefer_latest_opes_state),
                prefer_final=bool(args.prefer_opes_final),
                cv_path=cv_path,
            )

    if state_path is None:
        print(
            "No OPES state file found.  Use --opes-state PATH or --run-dir DIR "
            "(expects opes_state_final.json or opes_states/opes_state_latest.json).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not state_path.is_file():
        print(f"Not found: {state_path}", file=sys.stderr)
        sys.exit(1)

    if cv_path is None:
        print(
            "No CV trajectory file found.\n"
            "  • cv_values.json and tps_steps.jsonl are updated during run_opes_tps.py "
            "(incremental); tps_steps.jsonl is written every MC step.\n"
            "  • You can also use opes_tps_summary.json or pass --cv-json.\n"
            "  • Otherwise use --cv-json from another run.",
            file=sys.stderr,
        )
        if run_dir is not None:
            print(f"  (looked under {run_dir})", file=sys.stderr)
        sys.exit(1)
    if not cv_path.is_file():
        print(f"Not found: {cv_path}", file=sys.stderr)
        sys.exit(1)

    if run_dir is not None:
        _note_auto_latest_vs_stale_final(
            run_dir,
            state_path,
            cv_path,
            prefer_latest=bool(args.prefer_latest_opes_state),
            prefer_final=bool(args.prefer_opes_final),
        )
        _warn_forced_final_is_stale(
            run_dir,
            state_path,
            cv_path,
            prefer_final=bool(args.prefer_opes_final),
        )

    bias = OPESBias.load_state(state_path)
    json_doc: dict | None = None
    try:
        if cv_path.suffix == ".jsonl" or cv_path.name == "tps_steps.jsonl":
            cv_samples = _load_cv_samples(cv_path, ndim=bias.ndim)
        else:
            json_doc = json.loads(cv_path.read_text(encoding="utf-8"))
            cv_samples = _samples_from_json_data(json_doc, cv_path, ndim=bias.ndim)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{cv_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"{cv_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _check_bias_cv_matches_json(bias, json_doc)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(1)
    _warn_kernel_vs_sample_mismatch(bias, cv_samples)

    inferred_label = (
        _infer_cv_label_from_run_json(json_doc) if json_doc is not None else None
    )
    cv_descriptor = (
        _infer_cv_title_suffix(json_doc) if json_doc is not None else None
    )

    out_resolved = args.out.expanduser().resolve()

    if bias.ndim == 1:
        cv_label = args.cv_label
        if cv_label is None:
            cv_label = inferred_label or r"C$\alpha$-RMSD (Å)"
        plot_opes_fes(
            bias,
            cv_samples,
            out_resolved,
            title=args.title,
            cv_label=cv_label,
            cv_descriptor=cv_descriptor,
            n_grid=args.n_grid,
            n_hist_bins=args.hist_bins,
            unbiased_fes_scale=args.unbiased_fes_scale,
            x_pad_fraction=args.x_pad_fraction,
            x_min=args.x_min,
            x_max=args.x_max,
            fes_y_max=args.fes_y_max,
            density_y_max=args.density_y_max,
            opes_raw_scale=args.opes_raw_scale,
            usetex=bool(args.usetex),
        )
    elif bias.ndim == 2:
        lx, ly = _infer_cv_labels_2d(json_doc if isinstance(json_doc, dict) else {})
        if args.cv_label is not None:
            lx = args.cv_label
        if args.cv_label_y is not None:
            ly = args.cv_label_y
        pad_2d = (
            float(args.pad_fraction_2d)
            if args.pad_fraction_2d is not None
            else float(args.x_pad_fraction)
        )
        plot_opes_fes_2d(
            bias,
            cv_samples,
            out_resolved,
            title=args.title,
            label_x=lx,
            label_y=ly,
            cv_descriptor=cv_descriptor,
            n_per_axis=int(args.n_per_axis_2d),
            pad_fraction=pad_2d,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            opes_raw_scale=args.opes_raw_scale,
            ln_rho_n_levels=int(args.ln_rho_n_levels_2d),
            ln_rho_cap_percentile=float(args.ln_rho_cap_percentile_2d),
            usetex=bool(args.usetex),
        )
    else:
        print(
            f"plot_opes_fes supports ndim 1 or 2 only; this OPES state has ndim={bias.ndim}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Wrote {out_resolved}")


if __name__ == "__main__":
    main()

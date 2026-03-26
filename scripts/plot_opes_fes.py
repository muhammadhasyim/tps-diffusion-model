#!/usr/bin/env python3
"""Plot OPES-TPS diagnostics: kernel density, biased vs reweighted FES, histogram overlay.

Reproduces the three-panel figure style used for ``opes_fes.png`` (OPES kernel
density with center markers, free energy from biased vs reweighted CV samples,
and raw histogram vs OPES P(s)).

Usage::

    # Shortcut: use the TPS output folder (final state if present, else latest checkpoint)
    python scripts/plot_opes_fes.py \\
        --run-dir     opes_tps_out_case2 \\
        --out         opes_tps_out_case2/opes_fes.png

    # Explicit paths (after a full run you get opes_state_final.json + cv_values.json)
    python scripts/plot_opes_fes.py \\
        --opes-state  opes_tps_out/opes_state_final.json \\
        --cv-json     opes_tps_out/cv_values.json \\
        --out         opes_fes.png

``--run-dir`` looks for, in order:

* State: ``opes_state_final.json``, else ``opes_states/opes_state_latest.json``
* CVs: ``cv_values.json``, else ``opes_tps_summary.json`` (uses ``tps_steps[].cv_value``)

If the run was killed before the end, you still get checkpoints under ``opes_states/``,
but **``cv_values.json`` is only written when ``run_opes_tps.py`` finishes**.  Without it
(or a saved ``opes_tps_summary.json``), there is no CV time series to plot—re-run TPS
to completion or point ``--cv-json`` at another run's file.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_genai_tps_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "python"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_genai_tps_path()

from genai_tps.enhanced_sampling.opes_bias import OPESBias  # noqa: E402


def _load_cv_samples(cv_path: Path) -> np.ndarray:
    """Load CV trajectory from ``cv_values.json`` or ``opes_tps_summary.json``."""
    data = json.loads(cv_path.read_text())
    if "cv_values" in data:
        arr = np.asarray(data["cv_values"], dtype=np.float64)
    elif "tps_steps" in data:
        raw = []
        for step in data["tps_steps"]:
            v = step.get("cv_value")
            if v is None:
                continue
            x = float(v)
            if np.isfinite(x):
                raw.append(x)
        arr = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(
            f"{cv_path}: need 'cv_values' (cv_values.json) or "
            "'tps_steps' (opes_tps_summary.json)"
        )
    if arr.size == 0:
        raise ValueError(f"{cv_path}: no finite CV samples")
    return arr


def _resolve_state_path(run_dir: Path) -> Path | None:
    final = run_dir / "opes_state_final.json"
    latest = run_dir / "opes_states" / "opes_state_latest.json"
    if final.is_file():
        return final
    if latest.is_file():
        return latest
    return None


def _resolve_cv_path(run_dir: Path) -> Path | None:
    for name in ("cv_values.json", "opes_tps_summary.json"):
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _fes_from_density(
    grid: np.ndarray,
    density: np.ndarray,
    kbt: float,
) -> np.ndarray:
    """F(s) = -kT ln(p(s) / max(p)), shape for relative free energy (min 0)."""
    p = np.clip(density, 1e-300, None)
    p = p / (np.nanmax(p) + 1e-300)
    return -float(kbt) * np.log(p)


def plot_opes_fes(
    bias: OPESBias,
    cv_samples: np.ndarray,
    out_path: Path,
    *,
    title: str | None = None,
    cv_label: str = r"C$\alpha$-RMSD (Å)",
    n_grid: int = 400,
    n_hist_bins: int = 36,
    unbiased_fes_scale: float = 10.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    kbt = float(bias.kbt)
    s_min = float(np.min(cv_samples))
    s_max = float(np.max(cv_samples))
    pad = 0.05 * max(s_max - s_min, 0.1)
    lo = s_min - pad
    hi = s_max + pad
    grid = np.linspace(lo, hi, n_grid)

    p_opes = np.array([bias.kde_probability(float(s)) for s in grid])

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

    f_bias = _fes_from_density(grid, p_bias, kbt)
    f_unb = _fes_from_density(grid, p_unb, kbt)

    if title is None:
        gamma = bias.biasfactor
        gstr = "∞" if np.isinf(gamma) else f"{gamma:g}"
        title = (
            rf"OPES-TPS: C$\alpha$-RMSD CV "
            rf"($\gamma$={gstr}, barrier={bias.barrier:.1f} $k_\mathrm{{B}}T$, "
            f"{bias.counter} depositions, {bias.n_kernels} kernels)"
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- Panel 1: OPES kernel mixture density ---
    ax = axes[0]
    ax.fill_between(grid, p_opes, alpha=0.35, color="#90CAF9", linewidth=0)
    ax.plot(grid, p_opes, color="#1976D2", linewidth=2.0)
    for k in bias.kernels:
        ax.axvline(k.center, color="#FF9800", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel(cv_label, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("OPES exploration density", fontsize=11)
    ax.set_xlim(lo, hi)

    # --- Panel 2: Biased vs reweighted FES ---
    ax = axes[1]
    ax.set_facecolor("#FCE4EC")
    ax.plot(grid, f_bias, color="#2E7D32", linewidth=2.0, label="Biased FES")
    ax.plot(
        grid,
        unbiased_fes_scale * f_unb,
        color="#C62828",
        linewidth=2.0,
        label=rf"Unbiased FES ($\times${unbiased_fes_scale:g})",
    )
    ax.set_xlabel(cv_label, fontsize=11)
    ax.set_ylabel(r"Free energy ($k_\mathrm{B}T$)", fontsize=11)
    ax.set_title("Free energy surface", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.set_xlim(lo, hi)

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
        label=rf"Raw CV ($n$={len(cv_samples)})",
    )
    ax.plot(grid, p_opes, color="#1976D2", linewidth=2.0, label="OPES P(s)")
    ax.set_xlabel(cv_label, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Histogram vs OPES density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(lo, hi)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
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
        help="Override figure suptitle (default: built from OPES state).",
    )
    ap.add_argument(
        "--cv-label",
        type=str,
        default=r"C$\alpha$-RMSD (Å)",
        help="X-axis label (matplotlib mathtext allowed).",
    )
    ap.add_argument("--n-grid", type=int, default=400, help="Grid points for curves.")
    ap.add_argument("--hist-bins", type=int, default=36, help="Histogram bins (panel 3).")
    ap.add_argument(
        "--unbiased-fes-scale",
        type=float,
        default=10.0,
        help="Multiply unbiased FES curve in panel 2 for visibility (default: 10).",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else None
    if run_dir is not None and not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    state_path = args.opes_state.expanduser().resolve() if args.opes_state else None
    cv_path = args.cv_json.expanduser().resolve() if args.cv_json else None

    if run_dir is not None:
        if state_path is None:
            state_path = _resolve_state_path(run_dir)
        if cv_path is None:
            cv_path = _resolve_cv_path(run_dir)

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
            "  • cv_values.json is only written when run_opes_tps.py exits normally.\n"
            "  • If you have opes_tps_summary.json in the same folder, place it there "
            "or pass --cv-json.\n"
            "  • Otherwise re-run TPS to completion or use --cv-json from another run.",
            file=sys.stderr,
        )
        if run_dir is not None:
            print(f"  (looked under {run_dir})", file=sys.stderr)
        sys.exit(1)
    if not cv_path.is_file():
        print(f"Not found: {cv_path}", file=sys.stderr)
        sys.exit(1)

    bias = OPESBias.load_state(state_path)
    try:
        cv_samples = _load_cv_samples(cv_path)
    except ValueError as exc:
        print(f"{cv_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    plot_opes_fes(
        bias,
        cv_samples,
        args.out.expanduser().resolve(),
        title=args.title,
        cv_label=args.cv_label,
        n_grid=args.n_grid,
        n_hist_bins=args.hist_bins,
        unbiased_fes_scale=args.unbiased_fes_scale,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()

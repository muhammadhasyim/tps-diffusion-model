#!/usr/bin/env python3
"""Plot ``fes_reweighted_*.dat`` from PLUMED ``FES_from_Reweighting.py`` to PNG.

Use real Stage-0 output only: either pass ``--fes`` to an existing ``.dat``, or
``--campaign`` (same root as ``00_generate_reference_data.py --out``) to run
the reweighting step on each case's ``openmm_opes_md/opes_states/COLVAR`` and
write ``fes_reweighted_2d.png`` beside the ``.dat`` (same defaults as
``01b_plumed_colvar_fes.py``).

For OPES diagnostics matching ``opes_tps_ligand_2d_fes.png`` (kernel density,
FES, COLVAR histogram), use ``--triptych`` (2D, pcolormesh histogram) or
``--triptych-opes-style`` (same inputs: **normalized** kernel PDF + markers,
``-ln(ρ/ρ_max)`` FES, **hexbin** log counts + F contours — matches
``scripts/plot_opes_fes.py`` ``plot_opes_fes_2d``).  Use ``--triptych-3d`` for three
CV rows × three panels (Boltzmann marginals).  Use ``--triptych-3d-opes-style`` for the
same 3×3 layout with **volume slices** and the OPES-style columns (normalized kernel slice,
``-ln(ρ/ρ_max)``, hexbin + *F* contours).  Use ``--dashboard-3d`` for
orthogonal **slices** through the 3D reweighted FES plus kernel mixture and
deposit histograms.

**Note:** The middle FES panel is intentionally smooth: OPES merges kernels in
CV space (watch ``opes.nker`` in ``COLVAR``), and reweighting convolves with a
KDE bandwidth — many kernel markers in the left panel do not imply a jagged FES.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

_CASE_NAMES = [
    "case1_mek1_fzc_novel",
    "case2_cdk2_atp_wildtype",
    "case3_cdk2_atp_packed",
]


def _resolve_campaign_out_root(raw: Path) -> Path:
    """Same resolution rules as ``01b_plumed_colvar_fes.py``."""
    expanded = raw.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_root = (Path.cwd() / expanded).resolve()
    repo_root = (_REPO_ROOT / expanded).resolve()

    def _looks_like_stage0_root(p: Path) -> bool:
        if not p.is_dir():
            return False
        return any((p / name / "openmm_opes_md").is_dir() for name in _CASE_NAMES)

    if _looks_like_stage0_root(cwd_root):
        return cwd_root
    if _looks_like_stage0_root(repo_root):
        return repo_root
    return cwd_root


def _marginal_plot_kwargs(
    args: argparse.Namespace,
    *,
    colvar_for_mask: Path | None = None,
    fes_path: Path | None = None,
) -> dict:
    """Forward 3D marginal masking / axis padding to :func:`plot_fes_dat_to_png`."""
    sig_parts = [float(x.strip()) for x in args.marginal_mask_sigma.split(",") if x.strip()]
    sig3: tuple[float, float, float] | None
    if len(sig_parts) == 3:
        sig3 = (sig_parts[0], sig_parts[1], sig_parts[2])
    else:
        sig3 = None
    mask_path: Path | None = None
    if args.marginal_colvar_mask is not None:
        mask_path = args.marginal_colvar_mask.expanduser().resolve()
    elif args.mask_marginals_with_colvar:
        if colvar_for_mask is not None:
            mask_path = colvar_for_mask
        elif fes_path is not None:
            c = fes_path.parent / "COLVAR"
            if c.is_file():
                mask_path = c
    return {
        "marginal_colvar_mask_path": mask_path,
        "marginal_colvar_skiprows": int(args.marginal_colvar_skiprows),
        "marginal_mask_nsigma": float(args.marginal_mask_nsigma),
        "marginal_mask_sigmas": sig3,
        "marginal_axis_pad_fraction": float(args.marginal_axis_pad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PLUMED reweighted FES .dat (from 01b or --campaign).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--fes",
        type=Path,
        default=None,
        help="Path to an existing fes_reweighted_2d.dat (or similar).",
    )
    src.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="Stage-0 campaign root (same as 00 --out); runs 01b-equivalent FES then plots.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (--fes mode only; default: same stem as --fes with .png).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="1,2,3",
        help="Comma-separated case numbers (--campaign only).",
    )
    parser.add_argument(
        "--outfile-name",
        type=str,
        default="fes_reweighted_2d.dat",
        help="FES filename under opes_states/ (--campaign only).",
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Kelvin.")
    parser.add_argument(
        "--sigma",
        type=str,
        default="0.3,0.5",
        help="KDE bandwidth(s), comma-separated (match OPES SIGMA).",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="lig_rmsd,lig_dist",
        help="Comma-separated CV names matching COLVAR FIELDS.",
    )
    parser.add_argument("--bias", type=str, default="opes.bias", help="Bias column.")
    parser.add_argument("--grid-min", type=str, default=None)
    parser.add_argument("--grid-max", type=str, default=None)
    parser.add_argument("--grid-bin", type=str, default="100,100")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=98.0,
        help="Cap 2D color scale at this percentile of finite FES values.",
    )
    parser.add_argument(
        "--mask-marginals-with-colvar",
        action="store_true",
        help=(
            "3D FES marginal plots only: mask grid points farther than "
            "--marginal-mask-nsigma σ (per CV, see --marginal-mask-sigma) from "
            "any COLVAR sample (white = unsampled). For --campaign, uses each "
            "case's COLVAR; for --fes, uses COLVAR beside the .dat in opes_states/."
        ),
    )
    parser.add_argument(
        "--marginal-colvar-mask",
        type=Path,
        default=None,
        help=(
            "3D marginals only: explicit COLVAR path for masking (overrides "
            "--mask-marginals-with-colvar sibling resolution)."
        ),
    )
    parser.add_argument(
        "--marginal-mask-nsigma",
        type=float,
        default=4.0,
        help="3D marginals: σ-scaled distance beyond which FES is masked (white).",
    )
    parser.add_argument(
        "--marginal-mask-sigma",
        type=str,
        default="0.3,0.5,1.0",
        help="Three comma-separated OPES SIGMA values matching 3D FES CV order in .dat.",
    )
    parser.add_argument(
        "--marginal-axis-pad",
        type=float,
        default=0.0,
        help="3D marginals: expand axis limits by this fraction of each axis range.",
    )
    parser.add_argument(
        "--marginal-colvar-skiprows",
        type=int,
        default=0,
        help="COLVAR data rows to skip after header when masking 3D marginals.",
    )
    parser.add_argument(
        "--triptych",
        action="store_true",
        help=(
            "Write a 3-panel PNG (log kernel pcolormesh + centers, FES, COLVAR histogram2d). "
            "Requires --fes, --kernels, --colvar, and --out."
        ),
    )
    parser.add_argument(
        "--triptych-opes-style",
        action="store_true",
        help=(
            "Like --triptych but matches internal OPES-TPS 2D layout: normalized "
            "kernel PDF + markers, -ln(ρ/ρ_max) FES, hexbin (log) + F contours. "
            "Requires 2D --fes, --kernels, --colvar, and --out."
        ),
    )
    parser.add_argument(
        "--triptych-3d",
        action="store_true",
        help=(
            "Write a 3×3-panel PNG: for each CV pair, kernel density + centers, "
            "Boltzmann-marginal FES, and COLVAR histogram (same idea as --triptych "
            "for 2D). Requires 3D FES .dat, --fes, --kernels, --colvar, and --out."
        ),
    )
    parser.add_argument(
        "--triptych-3d-opes-style",
        action="store_true",
        help=(
            "Write a 3×3 PNG like --triptych-opes-style but for 3D FES: each row is an "
            "orthogonal slice (mid third CV); columns are normalized kernel mixture on the "
            "slice, -ln(ρ/ρ_max) from the slice F, and hexbin + F contours. "
            "Requires 3D --fes, --kernels, --colvar, and --out."
        ),
    )
    parser.add_argument(
        "--dashboard-3d",
        action="store_true",
        help=(
            "Write a 3×3 PNG: per orthogonal slice through the 3D reweighted FES volume, "
            "OPES kernel-mixture slice (KDE), and 2D histogram of kernel deposit centers. "
            "Requires --fes (3D .dat), --kernels, --colvar, and --out (same as --triptych-3d)."
        ),
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        default=None,
        help=(
            "OPES KERNELS path (--triptych / --triptych-opes-style / --triptych-3d / "
            "--triptych-3d-opes-style / --dashboard-3d)."
        ),
    )
    parser.add_argument(
        "--colvar",
        type=Path,
        default=None,
        help=(
            "COLVAR path (--triptych / --triptych-opes-style / --triptych-3d / "
            "--triptych-3d-opes-style / --dashboard-3d)."
        ),
    )
    parser.add_argument(
        "--triptych-cv",
        type=str,
        default=None,
        help=(
            "Comma-separated two CV names for COLVAR (--triptych); "
            "default: names from the FES .dat FIELDS line."
        ),
    )
    parser.add_argument(
        "--triptych-skiprows",
        type=int,
        default=0,
        help="COLVAR data rows to skip after header (--triptych).",
    )
    parser.add_argument(
        "--triptych-grid-bins",
        type=int,
        default=100,
        help="Square grid resolution for kernel-density panels (--triptych / --triptych-3d).",
    )
    parser.add_argument(
        "--triptych-hist-bins",
        type=int,
        default=90,
        help="Histogram bins per axis (--triptych / --triptych-3d).",
    )
    parser.add_argument(
        "--triptych-opes-hexbin-gridsize",
        type=int,
        default=14,
        help=(
            "matplotlib hexbin gridsize for --triptych-opes-style (right panel). "
            "Lower = larger hex cells and higher counts per cell (default 14; clamped 8–90). "
            "Does not apply to --triptych / --triptych-3d (those use --triptych-hist-bins)."
        ),
    )
    parser.add_argument(
        "--triptych-opes-hexbin-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2D OPES-style triptych: log color scale for hex counts (default: on).",
    )
    parser.add_argument(
        "--triptych-opes-colvar-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2D OPES-style triptych: faint scatter of raw COLVAR samples under hexbin.",
    )
    parser.add_argument(
        "--triptych-cv-3d",
        type=str,
        default=None,
        help=(
            "Comma-separated three CV names for COLVAR (--triptych-3d / --triptych-3d-opes-style / "
            "--dashboard-3d); default: names from the 3D FES .dat FIELDS line."
        ),
    )
    parser.add_argument(
        "--triptych-3d-kernel-center-alpha",
        type=float,
        default=0.22,
        help=(
            "Opacity of orange kernel-center markers on the normalized-mixture panels "
            "(--triptych-3d-opes-style). Lower = more transparent (default 0.22)."
        ),
    )
    parser.add_argument(
        "--triptych-3d-opes-hexbin-gridsize",
        type=int,
        default=36,
        help=(
            "matplotlib hexbin gridsize for --triptych-3d-opes-style (right column). "
            "Lower = larger hex cells (default 36). Ignores --triptych-hist-bins for that mode."
        ),
    )
    parser.add_argument(
        "--triptych-3d-opes-hexbin-linewidths",
        type=float,
        default=0.55,
        help=(
            "Hex cell edge linewidth for --triptych-3d-opes-style hexbin panels; "
            "0 disables drawn edges (default 0.55)."
        ),
    )
    args = parser.parse_args()

    from genai_tps.simulation.plumed_colvar_fes import run_fes_from_reweighting_script
    from genai_tps.simulation.plumed_fes_plot import (
        plot_fes_dat_to_png,
        plot_opes_2d_fes_triptych,
        plot_opes_2d_fes_triptych_opes_style,
        plot_opes_3d_fes_slices_kde_deposits,
        plot_opes_3d_fes_triptych,
        plot_opes_3d_fes_triptych_opes_style,
    )

    dash_flags = (
        int(bool(args.triptych))
        + int(bool(args.triptych_opes_style))
        + int(bool(args.triptych_3d))
        + int(bool(args.triptych_3d_opes_style))
        + int(bool(args.dashboard_3d))
    )
    if dash_flags > 1:
        print(
            "Use only one of --triptych, --triptych-opes-style, --triptych-3d, "
            "--triptych-3d-opes-style, or --dashboard-3d.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dashboard_3d:
        if args.campaign is not None:
            print("--dashboard-3d cannot be combined with --campaign.", file=sys.stderr)
            sys.exit(2)
        if args.fes is None or args.kernels is None or args.colvar is None or args.out is None:
            print(
                "--dashboard-3d requires --fes, --kernels, --colvar, and --out.",
                file=sys.stderr,
            )
            sys.exit(2)
        fes_path = args.fes.expanduser().resolve()
        kernels_path = args.kernels.expanduser().resolve()
        colvar_path = args.colvar.expanduser().resolve()
        out_png = args.out.expanduser().resolve()
        for label, p in (
            ("--fes", fes_path),
            ("--kernels", kernels_path),
            ("--colvar", colvar_path),
        ):
            if not p.is_file():
                print(f"Not found ({label}): {p}", file=sys.stderr)
                sys.exit(1)
        cv_trip: tuple[str, str, str] | None = None
        if args.triptych_cv_3d is not None:
            parts = [c.strip() for c in args.triptych_cv_3d.split(",")]
            if len(parts) != 3:
                print("--triptych-cv-3d must be exactly three comma-separated names.", file=sys.stderr)
                sys.exit(2)
            cv_trip = (parts[0], parts[1], parts[2])
        plot_opes_3d_fes_slices_kde_deposits(
            fes_path,
            kernels_path,
            colvar_path,
            out_png,
            colvar_cv_names=cv_trip,
            dpi=150,
            skiprows=int(args.triptych_skiprows),
            vmax_percentile=float(args.vmax_percentile),
        )
        print(f"Wrote 3D slice dashboard {out_png}", flush=True)
        return

    if args.triptych_opes_style:
        if args.campaign is not None:
            print("--triptych-opes-style cannot be combined with --campaign.", file=sys.stderr)
            sys.exit(2)
        if args.fes is None or args.kernels is None or args.colvar is None or args.out is None:
            print(
                "--triptych-opes-style requires --fes, --kernels, --colvar, and --out.",
                file=sys.stderr,
            )
            sys.exit(2)
        fes_path = args.fes.expanduser().resolve()
        kernels_path = args.kernels.expanduser().resolve()
        colvar_path = args.colvar.expanduser().resolve()
        out_png = args.out.expanduser().resolve()
        for label, p in (
            ("--fes", fes_path),
            ("--kernels", kernels_path),
            ("--colvar", colvar_path),
        ):
            if not p.is_file():
                print(f"Not found ({label}): {p}", file=sys.stderr)
                sys.exit(1)
        cv_pair = None
        if args.triptych_cv is not None:
            parts = [c.strip() for c in args.triptych_cv.split(",")]
            if len(parts) != 2:
                print("--triptych-cv must be exactly two comma-separated names.", file=sys.stderr)
                sys.exit(2)
            cv_pair = (parts[0], parts[1])
        hex_gs = int(np.clip(int(args.triptych_opes_hexbin_gridsize), 8, 90))
        plot_opes_2d_fes_triptych_opes_style(
            fes_path,
            kernels_path,
            colvar_path,
            out_png,
            colvar_cv_names=cv_pair,
            dpi=150,
            grid_bins=int(args.triptych_grid_bins),
            hexbin_gridsize=hex_gs,
            hexbin_bins_log=bool(args.triptych_opes_hexbin_log),
            colvar_scatter_overlay=bool(args.triptych_opes_colvar_scatter),
            skiprows=int(args.triptych_skiprows),
            temperature_k=float(args.temperature),
        )
        print(f"Wrote OPES-style 2D triptych {out_png}", flush=True)
        return

    if args.triptych_3d:
        if args.campaign is not None:
            print("--triptych-3d cannot be combined with --campaign.", file=sys.stderr)
            sys.exit(2)
        if args.fes is None or args.kernels is None or args.colvar is None or args.out is None:
            print(
                "--triptych-3d requires --fes, --kernels, --colvar, and --out.",
                file=sys.stderr,
            )
            sys.exit(2)
        fes_path = args.fes.expanduser().resolve()
        kernels_path = args.kernels.expanduser().resolve()
        colvar_path = args.colvar.expanduser().resolve()
        out_png = args.out.expanduser().resolve()
        for label, p in (
            ("--fes", fes_path),
            ("--kernels", kernels_path),
            ("--colvar", colvar_path),
        ):
            if not p.is_file():
                print(f"Not found ({label}): {p}", file=sys.stderr)
                sys.exit(1)
        cv_trip: tuple[str, str, str] | None = None
        if args.triptych_cv_3d is not None:
            parts = [c.strip() for c in args.triptych_cv_3d.split(",")]
            if len(parts) != 3:
                print("--triptych-cv-3d must be exactly three comma-separated names.", file=sys.stderr)
                sys.exit(2)
            cv_trip = (parts[0], parts[1], parts[2])
        plot_opes_3d_fes_triptych(
            fes_path,
            kernels_path,
            colvar_path,
            out_png,
            colvar_cv_names=cv_trip,
            dpi=150,
            grid_bins=int(args.triptych_grid_bins),
            hist_bins=int(args.triptych_hist_bins),
            skiprows=int(args.triptych_skiprows),
            vmax_percentile=float(args.vmax_percentile),
        )
        print(f"Wrote 3D triptych {out_png}", flush=True)
        return

    if args.triptych_3d_opes_style:
        if args.campaign is not None:
            print("--triptych-3d-opes-style cannot be combined with --campaign.", file=sys.stderr)
            sys.exit(2)
        if args.fes is None or args.kernels is None or args.colvar is None or args.out is None:
            print(
                "--triptych-3d-opes-style requires --fes, --kernels, --colvar, and --out.",
                file=sys.stderr,
            )
            sys.exit(2)
        fes_path = args.fes.expanduser().resolve()
        kernels_path = args.kernels.expanduser().resolve()
        colvar_path = args.colvar.expanduser().resolve()
        out_png = args.out.expanduser().resolve()
        for label, p in (
            ("--fes", fes_path),
            ("--kernels", kernels_path),
            ("--colvar", colvar_path),
        ):
            if not p.is_file():
                print(f"Not found ({label}): {p}", file=sys.stderr)
                sys.exit(1)
        cv_trip3: tuple[str, str, str] | None = None
        if args.triptych_cv_3d is not None:
            parts = [c.strip() for c in args.triptych_cv_3d.split(",")]
            if len(parts) != 3:
                print("--triptych-cv-3d must be exactly three comma-separated names.", file=sys.stderr)
                sys.exit(2)
            cv_trip3 = (parts[0], parts[1], parts[2])
        hex_gs = int(np.clip(int(args.triptych_3d_opes_hexbin_gridsize), 10, 80))
        hex_lw = float(args.triptych_3d_opes_hexbin_linewidths)
        plot_opes_3d_fes_triptych_opes_style(
            fes_path,
            kernels_path,
            colvar_path,
            out_png,
            colvar_cv_names=cv_trip3,
            dpi=150,
            hexbin_gridsize=hex_gs,
            hexbin_linewidths=None if hex_lw <= 0 else hex_lw,
            skiprows=int(args.triptych_skiprows),
            temperature_k=float(args.temperature),
            kernel_center_alpha=float(args.triptych_3d_kernel_center_alpha),
        )
        print(f"Wrote 3D OPES-style triptych {out_png}", flush=True)
        return

    if args.triptych:
        if args.campaign is not None:
            print("--triptych cannot be combined with --campaign.", file=sys.stderr)
            sys.exit(2)
        if args.fes is None or args.kernels is None or args.colvar is None or args.out is None:
            print(
                "--triptych requires --fes, --kernels, --colvar, and --out.",
                file=sys.stderr,
            )
            sys.exit(2)
        fes_path = args.fes.expanduser().resolve()
        kernels_path = args.kernels.expanduser().resolve()
        colvar_path = args.colvar.expanduser().resolve()
        out_png = args.out.expanduser().resolve()
        for label, p in (
            ("--fes", fes_path),
            ("--kernels", kernels_path),
            ("--colvar", colvar_path),
        ):
            if not p.is_file():
                print(f"Not found ({label}): {p}", file=sys.stderr)
                sys.exit(1)
        cv_pair = None
        if args.triptych_cv is not None:
            parts = [c.strip() for c in args.triptych_cv.split(",")]
            if len(parts) != 2:
                print("--triptych-cv must be exactly two comma-separated names.", file=sys.stderr)
                sys.exit(2)
            cv_pair = (parts[0], parts[1])
        plot_opes_2d_fes_triptych(
            fes_path,
            kernels_path,
            colvar_path,
            out_png,
            colvar_cv_names=cv_pair,
            dpi=150,
            grid_bins=int(args.triptych_grid_bins),
            hist_bins=int(args.triptych_hist_bins),
            skiprows=int(args.triptych_skiprows),
            vmax_percentile=float(args.vmax_percentile),
        )
        print(f"Wrote triptych {out_png}", flush=True)
        return

    if args.campaign is not None:
        out_root = _resolve_campaign_out_root(args.campaign)
        print(f"[plot-fes] Campaign root: {out_root}", flush=True)
        selected = {int(c.strip()) for c in args.cases.split(",")}
        any_ok = False
        for idx, name in enumerate(_CASE_NAMES, start=1):
            if idx not in selected:
                continue
            colvar = out_root / name / "openmm_opes_md" / "opes_states" / "COLVAR"
            if not colvar.is_file():
                print(f"  [plot-fes] Skip case {idx}: no COLVAR at {colvar}", flush=True)
                continue
            fes_dat = colvar.parent / args.outfile_name
            out_png = fes_dat.with_suffix(".png")
            print(f"  [plot-fes] Case {idx}: COLVAR {colvar}", flush=True)
            try:
                run_fes_from_reweighting_script(
                    colvar_path=colvar,
                    outfile=fes_dat,
                    temperature_k=args.temperature,
                    sigma=args.sigma,
                    cv_names=args.cv,
                    bias_name=args.bias,
                    grid_min=args.grid_min,
                    grid_max=args.grid_max,
                    grid_bin=args.grid_bin,
                    skiprows=args.skiprows,
                    blocks=args.blocks,
                )
            except FileNotFoundError as exc:
                print(f"  [plot-fes] ERROR: {exc}", file=sys.stderr, flush=True)
                sys.exit(1)
            except ValueError as exc:
                print(f"  [plot-fes] ERROR: {exc}", file=sys.stderr, flush=True)
                sys.exit(1)
            except subprocess.CalledProcessError as exc:
                print(f"  [plot-fes] FES script failed: {exc}", file=sys.stderr, flush=True)
                sys.exit(exc.returncode if exc.returncode else 1)
            plot_fes_dat_to_png(
                fes_dat,
                out_png,
                vmax_percentile=float(args.vmax_percentile),
                **_marginal_plot_kwargs(args, colvar_for_mask=colvar, fes_path=fes_dat),
            )
            print(f"  [plot-fes] Wrote {fes_dat} and {out_png}", flush=True)
            any_ok = True
        if not any_ok:
            print(
                "[plot-fes] No case produced a plot (missing COLVAR or no matching --cases).",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        return

    fes_path = args.fes.expanduser().resolve()
    if not fes_path.is_file():
        print(f"Not found: {fes_path}", file=sys.stderr)
        sys.exit(1)
    out_png = (
        args.out.expanduser().resolve()
        if args.out is not None
        else fes_path.with_suffix(".png")
    )
    mkw = _marginal_plot_kwargs(args, fes_path=fes_path)
    if args.mask_marginals_with_colvar and mkw["marginal_colvar_mask_path"] is None:
        print(
            "[plot-fes] Warning: --mask-marginals-with-colvar but no COLVAR found "
            f"beside {fes_path}; marginal masking skipped.",
            file=sys.stderr,
            flush=True,
        )
    plot_fes_dat_to_png(
        fes_path,
        out_png,
        vmax_percentile=float(args.vmax_percentile),
        **mkw,
    )
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()

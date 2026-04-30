#!/usr/bin/env python3
"""Stage 1b — Reweighted FES from PLUMED ``COLVAR`` plus OPES-style triptych PNGs.

Runs the PLUMED OPES tutorial ``FES_from_Reweighting.py`` (GPL, in
``plumed2/user-doc/tutorials/others/opes-metad/``) on each case's
``openmm_opes_md/opes_states/COLVAR`` produced by Stage 0 PLUMED OPES-MD.

CV names, ``sigma``, ``--grid-bin``, and the output ``fes_reweighted_*.dat`` path
are **inferred from COLVAR** ``#! FIELDS`` (OneOPES ``pp.proj,cmap``, 3D
``lig_rmsd,lig_dist,lig_contacts``, or classic 2D) unless you pass explicit
overrides.

After each successful FES, writes an OPES-style triptych PNG next to
``KERNELS`` (requires ``KERNELS`` to exist; otherwise logs a warning).

Typical use after Stage 0 (and optionally after Stage 1)::

    python scripts/campaign/01b_plumed_colvar_fes.py \\
        --out outputs/campaign_stage00_full_20260430 \\
        --cases 1,2,3

Pipeline role:
    00_generate_reference_data  →  01_assemble_datasets (optional ``--plumed-fes``)
    or run this script standalone on an existing campaign tree.
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
    """Same resolution rules as ``01_assemble_datasets.py``."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1b: PLUMED COLVAR → reweighted FES (OPES tutorial script) "
            "+ OPES-style triptych PNG."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/campaign"),
        help="Root campaign output directory (same as Stage 0 --out).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="1,2,3",
        help="Comma-separated case numbers to process.",
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Kelvin (for kBT).")
    parser.add_argument(
        "--sigma",
        type=str,
        default="0.3,0.5,1.0",
        help=(
            "KDE bandwidth(s), comma-separated. "
            "Two values are used for 2D COLVARs; three for 3D (lig_contacts); "
            "inference adjusts if the count does not match dimensionality."
        ),
    )
    parser.add_argument(
        "--cv",
        type=str,
        default=None,
        help="Override inferred comma-separated CV names (default: from COLVAR FIELDS).",
    )
    parser.add_argument(
        "--bias",
        type=str,
        default="opes.bias",
        help="Bias column name(s) for reweighting.",
    )
    parser.add_argument(
        "--grid-min",
        type=str,
        default=None,
        help="Optional --min for FES script (e.g. '0,0' for 2D); omit for auto from data.",
    )
    parser.add_argument(
        "--grid-max",
        type=str,
        default=None,
        help="Optional --max (e.g. '15,20'); omit for auto from data.",
    )
    parser.add_argument(
        "--grid-bin",
        type=str,
        default=None,
        help="Override inferred bin counts (e.g. '100,100' or '40,40,40' for 3D).",
    )
    parser.add_argument("--skiprows", type=int, default=0, help="Burn-in rows to skip.")
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Block averaging for uncertainty (1 = single FES).",
    )
    parser.add_argument(
        "--outfile-name",
        type=str,
        default=None,
        help=(
            "Basename for FES output under opes_states/ (default: "
            "fes_reweighted_2d.dat or fes_reweighted_3d.dat from COLVAR kind)."
        ),
    )
    parser.add_argument(
        "--skip-triptych",
        action="store_true",
        help="Only run reweighting; do not write OPES-style triptych PNGs.",
    )
    parser.add_argument(
        "--triptych-opes-hexbin-gridsize",
        type=int,
        default=14,
        help="2D triptych right panel: matplotlib hexbin gridsize (clamped 8–90).",
    )
    parser.add_argument(
        "--triptych-opes-hexbin-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2D triptych: log color scale for hex counts (default: on).",
    )
    parser.add_argument(
        "--triptych-opes-colvar-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2D triptych: faint scatter of COLVAR samples under hexbin.",
    )
    args = parser.parse_args()

    from genai_tps.simulation.plumed_colvar_fes import (
        read_colvar_field_names,
        reweighting_kwargs_from_colvar_path,
        run_fes_from_reweighting_script,
    )
    from genai_tps.simulation.plumed_fes_plot import (
        plot_opes_2d_fes_triptych_opes_style,
        plot_opes_3d_fes_triptych_opes_style,
    )

    out_root = _resolve_campaign_out_root(args.out)
    print(f"[01b] Campaign output root: {out_root}", flush=True)
    selected = {int(c.strip()) for c in args.cases.split(",")}

    hex_gs = int(np.clip(int(args.triptych_opes_hexbin_gridsize), 8, 90))

    for idx, name in enumerate(_CASE_NAMES, start=1):
        if idx not in selected:
            print(f"\n[01b] Skipping {name} (not in --cases {args.cases})")
            continue
        print(f"\n{'='*60}\n  Case {idx}: {name}\n{'='*60}", flush=True)
        colvar = out_root / name / "openmm_opes_md" / "opes_states" / "COLVAR"
        if not colvar.is_file():
            print(
                f"  [01b] No COLVAR at {colvar}; skip (observer mode or incomplete Stage 0).",
                flush=True,
            )
            continue
        print(f"  [01b] COLVAR: {colvar}", flush=True)

        rw = reweighting_kwargs_from_colvar_path(colvar, sigma_arg=args.sigma)
        cv_names = str(rw["cv_names"]) if args.cv is None else str(args.cv)
        grid_bin = str(rw["grid_bin"]) if args.grid_bin is None else str(args.grid_bin)
        outfile = (
            colvar.parent / args.outfile_name
            if args.outfile_name is not None
            else Path(rw["outfile"])
        )
        sigma = str(rw["sigma"])

        print(
            f"  [01b] FES kwargs: cv_names={cv_names!r} sigma={sigma!r} "
            f"grid_bin={grid_bin!r} outfile={outfile.name}",
            flush=True,
        )
        try:
            run_fes_from_reweighting_script(
                colvar_path=colvar,
                outfile=outfile,
                temperature_k=args.temperature,
                sigma=sigma,
                cv_names=cv_names,
                bias_name=args.bias,
                grid_min=args.grid_min,
                grid_max=args.grid_max,
                grid_bin=grid_bin,
                skiprows=args.skiprows,
                blocks=args.blocks,
            )
        except FileNotFoundError as exc:
            print(f"  [01b] ERROR: {exc}", flush=True)
            sys.exit(1)
        except ValueError as exc:
            print(f"  [01b] ERROR: {exc}", flush=True)
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            print(f"  [01b] FES script failed: {exc}", flush=True)
            sys.exit(exc.returncode if exc.returncode else 1)
        print(f"  [01b] Wrote {outfile}", flush=True)

        if args.skip_triptych:
            continue

        kernels = colvar.parent / "KERNELS"
        if not kernels.is_file():
            print(
                f"  [01b] No KERNELS at {kernels}; skip triptych (Stage 0 may not have deposited).",
                flush=True,
            )
            continue

        fields = read_colvar_field_names(colvar)
        is_3d = "lig_contacts" in fields
        is_oneopes = "pp.proj" in fields and "cmap" in fields

        if is_3d:
            trip_out = colvar.parent / "fes_3d_opes_style_triptych.png"
            cvs = tuple(c.strip() for c in cv_names.split(",") if c.strip())
            if len(cvs) != 3:
                print(
                    f"  [01b] Skip 3D triptych: expected three CV names, got {cvs!r}.",
                    flush=True,
                )
                continue
            plot_opes_3d_fes_triptych_opes_style(
                outfile,
                kernels,
                colvar,
                trip_out,
                colvar_cv_names=cvs,
                skiprows=int(args.skiprows),
            )
            print(f"  [01b] Wrote triptych {trip_out}", flush=True)
            continue

        tokens = [c.strip() for c in cv_names.split(",") if c.strip()]
        if len(tokens) != 2:
            print(
                f"  [01b] Skip 2D triptych: expected two CV names, got {tokens!r}.",
                flush=True,
            )
            continue
        cv_pair = (tokens[0], tokens[1])
        if is_oneopes:
            trip_out = colvar.parent / "fes_oneopes_2d_opes_style_triptych.png"
        else:
            trip_out = colvar.parent / "fes_2d_opes_style_triptych.png"
        plot_opes_2d_fes_triptych_opes_style(
            outfile,
            kernels,
            colvar,
            trip_out,
            colvar_cv_names=cv_pair,
            dpi=150,
            grid_bins=96,
            hexbin_gridsize=hex_gs,
            hexbin_bins_log=bool(args.triptych_opes_hexbin_log),
            colvar_scatter_overlay=bool(args.triptych_opes_colvar_scatter),
            skiprows=int(args.skiprows),
            temperature_k=float(args.temperature),
        )
        print(f"  [01b] Wrote triptych {trip_out}", flush=True)

    print("\n[01b] Stage 1b complete.", flush=True)


if __name__ == "__main__":
    main()

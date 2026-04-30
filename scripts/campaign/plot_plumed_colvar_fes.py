#!/usr/bin/env python3
"""Plot ``fes_reweighted_*.dat`` from PLUMED ``FES_from_Reweighting.py`` to PNG.

Use real Stage-0 output only: either pass ``--fes`` to an existing ``.dat``, or
``--campaign`` (same root as ``00_generate_reference_data.py --out``) to run
the reweighting step on each case's ``openmm_opes_md/opes_states/COLVAR`` and
write ``fes_reweighted_2d.png`` beside the ``.dat`` (same defaults as
``01b_plumed_colvar_fes.py``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    args = parser.parse_args()

    from genai_tps.simulation.plumed_colvar_fes import run_fes_from_reweighting_script
    from genai_tps.simulation.plumed_fes_plot import plot_fes_dat_to_png

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
                fes_dat, out_png, vmax_percentile=float(args.vmax_percentile)
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
    plot_fes_dat_to_png(
        fes_path, out_png, vmax_percentile=float(args.vmax_percentile)
    )
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()

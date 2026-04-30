#!/usr/bin/env python3
"""Stage 1b — Reweighted FES from PLUMED ``COLVAR`` (OPES tutorial script).

Runs the PLUMED OPES tutorial ``FES_from_Reweighting.py`` (GPL, in
``plumed2/user-doc/tutorials/others/opes-metad/``) on each case's
``openmm_opes_md/opes_states/COLVAR`` produced by Stage 0 PLUMED OPES-MD.

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
        description="Stage 1b: PLUMED COLVAR → reweighted FES (OPES tutorial script).",
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
        default="0.3,0.5",
        help="KDE bandwidth(s), comma-separated (match OPES SIGMA in Å).",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default="lig_rmsd,lig_dist",
        help="Comma-separated CV names matching COLVAR FIELDS.",
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
        default="100,100",
        help="Comma-separated bin counts for FES grid.",
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
        default="fes_reweighted_2d.dat",
        help="Output filename written under each case's opes_states/.",
    )
    args = parser.parse_args()

    from genai_tps.simulation.plumed_colvar_fes import run_fes_from_reweighting_script

    out_root = _resolve_campaign_out_root(args.out)
    print(f"[01b] Campaign output root: {out_root}", flush=True)
    selected = {int(c.strip()) for c in args.cases.split(",")}

    for idx, name in enumerate(_CASE_NAMES, start=1):
        if idx not in selected:
            print(f"\n[01b] Skipping {name} (not in --cases {args.cases})")
            continue
        print(f"\n{'='*60}\n  Case {idx}: {name}\n{'='*60}", flush=True)
        colvar = out_root / name / "openmm_opes_md" / "opes_states" / "COLVAR"
        if not colvar.is_file():
            print(f"  [01b] No COLVAR at {colvar}; skip (observer mode or incomplete Stage 0).", flush=True)
            continue
        outfile = colvar.parent / args.outfile_name
        print(f"  [01b] COLVAR: {colvar}", flush=True)
        try:
            run_fes_from_reweighting_script(
                colvar_path=colvar,
                outfile=outfile,
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
            print(f"  [01b] ERROR: {exc}", flush=True)
            sys.exit(1)
        except ValueError as exc:
            print(f"  [01b] ERROR: {exc}", flush=True)
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            print(f"  [01b] FES script failed: {exc}", flush=True)
            sys.exit(exc.returncode if exc.returncode else 1)
        print(f"  [01b] Wrote {outfile}", flush=True)

    print("\n[01b] Stage 1b complete.", flush=True)


if __name__ == "__main__":
    main()

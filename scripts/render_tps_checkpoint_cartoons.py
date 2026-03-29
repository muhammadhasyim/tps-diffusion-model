#!/usr/bin/env python3
"""Export last diffusion frames from TPS ``tps_mc_step_*.npz`` checkpoints and render PyMOL cartoons.

Writes, for each checkpoint:

* ``<out-dir>/pdb/<stem>_last.pdb`` — terminal frame via :func:`batch_export`
* ``<out-dir>/png/<stem>_last.png`` — cartoon (DSSP colors) via the same PyMOL path as
  ``scripts/visualize_cofolding_trajectory.py`` (subprocess ``pymol -cqx`` by default).

Example (OPES-TPS layout)::

    python scripts/render_tps_checkpoint_cartoons.py \\
        --ckpt-dir opes_tps_out_case2/trajectory_checkpoints \\
        --topo-npz opes_tps_out_case2/boltz_results_case2_cdk2_atp_wildtype/processed/structures/case2_cdk2_atp_wildtype.npz \\
        --out-dir opes_tps_out_case2/pymol_cartoons

If there is exactly one ``boltz_results_*/processed/structures/*.npz`` under the parent
of ``--ckpt-dir``, ``--topo-npz`` may be omitted.

**Dependencies:** Boltz (submodule on path), ``pymol`` on ``PATH`` (``pip install pymol-open-source``
or conda-forge). For headless servers use ``--no-ray`` and/or ``LIBGL_ALWAYS_SOFTWARE=1``;
the script delegates to ``render_cartoon_png_pymol(..., force_subprocess=True)``.

Ligands are **HETATM** in the PDB; PyMOL **cartoon** does not show them. By default this script
adds **sticks** for ``hetatm and not resn HOH`` with CPK colors. Pass ``--no-ligand-sticks`` for
protein-only ribbons like the old behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "python"
    scripts = root / "scripts"
    for p in (src, scripts):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_visualize_module():
    path = Path(__file__).resolve().parent / "visualize_cofolding_trajectory.py"
    spec = importlib.util.spec_from_file_location("_viz_cofold_traj", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_topo_npz(ckpt_dir: Path, explicit: Path | None) -> Path:
    """Resolve Boltz ``processed/structures/*.npz`` for checkpoint export."""
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--topo-npz not found: {p}")
        return p
    parent = ckpt_dir.resolve().parent
    cands = sorted(parent.glob("boltz_results_*/processed/structures/*.npz"))
    if len(cands) == 0:
        raise FileNotFoundError(
            f"No topology: pass --topo-npz or place a single "
            f"boltz_results_*/processed/structures/*.npz under {parent}"
        )
    if len(cands) > 1:
        raise ValueError(
            "Multiple topology NPZ files found; pass --topo-npz explicitly:\n  "
            + "\n  ".join(str(c) for c in cands)
        )
    return cands[0]


def main() -> None:
    _ensure_paths()
    from genai_tps.analysis.boltz_npz_export import batch_export  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Directory containing tps_mc_step_*.npz (e.g. .../trajectory_checkpoints).",
    )
    ap.add_argument(
        "--topo-npz",
        type=Path,
        default=None,
        help="Boltz processed/structures/*.npz (optional if uniquely discoverable).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root: pdb/ and png/ subdirectories are created.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="tps_mc_step_*.npz",
        help="Checkpoint glob under --ckpt-dir (default: tps_mc_step_*.npz).",
    )
    ap.add_argument(
        "--pdb-only",
        action="store_true",
        help="Only write PDBs; skip PyMOL rendering.",
    )
    ap.add_argument(
        "--png-only",
        action="store_true",
        help="Only render PNGs; assume PDBs already exist in out-dir/pdb.",
    )
    ap.add_argument(
        "--width", type=int, default=1600, help="PNG width (default: 1600).",
    )
    ap.add_argument(
        "--height", type=int, default=1600, help="PNG height (default: 1600).",
    )
    ap.add_argument("--dpi", type=int, default=150, help="PNG dpi (default: 150).")
    ap.add_argument(
        "--no-ray",
        action="store_true",
        help="OpenGL framebuffer capture instead of ray tracing (more stable headless).",
    )
    ap.add_argument(
        "--no-software-gl",
        action="store_true",
        help="Do not force LIBGL_ALWAYS_SOFTWARE=1 for the PyMOL subprocess.",
    )
    ap.add_argument(
        "--no-ligand-sticks",
        action="store_true",
        help="Hide HETATM in the image (default: draw ligands/cofactors as sticks on the cartoon).",
    )
    ap.add_argument(
        "--pocket-zoom",
        action="store_true",
        help=(
            "Render with translucent protein surface + CPK ligand sticks + pocket-zoomed camera "
            "(Skrinjar-style). Replaces the default cartoon render when set."
        ),
    )
    ap.add_argument(
        "--reference-pdb",
        type=Path,
        default=None,
        help=(
            "Reference PDB for Cα alignment before pocket-zoom rendering (consistent orientation). "
            "If omitted the first checkpoint PDB is used as reference."
        ),
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most this many checkpoints (0 = no limit).",
    )
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    if not ckpt_dir.is_dir():
        print(f"Not a directory: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    topo_npz = discover_topo_npz(ckpt_dir, args.topo_npz)
    out_root = args.out_dir.expanduser().resolve()
    pdb_dir = out_root / "pdb"
    png_dir = out_root / "png"

    viz = _load_visualize_module()
    render = viz.render_cartoon_png_pymol
    render_pocket_zoom = viz.render_pocket_zoom_png_pymol

    pdb_paths: list[Path] = []
    if not args.png_only:
        pdb_dir.mkdir(parents=True, exist_ok=True)
        max_f = args.max_files if args.max_files > 0 else None
        pdb_paths = batch_export(
            ckpt_dir,
            topo_npz,
            pdb_dir,
            frame_idx=-1,
            glob=args.glob,
            max_files=max_f,
        )
        print(f"Wrote {len(pdb_paths)} PDB(s) under {pdb_dir}", file=sys.stderr)
    else:
        for p in sorted(pdb_dir.glob("tps_mc_step_*_last.pdb")):
            if any(x in p.stem for x in ("last_frame", "latest")):
                continue
            pdb_paths.append(p)
        if args.max_files > 0:
            pdb_paths = pdb_paths[: args.max_files]
        if not pdb_paths:
            print(f"No *_last.pdb under {pdb_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Rendering {len(pdb_paths)} existing PDB(s)", file=sys.stderr)

    if args.pdb_only:
        return

    png_dir.mkdir(parents=True, exist_ok=True)
    use_ray = not args.no_ray
    prefer_sw = not args.no_software_gl
    n_ok = 0

    # Resolve reference PDB for pocket-zoom alignment
    reference_pdb: Path | None = None
    if args.pocket_zoom:
        if args.reference_pdb is not None:
            reference_pdb = args.reference_pdb.expanduser().resolve()
        elif pdb_paths:
            reference_pdb = pdb_paths[0]
            print(f"Pocket-zoom: using {reference_pdb.name} as alignment reference", file=sys.stderr)

    for pdb_path in pdb_paths:
        png_path = png_dir / (pdb_path.stem + ".png")
        try:
            if args.pocket_zoom:
                render_pocket_zoom(
                    pdb_path,
                    png_path,
                    reference_pdb=reference_pdb,
                    width=args.width,
                    height=args.height,
                    dpi=args.dpi,
                    use_ray=use_ray,
                    prefer_software_gl=prefer_sw,
                )
            else:
                render(
                    pdb_path,
                    png_path,
                    width=args.width,
                    height=args.height,
                    dpi=args.dpi,
                    force_subprocess=True,
                    use_ray=use_ray,
                    prefer_software_gl=prefer_sw,
                    show_ligand_sticks=not args.no_ligand_sticks,
                )
            n_ok += 1
            print(png_path, file=sys.stderr)
        except Exception as exc:
            print(f"ERROR {pdb_path.name}: {exc}", file=sys.stderr)
    print(f"Rendered {n_ok}/{len(pdb_paths)} PNG(s) under {png_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

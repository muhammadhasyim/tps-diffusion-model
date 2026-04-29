#!/usr/bin/env python3
"""Analyse the terminal-structure ensemble from TPS checkpoints using ProDy.

Accepts either:

- a checkpoint directory (``--ckpt-dir``) + Boltz topology (``--topo-npz``),
  converting NPZ files to PDB on the fly; or
- a pre-built PDB directory (``--pdb-dir``).

Outputs written to ``--out-dir``:

- ``prody_terminal_summary.json`` — eigenvalues, explained variance, per-
  structure PC projections, and optional ANM/GNM eigenvalues.
- ``pc_scree.png`` — scree plot of cumulative explained variance.
- ``pc1_pc2.png`` — scatter of PC1 vs PC2, coloured by MC step (if
  structure labels carry that information).

Usage::

    # From raw NPZ checkpoints (batch-exports PDBs first)
    python scripts/analyze_terminal_ensemble_prody.py \\
        --ckpt-dir  cofolding_tps_out/trajectory_checkpoints \\
        --topo-npz  cofolding_tps_out/boltz_results_cofolding_multimer_msa_empty/processed/structures/cofolding_multimer_msa_empty.npz \\
        --out-dir   cofolding_tps_out/prody_analysis \\
        --n-pcs     10 --anm --gnm

    # From pre-exported PDB files
    python scripts/analyze_terminal_ensemble_prody.py \\
        --pdb-dir   cofolding_tps_out/cv_rmsd_analysis/last_frame_pdbs \\
        --out-dir   cofolding_tps_out/prody_analysis \\
        --n-pcs     10 --anm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path setup — make src/python and boltz/src importable
# ---------------------------------------------------------------------------

def _setup_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for sub in ("src/python", "boltz/src"):
        p = root / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_setup_paths()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_scree(
    cumvar: list[float],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(cumvar)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, n + 1), cumvar, marker="o", color="#1565C0", linewidth=2,
            markersize=6)
    ax.axhline(0.9, color="#E53935", linestyle="--", linewidth=1,
               label="90% variance")
    ax.set_xlabel("Principal component", fontsize=11)
    ax.set_ylabel("Cumulative explained variance", fontsize=11)
    ax.set_title("PCA scree — terminal structure ensemble", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, n + 1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Scree plot → %s", out_path)


def _extract_mc_step(label: str) -> int | None:
    """Try to parse a MC-step integer from a label like ``tps_mc_step_00001000``."""
    m = re.search(r"tps_mc_step_(\d+)", label)
    return int(m.group(1)) if m else None


def _plot_pc1_pc2(
    projections: list[list[float]],
    labels: list[str],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    proj = np.asarray(projections)
    if proj.shape[1] < 2:
        logger.warning("Fewer than 2 PCs — skipping PC1 vs PC2 plot.")
        return

    mc_steps = [_extract_mc_step(lbl) for lbl in labels]
    has_steps = all(s is not None for s in mc_steps)

    fig, ax = plt.subplots(figsize=(6, 5))

    if has_steps:
        steps = [float(s) for s in mc_steps]
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=steps, cmap="viridis",
                        s=60, zorder=3, edgecolors="white", linewidths=0.4)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("MC step", fontsize=9)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], color="#1565C0", s=60,
                   zorder=3, edgecolors="white", linewidths=0.4)

    for i, lbl in enumerate(labels):
        step = _extract_mc_step(lbl)
        tag = str(step) if step is not None else lbl[-8:]
        ax.annotate(tag, (proj[i, 0], proj[i, 1]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", alpha=0.7)

    ax.set_xlabel("PC1 (Å)", fontsize=11)
    ax.set_ylabel("PC2 (Å)", fontsize=11)
    ax.set_title("Terminal ensemble — PC1 vs PC2", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("PC1/PC2 scatter → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-dir", type=Path,
                     help="Directory of single-frame PDB files.")
    src.add_argument("--ckpt-dir", type=Path,
                     help="Checkpoint directory with tps_mc_step_*.npz files.")
    ap.add_argument("--topo-npz", type=Path,
                    help="Boltz topology NPZ (required with --ckpt-dir).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-pcs", type=int, default=10,
                    help="Number of principal components (default: 10).")
    ap.add_argument("--anm", action="store_true",
                    help="Compute ANM on ensemble mean structure.")
    ap.add_argument("--gnm", action="store_true",
                    help="Compute GNM on ensemble mean structure.")
    ap.add_argument("--n-enm-modes", type=int, default=20,
                    help="Number of ANM/GNM modes (default: 20).")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="Subsample at most this many structures (0 = all).")
    args = ap.parse_args()

    if args.ckpt_dir and not args.topo_npz:
        ap.error("--topo-npz is required when using --ckpt-dir.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: resolve PDB directory ────────────────────────────────────────
    if args.ckpt_dir:
        from genai_tps.io.boltz_npz_export import batch_export  # noqa: PLC0415
        pdb_dir = args.out_dir / "exported_pdbs"
        logger.info("Exporting last frames from %s → %s …", args.ckpt_dir, pdb_dir)
        pdb_paths = batch_export(args.ckpt_dir, args.topo_npz, pdb_dir)
        logger.info("Exported %d PDB files.", len(pdb_paths))
    else:
        pdb_dir = args.pdb_dir
        pdb_paths = sorted(pdb_dir.glob("*.pdb"))
        logger.info("Using %d PDB files from %s.", len(pdb_paths), pdb_dir)

    if args.max_structures and len(pdb_paths) > args.max_structures:
        import numpy as np
        idx = np.round(
            np.linspace(0, len(pdb_paths) - 1, args.max_structures)
        ).astype(int)
        pdb_paths = [pdb_paths[i] for i in idx]
        logger.info("Subsampled to %d structures.", len(pdb_paths))
        # re-write the pdb_dir to point to a filtered subset is not needed;
        # run_ensemble_analysis accepts a glob so we pass pdb_dir directly,
        # but we want to restrict to the sampled list.  Work-around: use a
        # temp symlink directory — simpler: just pass a custom glob via a
        # wrapper that builds the ensemble from a list of paths.
        # For simplicity, call the library API directly here.
        from genai_tps.evaluation.terminal_ensemble_prody import (  # noqa: PLC0415
            run_ensemble_analysis,
        )
        import prody as pd  # noqa: PLC0415
        pd.confProDy(verbosity="none")
        # build the ensemble from the subsampled paths using the public API
        result = run_ensemble_analysis(
            pdb_dir,
            n_pcs=args.n_pcs,
            run_anm=args.anm,
            run_gnm=args.gnm,
            n_enm_modes=args.n_enm_modes,
            glob="*.pdb",
        )
        # Filter result to only subsampled labels (order may differ)
        wanted = {p.stem for p in pdb_paths}
        keep = [i for i, lbl in enumerate(result.labels) if lbl in wanted]
        import numpy as np
        result.projections = result.projections[keep]
        result.labels = [result.labels[i] for i in keep]
        result.n_structures = len(keep)
    else:
        from genai_tps.evaluation.terminal_ensemble_prody import (  # noqa: PLC0415
            run_ensemble_analysis,
        )
        result = run_ensemble_analysis(
            pdb_dir,
            n_pcs=args.n_pcs,
            run_anm=args.anm,
            run_gnm=args.gnm,
            n_enm_modes=args.n_enm_modes,
        )

    if result.n_structures < 2:
        logger.error("Fewer than 2 structures analysed — aborting.")
        sys.exit(1)

    # ── Step 2: write JSON summary ────────────────────────────────────────────
    json_path = args.out_dir / "prody_terminal_summary.json"
    # Add human-readable mc_steps alongside labels for convenience
    mc_steps = [_extract_mc_step(lbl) for lbl in result.labels]
    d = result.to_dict()
    d["mc_steps"] = mc_steps
    json_path.write_text(json.dumps(d, indent=2))
    logger.info("Summary JSON → %s  (n=%d structures)", json_path, result.n_structures)

    # ── Step 3: plots ─────────────────────────────────────────────────────────
    _plot_scree(
        result.cumulative_variance_ratio.tolist(),
        args.out_dir / "pc_scree.png",
    )
    _plot_pc1_pc2(
        result.projections.tolist(),
        result.labels,
        args.out_dir / "pc1_pc2.png",
    )

    # ── Step 4: print summary ─────────────────────────────────────────────────
    n_90 = int(
        next(
            (i + 1 for i, v in enumerate(result.cumulative_variance_ratio) if v >= 0.9),
            len(result.cumulative_variance_ratio),
        )
    )
    logger.info(
        "PCA summary: %d structures, %d Cα atoms, "
        "%d PCs needed for 90%% variance.",
        result.n_structures,
        result.n_ca_atoms,
        n_90,
    )
    for i, (ev, evr) in enumerate(
        zip(result.eigenvalues, result.explained_variance_ratio)
    ):
        logger.info("  PC%d: eigval=%.1f Å²  var=%.1f%%", i + 1, ev, 100.0 * evr)


if __name__ == "__main__":
    main()

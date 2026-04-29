#!/usr/bin/env python3
"""Live Cα-RMSD watcher for an ongoing TPS run.

Polls a checkpoint directory for new ``tps_mc_step_*.npz`` files, converts
each to a single-frame PDB (using the Boltz topology), runs AMBER14 + GBn2
implicit-solvent energy minimisation, computes Kabsch-aligned Cα-RMSD, and
continuously updates a results JSON + distribution PNG.

Usage::

    python scripts/watch_rmsd_live.py \\
        --ckpt-dir  cofolding_tps_out/trajectory_checkpoints \\
        --topo-npz  cofolding_tps_out/boltz_results_cofolding_multimer_msa_empty/processed/structures/cofolding_multimer_msa_empty.npz \\
        --out-dir   cofolding_tps_out/cv_rmsd_analysis_live \\
        --platform  CUDA \\
        --poll-s    15

The script runs until interrupted (Ctrl-C) or until ``--max-steps`` new
checkpoints have been processed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Boltz NPZ → single-frame PDB  (delegated to shared helper)
# ---------------------------------------------------------------------------

def _ensure_analysis_on_path() -> None:
    """Add src/python to sys.path so genai_tps (including evaluation/io) is importable."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src" / "python"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_analysis_on_path()

from genai_tps.io.boltz_npz_export import load_topo, npz_to_pdb  # noqa: E402


# ---------------------------------------------------------------------------
# Plot helper (redrawn after every new result)
# ---------------------------------------------------------------------------

def redraw_plot(
    rmsd_values: list[float],
    out_path: Path,
    n_total_seen: int,
) -> None:
    """Overwrite the distribution PNG with all results collected so far."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmsd_values, bins="auto", density=True,
            alpha=0.45, color="#2196F3", edgecolor="white", linewidth=0.5,
            label="histogram")
    if len(rmsd_values) >= 4:
        xs  = np.linspace(min(rmsd_values) * 0.8, max(rmsd_values) * 1.2, 300)
        kde = gaussian_kde(rmsd_values, bw_method="scott")
        ax.plot(xs, kde(xs), color="#E91E63", linewidth=2, label="KDE")

    mean_v = float(np.mean(rmsd_values))
    ax.axvline(mean_v, color="#FF9800", linestyle="--", linewidth=1.4,
               label=f"mean = {mean_v:.3f} Å")

    ax.set_xlabel("Cα-RMSD to local energy minimum (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Boltz terminal structures: Cα RMSD to AMBER14/GBn2 local minimum\n"
        f"n = {len(rmsd_values)}  (MC steps seen: {n_total_seen})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main watcher loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir",  required=True, type=Path,
                    help="Directory containing tps_mc_step_*.npz checkpoints.")
    ap.add_argument("--topo-npz",  required=True, type=Path,
                    help="Boltz processed/structures/*.npz for atom topology.")
    ap.add_argument("--out-dir",   required=True, type=Path,
                    help="Output directory for JSON results and PNG plot.")
    ap.add_argument("--platform",  default="CUDA",
                    choices=["CUDA", "OpenCL", "CPU"],
                    help="OpenMM platform (default: CUDA).")
    ap.add_argument("--max-iter",  type=int, default=1000,
                    help="Max OpenMM minimisation iterations per structure.")
    ap.add_argument("--poll-s",    type=float, default=15.0,
                    help="Seconds between directory polls (default: 15).")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Stop after processing this many new checkpoints "
                         "(0 = run forever until Ctrl-C).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = args.out_dir / "live_pdbs"
    pdb_dir.mkdir(exist_ok=True)

    json_path = args.out_dir / "rmsd_results_live.json"
    plot_path = args.out_dir / "rmsd_distribution_live.png"

    # ── load existing results so the watcher is restartable ─────────────────
    results: list[dict] = []
    processed_stems: set[str] = set()
    if json_path.exists():
        try:
            results = json.loads(json_path.read_text())
            processed_stems = {r["stem"] for r in results if "stem" in r}
            logger.info("Resumed: %d structures already in %s", len(results), json_path)
        except Exception as exc:
            logger.warning("Could not load existing results (%s); starting fresh.", exc)

    # ── load Boltz topology (once) ───────────────────────────────────────────
    root = Path(__file__).resolve().parents[1]
    boltz_src = root / "boltz" / "src"
    if boltz_src.is_dir() and str(boltz_src) not in sys.path:
        sys.path.insert(0, str(boltz_src))
    topo, n_struct = load_topo(args.topo_npz)
    logger.info("Boltz topology loaded: %d atoms", n_struct)

    # ── import minimisation helper ───────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_cv_rmsd import minimize_pdb  # type: ignore[import]

    n_processed = 0
    logger.info("Watching %s  (poll every %.0fs)  — Ctrl-C to stop",
                args.ckpt_dir, args.poll_s)

    try:
        while True:
            # discover all checkpoint NPZ files not yet processed
            all_ckpts = sorted(
                p for p in args.ckpt_dir.glob("tps_mc_step_*.npz")
                if "last_frame" not in p.stem and "latest" not in p.stem
                and p.stem not in processed_stems
            )

            for ckpt in all_ckpts:
                # Convert NPZ → PDB
                pdb_out = pdb_dir / (ckpt.stem + "_last.pdb")
                try:
                    npz_to_pdb(ckpt, topo, n_struct, pdb_out)
                except Exception as exc:
                    logger.warning("NPZ→PDB failed for %s: %s", ckpt.name, exc)
                    processed_stems.add(ckpt.stem)
                    continue

                # Minimise + RMSD
                logger.info("Processing %s …", ckpt.stem)
                res = minimize_pdb(pdb_out, max_iter=args.max_iter,
                                   platform_name=args.platform)
                res["stem"]    = ckpt.stem
                res["ckpt"]    = str(ckpt)
                res["pdb_out"] = str(pdb_out)
                results.append(res)
                processed_stems.add(ckpt.stem)
                n_processed += 1

                if res["converged"]:
                    logger.info("  → RMSD = %.3f Å  E = %.1f kJ/mol",
                                res["ca_rmsd_angstrom"], res["energy_kj_mol"])
                else:
                    logger.warning("  → FAILED: %s", res["error"])

                # Persist results and redraw plot after every structure
                json_path.write_text(json.dumps(results, indent=2))
                good_rmsds = [r["ca_rmsd_angstrom"] for r in results
                              if r.get("converged") and r["ca_rmsd_angstrom"] is not None]
                if good_rmsds:
                    redraw_plot(good_rmsds, plot_path, n_total_seen=len(processed_stems))
                    logger.info("  Plot updated → %s  (n=%d)", plot_path, len(good_rmsds))

                if args.max_steps and n_processed >= args.max_steps:
                    logger.info("Reached --max-steps %d; stopping.", args.max_steps)
                    return

            if not all_ckpts:
                logger.debug("No new checkpoints; sleeping %.0fs …", args.poll_s)
            time.sleep(args.poll_s)

    except KeyboardInterrupt:
        logger.info("Interrupted. Processed %d structures total.", n_processed)
        good_rmsds = [r["ca_rmsd_angstrom"] for r in results
                      if r.get("converged") and r["ca_rmsd_angstrom"] is not None]
        if good_rmsds:
            redraw_plot(good_rmsds, plot_path, n_total_seen=len(processed_stems))
        logger.info("Final results → %s", json_path)
        logger.info("Final plot    → %s", plot_path)


if __name__ == "__main__":
    main()

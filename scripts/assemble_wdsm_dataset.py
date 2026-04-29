#!/usr/bin/env python3
"""Assemble per-step WDSM NPZ files into a single ReweightedStructureDataset NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble WDSM dataset from per-step NPZs.")
    parser.add_argument("--wdsm-dir", type=Path, required=True, help="Directory with wdsm_step_*.npz files.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    parser.add_argument("--topo-npz", type=Path, default=None, help="Topology NPZ for n_struct (atom mask).")
    parser.add_argument("--max-log-ratio", type=float, default=None, help="Clip extreme log-weights.")
    parser.add_argument("--min-step", type=int, default=0, help="Minimum MC step to include.")
    parser.add_argument("--max-step", type=int, default=None, help="Maximum MC step to include.")
    args = parser.parse_args()

    from genai_tps.training.diagnostics import effective_sample_size, weight_statistics

    from genai_tps.simulation.dataset_assembly import assemble_wdsm_from_directory, save_assembled_npz

    assembled = assemble_wdsm_from_directory(
        Path(args.wdsm_dir),
        topo_npz=args.topo_npz,
        max_log_ratio=args.max_log_ratio,
        min_step=args.min_step,
        max_step=args.max_step,
    )

    stats = weight_statistics(assembled.logw)
    n_eff = effective_sample_size(assembled.logw)
    n_samples = assembled.coords.shape[0]
    n_atoms = assembled.coords.shape[1]
    print(f"[assemble] Dataset: {n_samples} samples, {n_atoms} atoms each")
    print(f"[assemble] N_eff={n_eff:.2f} ({n_eff/n_samples*100:.1f}%)")
    print(f"[assemble] logw range: [{assembled.logw.min():.4f}, {assembled.logw.max():.4f}]")
    print(f"[assemble] max_weight_fraction={stats['max_weight_fraction']:.4f}")

    save_assembled_npz(Path(args.output), assembled)
    print(f"[assemble] Saved {args.output}")


if __name__ == "__main__":
    main()

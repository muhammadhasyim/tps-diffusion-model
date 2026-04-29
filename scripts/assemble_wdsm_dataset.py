#!/usr/bin/env python3
"""Assemble per-step WDSM NPZ files into a single ReweightedStructureDataset NPZ.

Reads the ``wdsm_step_*.npz`` files produced by ``run_opes_tps.py --save-wdsm-data-every``
and stacks them into the ``(coords, logw, atom_mask)`` format expected by
``weighted_dsm/dataset.py``.

Example::

    python scripts/assemble_wdsm_dataset.py \
        --wdsm-dir /mnt/shared/.../opes_out/wdsm_samples \
        --topo-npz /mnt/shared/.../processed/structures/system.npz \
        --output /mnt/shared/.../training_data.npz
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble WDSM dataset from per-step NPZs.")
    parser.add_argument("--wdsm-dir", type=Path, required=True, help="Directory with wdsm_step_*.npz files.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    parser.add_argument("--topo-npz", type=Path, default=None, help="Topology NPZ for n_struct (atom mask).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight tempering exponent (default: 1.0).")
    parser.add_argument("--max-log-ratio", type=float, default=None, help="Clip extreme log-weights.")
    parser.add_argument("--min-step", type=int, default=0, help="Minimum MC step to include.")
    parser.add_argument("--max-step", type=int, default=None, help="Maximum MC step to include.")
    args = parser.parse_args()

    from genai_tps.weighted_dsm.diagnostics import (
        clip_log_weights,
        effective_sample_size,
        temper_log_weights,
        weight_statistics,
    )

    wdsm_dir = Path(args.wdsm_dir)
    pattern = re.compile(r"wdsm_step_(\d+)\.npz$")
    files = []
    for f in sorted(wdsm_dir.glob("wdsm_step_*.npz")):
        m = pattern.search(f.name)
        if m:
            step = int(m.group(1))
            if step >= args.min_step and (args.max_step is None or step <= args.max_step):
                files.append((step, f))
    files.sort(key=lambda x: x[0])

    if not files:
        print(f"No wdsm_step_*.npz files found in {wdsm_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[assemble] Found {len(files)} WDSM samples (steps {files[0][0]}..{files[-1][0]})")

    # Determine n_struct for atom mask
    n_struct = None
    if args.topo_npz:
        from genai_tps.analysis.boltz_npz_export import load_topo
        _, n_struct = load_topo(Path(args.topo_npz))
        n_struct = int(n_struct)
        print(f"[assemble] n_struct={n_struct} from topology")

    coords_list = []
    logw_list = []
    cv_list = []

    for step, fpath in files:
        data = np.load(fpath)
        coords_list.append(data["coords"].astype(np.float32))
        logw_list.append(float(data["logw"]))
        if "cv" in data:
            cv_list.append(data["cv"])

    coords_array = np.stack(coords_list, axis=0)  # (N, M, 3)
    logw_array = np.array(logw_list, dtype=np.float64)  # (N,)
    N, M, _ = coords_array.shape

    # Atom mask
    if n_struct is not None and n_struct < M:
        atom_mask = np.zeros((N, M), dtype=np.float32)
        atom_mask[:, :n_struct] = 1.0
    else:
        atom_mask = np.ones((N, M), dtype=np.float32)

    # Weight processing
    if args.gamma != 1.0:
        logw_array = temper_log_weights(logw_array, args.gamma)
        print(f"[assemble] Applied tempering gamma={args.gamma}")
    if args.max_log_ratio is not None:
        logw_array = clip_log_weights(logw_array, args.max_log_ratio)
        print(f"[assemble] Clipped logw with max_log_ratio={args.max_log_ratio}")

    stats = weight_statistics(logw_array)
    n_eff = effective_sample_size(logw_array)
    print(f"[assemble] Dataset: {N} samples, {M} atoms each")
    print(f"[assemble] N_eff={n_eff:.2f} ({n_eff/N*100:.1f}%)")
    print(f"[assemble] logw range: [{logw_array.min():.4f}, {logw_array.max():.4f}]")
    print(f"[assemble] max_weight_fraction={stats['max_weight_fraction']:.4f}")

    if cv_list:
        cvs = np.array(cv_list)
        for i in range(cvs.shape[1] if cvs.ndim > 1 else 1):
            col = cvs[:, i] if cvs.ndim > 1 else cvs
            print(f"[assemble] CV dim {i}: mean={col.mean():.4f} std={col.std():.4f} "
                  f"range=[{col.min():.4f}, {col.max():.4f}]")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, coords=coords_array, logw=logw_array, atom_mask=atom_mask)
    print(f"[assemble] Saved {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Split an assembled WDSM NPZ into train/val sets preserving importance weights.

Example::

    python scripts/split_wdsm_dataset.py \
        --data /mnt/shared/.../training_data.npz \
        --output-dir /mnt/shared/.../merged \
        --val-fraction 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split WDSM dataset into train/val.")
    parser.add_argument("--data", type=Path, required=True, help="Input NPZ (coords, logw, atom_mask).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train.npz and val.npz.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction for validation (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    from genai_tps.weighted_dsm.diagnostics import effective_sample_size, weight_statistics

    data = np.load(args.data)
    coords = data["coords"]
    logw = data["logw"]
    atom_mask = data["atom_mask"] if "atom_mask" in data else np.ones(coords.shape[:2], dtype=np.float32)

    N = len(logw)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(N)
    split = int((1.0 - args.val_fraction) * N)
    train_idx, val_idx = indices[:split], indices[split:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez(out / "train.npz", coords=coords[train_idx], logw=logw[train_idx], atom_mask=atom_mask[train_idx])
    np.savez(out / "val.npz", coords=coords[val_idx], logw=logw[val_idx], atom_mask=atom_mask[val_idx])

    for name, idx in [("Full", np.arange(N)), ("Train", train_idx), ("Val", val_idx)]:
        lw = logw[idx]
        stats = weight_statistics(lw)
        n_eff = effective_sample_size(lw)
        print(f"[split] {name:5s}: N={len(idx):6d}  N_eff={n_eff:7.1f} ({n_eff/len(idx)*100:5.1f}%)  "
              f"logw=[{lw.min():.3f}, {lw.max():.3f}]  std={lw.std():.3f}")

    print(f"\n[split] Saved {out / 'train.npz'} ({len(train_idx)} samples)")
    print(f"[split] Saved {out / 'val.npz'} ({len(val_idx)} samples)")


if __name__ == "__main__":
    main()

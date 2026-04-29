#!/usr/bin/env python3
"""Split an assembled WDSM NPZ into train/val sets preserving importance weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    from genai_tps.training.dataset import split_wdsm_npz_train_val

    split_wdsm_npz_train_val(
        args.data,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

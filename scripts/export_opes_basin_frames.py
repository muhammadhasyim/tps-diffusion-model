#!/usr/bin/env python3
"""Copy TPS trajectory checkpoints whose logged CV lies in a rectangular basin.

The ``cv_value`` written each Metropolis step is the OPES bias CV evaluated on
the **last frame** of the **accepted** path (same definition as
``run_opes_tps.py``). Intermediate diffusion frames along the path are **not**
logged. To classify individual frames you must reload each snapshot from a
checkpoint NPZ and recompute CVs offline.

Checkpoint NPZs live under ``trajectory_checkpoints/tps_mc_step_XXXXXXXX.npz``
(see ``--save-trajectory-every``). This script copies those files when the
corresponding MC step's CV falls inside your axis-aligned basin.

Examples::

    python scripts/export_opes_basin_frames.py \\
        --run-dir opes_tps_out_case1_2d \\
        --x-min 0 --x-max 0.35 --y-min 5.0 --y-max 5.7 \\
        --out-dir opes_tps_out_case1_2d/basin_main

    python scripts/export_opes_basin_frames.py \\
        --run-dir opes_tps_out_case1 --x-min 0 --x-max 0.2 \\
        --out-dir opes_tps_out_case1/basin_low_rmsd --nearest
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

_CKPT_RE = re.compile(r"tps_mc_step_(\d+)\.npz$")


def _load_cv_series(run_dir: Path, cv_path: Path | None) -> tuple[np.ndarray, list[int]]:
    if cv_path is None:
        p = run_dir / "cv_values.json"
        if not p.is_file():
            p = run_dir / "tps_steps.jsonl"
    else:
        p = cv_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"CV file not found: {p}")

    if p.suffix == ".jsonl" or p.name == "tps_steps.jsonl":
        return _from_jsonl(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "cv_values" not in data or "mc_steps" not in data:
        raise ValueError(f"{p}: expected keys 'cv_values' and 'mc_steps'")
    steps = [int(s) for s in data["mc_steps"]]
    raw = data["cv_values"]
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != len(steps):
        raise ValueError(f"{p}: len(mc_steps) != len(cv_values)")
    return arr, steps


def _from_jsonl(p: Path) -> tuple[np.ndarray, list[int]]:
    steps: list[int] = []
    rows: list[list[float]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        st = obj.get("step")
        cv = obj.get("cv_value")
        if st is None or cv is None:
            continue
        steps.append(int(st))
        if isinstance(cv, (list, tuple)):
            rows.append([float(x) for x in cv])
        else:
            rows.append([float(cv)])
    if not steps:
        raise ValueError(f"{p}: no lines with step and cv_value")
    return np.asarray(rows, dtype=np.float64), steps


def _in_basin(
    row: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float | None,
    y_max: float | None,
) -> bool:
    if not (x_min <= float(row[0]) <= x_max):
        return False
    if row.shape[0] >= 2:
        if y_min is None or y_max is None:
            raise ValueError("2-D CV requires --y-min and --y-max")
        return bool(y_min <= float(row[1]) <= y_max)
    return True


def _find_checkpoint(ck_dir: Path, mc_step: int, *, nearest: bool) -> tuple[Path | None, int | None]:
    exact = ck_dir / f"tps_mc_step_{mc_step:08d}.npz"
    if exact.is_file():
        return exact, mc_step
    if not nearest:
        return None, None
    best_s: int | None = None
    best_p: Path | None = None
    for f in ck_dir.glob("tps_mc_step_*.npz"):
        m = _CKPT_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if s <= mc_step and (best_s is None or s > best_s):
            best_s, best_p = s, f
    return best_p, best_s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path, required=True, help="TPS output root.")
    ap.add_argument(
        "--cv-json",
        type=Path,
        default=None,
        help="cv_values.json or tps_steps.jsonl (default: under --run-dir).",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Basin export folder.")
    ap.add_argument("--x-min", type=float, required=True, help="Basin lower edge for CV axis 0.")
    ap.add_argument("--x-max", type=float, required=True, help="Basin upper edge for CV axis 0.")
    ap.add_argument("--y-min", type=float, default=None, help="Basin lower edge for CV axis 1 (2-D).")
    ap.add_argument("--y-max", type=float, default=None, help="Basin upper edge for CV axis 1 (2-D).")
    ap.add_argument(
        "--nearest",
        action="store_true",
        help="If exact tps_mc_step_*.npz is missing, copy latest checkpoint with mc_step <= step.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        cv_mat, mc_steps = _load_cv_series(run_dir, args.cv_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    d = int(cv_mat.shape[1])
    if d >= 3:
        print(f"Only 1-D and 2-D CV columns are supported (got {d}).", file=sys.stderr)
        sys.exit(1)
    if d == 2 and (args.y_min is None or args.y_max is None):
        print("2-D CV data requires --y-min and --y-max.", file=sys.stderr)
        sys.exit(1)

    ck_dir = run_dir / "trajectory_checkpoints"
    if not ck_dir.is_dir():
        print(f"No trajectory_checkpoints under {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_copy = 0
    n_miss = 0

    for mc_step, row in zip(mc_steps, cv_mat, strict=True):
        if not _in_basin(row, args.x_min, args.x_max, args.y_min, args.y_max):
            continue
        src, used_step = _find_checkpoint(ck_dir, mc_step, nearest=bool(args.nearest))
        entry: dict = {
            "mc_step": mc_step,
            "cv_value": row.tolist() if row.size > 1 else float(row[0]),
            "checkpoint_used": None,
            "copied_to": None,
        }
        if src is None:
            n_miss += 1
            entry["note"] = "no_checkpoint"
            manifest.append(entry)
            continue
        entry["checkpoint_used"] = used_step
        dest = out_dir / f"tps_mc_step_{mc_step:08d}.npz"
        shutil.copy2(src, dest)
        entry["copied_to"] = str(dest)
        n_copy += 1
        manifest.append(entry)

    man_path = out_dir / "basin_manifest.json"
    man_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "x_range": [args.x_min, args.x_max],
                "y_range": None
                if args.y_min is None
                else [args.y_min, args.y_max],
                "nearest": bool(args.nearest),
                "n_matched_cv_steps": len(manifest),
                "n_copied": n_copy,
                "n_missing_checkpoint": n_miss,
                "entries": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[export_opes_basin_frames] matched {len(manifest)} MC steps, "
        f"copied {n_copy} checkpoints, {n_miss} without file "
        f"(manifest: {man_path})"
    )


if __name__ == "__main__":
    main()

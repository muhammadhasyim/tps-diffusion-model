#!/usr/bin/env python3
"""Summarize OPES-TPS 2D CV trajectories for publication.

Reads ``cv_values.json`` and (optionally) ``tps_steps.jsonl`` from an OPES-TPS
run with a 2-D CV (e.g. ligand RMSD + ligand–pocket distance) and emits:

* Console: human-readable statistics and LaTeX-friendly inline numbers.
* ``--out-json``: machine-readable summary (for CI / reproducibility).

Usage::

    python scripts/summarize_opes_2d_cv.py \\
        --cv-json  opes_tps_out_case1_2d/cv_values.json \\
        --tps-steps opes_tps_out_case1_2d/tps_steps.jsonl \\
        --rmsd-thresholds 0.5 0.75 1.0 \\
        --pocket-thresholds 5.5 5.8 \\
        --out-json opes_tps_out_case1_2d/cv_summary.json

All statistics are reported both over **all** MC steps and restricted to the
**accepted** subset (steps where ``accepted == true`` in ``tps_steps.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_cv_json(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (cv_matrix, mc_steps, cv_names) from cv_values.json.

    Parameters
    ----------
    path
        Path to ``cv_values.json`` produced by ``run_opes_tps.py``.

    Returns
    -------
    cv_matrix : ndarray, shape (N, ndim)
    mc_steps  : ndarray, shape (N,), int
    cv_names  : list of str, length ndim
    """
    with open(path) as fh:
        data = json.load(fh)

    ndim: int = int(data.get("ndim", 1))
    raw_cvs = data.get("cv_values", [])
    raw_steps = data.get("mc_steps", list(range(len(raw_cvs))))
    cv_names: list[str] = list(data.get("cv_names", [f"cv_{i}" for i in range(ndim)]))

    rows: list[list[float]] = []
    steps_out: list[int] = []
    for row, step in zip(raw_cvs, raw_steps):
        if isinstance(row, (list, tuple)) and len(row) >= ndim:
            try:
                vals = [float(row[i]) for i in range(ndim)]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in vals):
                rows.append(vals)
                steps_out.append(int(step))
        elif isinstance(row, (int, float)):
            if math.isfinite(float(row)):
                rows.append([float(row)])
                steps_out.append(int(step))

    if not rows:
        raise ValueError(f"No valid CV values found in {path}")

    return (
        np.array(rows, dtype=np.float64),
        np.array(steps_out, dtype=np.int64),
        cv_names,
    )


def _parse_tps_steps(path: Path) -> dict[int, dict]:
    """Return {step: entry} from tps_steps.jsonl."""
    result: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = entry.get("step")
            if step is not None:
                result[int(step)] = entry
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentiles(arr: np.ndarray, pcts: list[float]) -> dict[str, float]:
    return {f"p{p:.0f}": float(np.percentile(arr, p)) for p in pcts}


def _tail_fraction(arr: np.ndarray, threshold: float, direction: str = "above") -> float:
    """Fraction of values strictly above (or below) threshold."""
    if direction == "above":
        return float(np.sum(arr > threshold) / len(arr))
    return float(np.sum(arr < threshold) / len(arr))


def _run_length_stats(binary_mask: np.ndarray) -> dict[str, float]:
    """Fraction of 'tail' events that are isolated vs clustered runs."""
    if not np.any(binary_mask):
        return {"isolated_fraction": float("nan"), "mean_run_length": float("nan")}
    # Build run-length encoding
    runs: list[int] = []
    current = binary_mask[0]
    length = 1
    for i in range(1, len(binary_mask)):
        if binary_mask[i] == current:
            length += 1
        else:
            if current:
                runs.append(length)
            current = binary_mask[i]
            length = 1
    if current:
        runs.append(length)
    isolated = sum(1 for r in runs if r == 1)
    return {
        "n_runs": len(runs),
        "isolated_fraction": isolated / len(runs) if runs else float("nan"),
        "mean_run_length": float(np.mean(runs)) if runs else float("nan"),
        "max_run_length": int(max(runs)) if runs else 0,
    }


def _peak_locations_2d(
    cv_matrix: np.ndarray, nbins: int = 30
) -> list[tuple[float, float]]:
    """Return (x, y) CV coordinates of local density maxima (2-D histogram peaks)."""
    x = cv_matrix[:, 0]
    y = cv_matrix[:, 1]
    counts, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    # Naive approach: find all 8-connected local maxima
    peaks = []
    for i in range(1, counts.shape[0] - 1):
        for j in range(1, counts.shape[1] - 1):
            c = counts[i, j]
            if c == 0:
                continue
            neighborhood = counts[i - 1 : i + 2, j - 1 : j + 2]
            if c == neighborhood.max() and c > neighborhood.mean():
                xc = float(0.5 * (xedges[i] + xedges[i + 1]))
                yc = float(0.5 * (yedges[j] + yedges[j + 1]))
                peaks.append((xc, yc, int(c)))
    # Sort by count descending
    peaks.sort(key=lambda t: -t[2])
    return [(t[0], t[1]) for t in peaks[:8]]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def summarize(
    cv_json: Path,
    tps_steps_path: Optional[Path],
    rmsd_thresholds: list[float],
    pocket_thresholds: list[float],
    pct_levels: list[float],
) -> dict:
    cv_matrix, mc_steps, cv_names = _parse_cv_json(cv_json)
    ndim = cv_matrix.shape[1]
    n_total = len(cv_matrix)

    # Build accepted mask aligned to mc_steps
    accepted_mask = np.ones(n_total, dtype=bool)  # default: all included
    n_accepted = n_total
    if tps_steps_path is not None:
        tps_data = _parse_tps_steps(tps_steps_path)
        accepted_mask = np.array(
            [bool(tps_data.get(int(s), {}).get("accepted", True)) for s in mc_steps]
        )
        n_accepted = int(accepted_mask.sum())

    rmsd = cv_matrix[:, 0]
    pocket = cv_matrix[:, 1] if ndim > 1 else None

    result: dict = {
        "n_steps_total": n_total,
        "n_steps_accepted": n_accepted,
        "acceptance_rate": n_accepted / n_total,
        "cv_names": cv_names,
    }

    # --- RMSD statistics ---
    for subset_name, mask in [("all", np.ones(n_total, dtype=bool)), ("accepted", accepted_mask)]:
        arr = rmsd[mask]
        if len(arr) == 0:
            continue
        pct_dict = _percentiles(arr, pct_levels)
        tail_dict = {
            f"tail_fraction_rmsd_gt_{r:.2f}A": _tail_fraction(arr, r, "above")
            for r in rmsd_thresholds
        }
        # Run-length stats for each threshold
        run_stats = {}
        for r in rmsd_thresholds:
            run_stats[f"run_stats_rmsd_gt_{r:.2f}A"] = _run_length_stats(arr > r)
        result[f"rmsd_{subset_name}"] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            **pct_dict,
            **tail_dict,
            **run_stats,
        }

    # --- Pocket distance statistics ---
    if pocket is not None:
        for subset_name, mask in [("all", np.ones(n_total, dtype=bool)), ("accepted", accepted_mask)]:
            arr = pocket[mask]
            if len(arr) == 0:
                continue
            pct_dict = _percentiles(arr, pct_levels)
            tail_dict = {
                f"tail_fraction_pocket_gt_{d:.1f}A": _tail_fraction(arr, d, "above")
                for d in pocket_thresholds
            }
            result[f"pocket_dist_{subset_name}"] = {
                "n": len(arr),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                **pct_dict,
                **tail_dict,
            }

    # --- 2-D peak locations ---
    if ndim >= 2:
        result["density_peaks_2d"] = _peak_locations_2d(cv_matrix)

    return result


def _fmt_latex(result: dict) -> str:
    """Return a block of LaTeX-friendly inline text useful for paper writing."""
    lines = []
    n = result["n_steps_total"]
    n_acc = result["n_steps_accepted"]
    acc_rate = result["acceptance_rate"]
    lines.append(f"% N_total={n}, N_accepted={n_acc}, acceptance_rate={acc_rate:.3f}")

    rmsd_all = result.get("rmsd_all", {})
    rmsd_acc = result.get("rmsd_accepted", {})
    if rmsd_all:
        lines.append(
            f"% RMSD (all): mean={rmsd_all['mean']:.3f} Å, "
            f"std={rmsd_all['std']:.3f} Å, "
            f"median={rmsd_all.get('p50', float('nan')):.3f} Å, "
            f"p95={rmsd_all.get('p95', float('nan')):.3f} Å, "
            f"p99={rmsd_all.get('p99', float('nan')):.3f} Å"
        )
    if rmsd_acc:
        lines.append(
            f"% RMSD (accepted): mean={rmsd_acc['mean']:.3f} Å, "
            f"std={rmsd_acc['std']:.3f} Å, "
            f"p50={rmsd_acc.get('p50', float('nan')):.3f} Å, "
            f"p95={rmsd_acc.get('p95', float('nan')):.3f} Å, "
            f"p99={rmsd_acc.get('p99', float('nan')):.3f} Å"
        )
    # Tail fractions
    for key, val in rmsd_all.items():
        if key.startswith("tail_fraction_rmsd"):
            lines.append(f"% {key} = {val:.4f} ({val*100:.2f}%)")
    pk = result.get("density_peaks_2d", [])
    if pk:
        lines.append("% 2D density peaks (RMSD, pocket_dist):")
        for i, (rx, dy) in enumerate(pk[:4]):
            lines.append(f"%   peak {i+1}: RMSD={rx:.3f} Å, pocket_dist={dy:.3f} Å")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv-json", type=Path, required=True, help="cv_values.json path")
    parser.add_argument("--tps-steps", type=Path, default=None, help="tps_steps.jsonl path (optional)")
    parser.add_argument("--rmsd-thresholds", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25],
                        help="RMSD threshold values (Å) for tail fraction computation")
    parser.add_argument("--pocket-thresholds", type=float, nargs="+", default=[5.5, 5.6, 5.7, 5.8],
                        help="Pocket distance thresholds (Å) for tail fraction computation")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[5, 25, 50, 75, 90, 95, 99],
                        help="Percentile levels to report")
    parser.add_argument("--out-json", type=Path, default=None, help="Write JSON summary to this file")
    args = parser.parse_args()

    result = summarize(
        cv_json=args.cv_json,
        tps_steps_path=args.tps_steps,
        rmsd_thresholds=args.rmsd_thresholds,
        pocket_thresholds=args.pocket_thresholds,
        pct_levels=args.percentiles,
    )

    print("=" * 60)
    print("OPES-TPS 2D CV Summary")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    print()
    print("--- LaTeX-friendly inline numbers ---")
    print(_fmt_latex(result))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\nSummary written to: {args.out_json}")


if __name__ == "__main__":
    main()

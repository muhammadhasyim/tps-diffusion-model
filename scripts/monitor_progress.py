#!/usr/bin/env python3
"""Live monitoring dashboard for multi-batch OPES-TPS pipeline.

Scans all batch directories, loads WDSM samples, and generates:
  1. 2D heatmap of (ligand_rmsd, ligand_pocket_dist) coverage
  2. logw distribution histogram with N_eff annotation
  3. Cumulative N_eff trend over steps
  4. CV time series with acceptance coloring
  5. Summary JSON with all statistics

Can run inside a lightweight CPU sandbox (no GPU needed) or locally.

Usage (inside sandbox):
    python monitor_progress.py --base-dir /mnt/shared/.../opes_wdsm_12k_long

Usage (periodic via local cron-style loop):
    python auto_monitor.py --interval 600  # every 10 min
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BASE_DIR_DEFAULT = "/mnt/shared/26-04-22-tps-diffusion/opes_wdsm_12k_long"


def collect_wdsm_samples(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CV values, logw, and step indices from all batches.

    Returns (cvs, logws, global_steps) arrays, sorted by global step.
    """
    all_cvs, all_logws, all_steps = [], [], []
    global_offset = 0

    batch_dirs = []
    for d in sorted(base_dir.iterdir()):
        m = re.match(r"batch_(\d+)$", d.name)
        if m and d.is_dir():
            batch_dirs.append((int(m.group(1)), d))

    if not batch_dirs:
        wdsm_dir = base_dir / "wdsm_samples"
        if wdsm_dir.is_dir():
            batch_dirs = [(0, base_dir)]

    for batch_num, batch_dir in batch_dirs:
        wdsm_dir = batch_dir / "wdsm_samples"
        if not wdsm_dir.is_dir():
            continue

        files = sorted(wdsm_dir.glob("wdsm_step_*.npz"))
        for f in files:
            m = re.search(r"wdsm_step_(\d+)\.npz$", f.name)
            if not m:
                continue
            local_step = int(m.group(1))
            try:
                d = np.load(f)
                cv = d["cv"]
                logw = float(d["logw"])
                all_cvs.append(cv)
                all_logws.append(logw)
                all_steps.append(global_offset + local_step)
            except Exception:
                continue

        if files:
            global_offset += len(files)

    if not all_cvs:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)

    return np.array(all_cvs), np.array(all_logws), np.array(all_steps)


def collect_tps_acceptance(base_dir: Path) -> list[dict]:
    """Load TPS step logs from all batches."""
    all_steps = []
    global_offset = 0

    batch_dirs = []
    for d in sorted(base_dir.iterdir()):
        m = re.match(r"batch_(\d+)$", d.name)
        if m and d.is_dir():
            batch_dirs.append((int(m.group(1)), d))

    if not batch_dirs:
        jsonl = base_dir / "tps_steps.jsonl"
        if jsonl.is_file():
            batch_dirs = [(0, base_dir)]

    for batch_num, batch_dir in batch_dirs:
        jsonl = batch_dir / "tps_steps.jsonl"
        if not jsonl.is_file():
            continue
        n_in_batch = 0
        with open(jsonl) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry["global_step"] = global_offset + entry.get("step", n_in_batch + 1)
                    all_steps.append(entry)
                    n_in_batch += 1
                except json.JSONDecodeError:
                    continue
        global_offset += n_in_batch

    return all_steps


def compute_cumulative_neff(logws: np.ndarray, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative N_eff fraction at each step."""
    steps_out, neff_out = [], []
    for i in range(window, len(logws) + 1, max(1, window // 5)):
        chunk = logws[:i]
        w = np.exp(chunk - chunk.max())
        w /= w.sum()
        n_eff = 1.0 / np.sum(w**2)
        steps_out.append(i)
        neff_out.append(n_eff / i)
    return np.array(steps_out), np.array(neff_out)


def compute_rolling_neff(logws: np.ndarray, window: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling N_eff fraction in a sliding window."""
    steps_out, neff_out = [], []
    for i in range(window, len(logws) + 1, max(1, window // 10)):
        chunk = logws[i - window:i]
        w = np.exp(chunk - chunk.max())
        w /= w.sum()
        n_eff = 1.0 / np.sum(w**2)
        steps_out.append(i)
        neff_out.append(n_eff / window)
    return np.array(steps_out), np.array(neff_out)


def generate_dashboard(
    base_dir: Path,
    output_dir: Path,
    target_steps: int = 12000,
) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cvs, logws, global_steps = collect_wdsm_samples(base_dir)
    tps_steps = collect_tps_acceptance(base_dir)
    N = len(logws)

    if N == 0:
        print("[monitor] No WDSM samples found.")
        return {"n_samples": 0}

    rmsd = cvs[:, 0]
    pocket = cvs[:, 1]

    w = np.exp(logws - logws.max())
    w_norm = w / w.sum()
    n_eff = 1.0 / np.sum(w_norm**2)
    n_eff_frac = n_eff / N

    n_accepted = sum(1 for s in tps_steps if s.get("accepted"))
    accept_rate = n_accepted / len(tps_steps) if tps_steps else 0

    cum_steps, cum_neff = compute_cumulative_neff(logws)
    roll_steps, roll_neff = compute_rolling_neff(logws)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        f"OPES-TPS Progress Dashboard — MEK1+FZC\n"
        f"{N}/{target_steps} steps ({N/target_steps*100:.1f}%) | "
        f"N_eff={n_eff:.0f} ({n_eff_frac*100:.1f}%) | "
        f"Accept={accept_rate*100:.1f}% | "
        f"logw std={logws.std():.3f}",
        fontsize=14, fontweight='bold', y=0.98,
    )

    # 1) Unweighted 2D heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    rmsd_bins = np.linspace(rmsd.min() - 0.05, rmsd.max() + 0.05, 40)
    pocket_bins = np.linspace(pocket.min() - 0.02, pocket.max() + 0.02, 30)
    ax1.hist2d(rmsd, pocket, bins=[rmsd_bins, pocket_bins], cmap='viridis', cmin=1)
    ax1.set_xlabel('Ligand RMSD (A)')
    ax1.set_ylabel('Ligand-Pocket Dist (A)')
    ax1.set_title(f'Unweighted Density ({N} samples)')

    # 2) Reweighted 2D heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    h2, _, _ = np.histogram2d(rmsd, pocket, bins=[rmsd_bins, pocket_bins], weights=w_norm * N)
    h2_masked = np.ma.masked_where(h2 == 0, h2)
    im2 = ax2.pcolormesh(rmsd_bins, pocket_bins, h2_masked.T, cmap='magma', shading='flat')
    ax2.set_xlabel('Ligand RMSD (A)')
    ax2.set_ylabel('Ligand-Pocket Dist (A)')
    ax2.set_title(f'Reweighted Density (OPES)')
    plt.colorbar(im2, ax=ax2, label='Weighted count')

    # 3) logw scatter in CV space
    ax3 = fig.add_subplot(gs[0, 2])
    vmin = logws.mean() - 2 * logws.std()
    vmax = logws.mean() + 2 * logws.std()
    sc = ax3.scatter(rmsd, pocket, c=logws, cmap='coolwarm', s=3, alpha=0.5, vmin=vmin, vmax=vmax)
    ax3.set_xlabel('Ligand RMSD (A)')
    ax3.set_ylabel('Ligand-Pocket Dist (A)')
    ax3.set_title(f'logw per sample')
    plt.colorbar(sc, ax=ax3, label='log w')

    # 4) logw distribution histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(logws, bins=60, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax4.axvline(logws.mean(), color='red', linestyle='--', label=f'mean={logws.mean():.2f}')
    ax4.axvline(0, color='black', linestyle=':', alpha=0.5)
    ax4.set_xlabel('log w')
    ax4.set_ylabel('Count')
    ax4.set_title(f'logw Distribution (std={logws.std():.3f})')
    ax4.legend(fontsize=9)

    # 5) Cumulative + rolling N_eff trend
    ax5 = fig.add_subplot(gs[1, 1])
    if len(cum_steps) > 0:
        ax5.plot(cum_steps, cum_neff * 100, 'b-', linewidth=1.5, label='Cumulative N_eff %')
    if len(roll_steps) > 0:
        ax5.plot(roll_steps, roll_neff * 100, 'r-', linewidth=1, alpha=0.7, label='Rolling (200-step)')
    ax5.axhline(100, color='gray', linestyle=':', alpha=0.3)
    ax5.axhline(50, color='orange', linestyle='--', alpha=0.4, label='50% line')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('N_eff / N (%)')
    ax5.set_title('N_eff Trend')
    ax5.legend(fontsize=9)
    ax5.set_ylim(0, 105)

    # 6) CV time series
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(global_steps, rmsd, c=logws, cmap='coolwarm', s=2, alpha=0.6, vmin=vmin, vmax=vmax)
    ax6.set_xlabel('Global Step')
    ax6.set_ylabel('Ligand RMSD (A)')
    ax6.set_title('RMSD vs Step (colored by logw)')

    # 7) logw time series
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(global_steps, logws, 'k-', linewidth=0.3, alpha=0.4)
    window = min(100, N // 5) if N > 20 else 1
    if window > 1:
        smoothed = np.convolve(logws, np.ones(window) / window, mode='valid')
        ax7.plot(global_steps[window-1:], smoothed, 'r-', linewidth=1.5, label=f'{window}-step avg')
        ax7.legend(fontsize=9)
    ax7.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax7.set_xlabel('Global Step')
    ax7.set_ylabel('log w')
    ax7.set_title('logw Time Series')

    # 8) Pocket distance time series
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(global_steps, pocket, c=logws, cmap='coolwarm', s=2, alpha=0.6, vmin=vmin, vmax=vmax)
    ax8.set_xlabel('Global Step')
    ax8.set_ylabel('Ligand-Pocket Dist (A)')
    ax8.set_title('Pocket Distance vs Step')

    # 9) Batch progress bar
    ax9 = fig.add_subplot(gs[2, 2])
    batch_dirs = []
    for d in sorted(base_dir.iterdir()):
        m = re.match(r"batch_(\d+)$", d.name)
        if m and d.is_dir():
            wdsm = d / "wdsm_samples"
            n = len(list(wdsm.glob("wdsm_step_*.npz"))) if wdsm.is_dir() else 0
            batch_dirs.append((int(m.group(1)), n))
    if not batch_dirs:
        wdsm = base_dir / "wdsm_samples"
        if wdsm.is_dir():
            n = len(list(wdsm.glob("wdsm_step_*.npz")))
            batch_dirs = [(1, n)]

    if batch_dirs:
        labels = [f"B{b}" for b, _ in batch_dirs]
        counts = [c for _, c in batch_dirs]
        colors = ['#2ecc71' if i < len(batch_dirs) - 1 else '#e67e22' for i in range(len(batch_dirs))]
        ax9.barh(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax9.set_xlabel('Steps')
        ax9.set_title(f'Batch Progress ({sum(counts)}/{target_steps})')
        for i, c in enumerate(counts):
            ax9.text(c + 10, i, str(c), va='center', fontsize=9)

    dashboard_path = output_dir / "opes_dashboard.png"
    fig.savefig(str(dashboard_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[monitor] Dashboard saved: {dashboard_path}")

    stats = {
        "n_samples": N,
        "target_steps": target_steps,
        "progress_fraction": N / target_steps,
        "n_eff": float(n_eff),
        "n_eff_fraction": float(n_eff_frac),
        "logw_mean": float(logws.mean()),
        "logw_std": float(logws.std()),
        "logw_min": float(logws.min()),
        "logw_max": float(logws.max()),
        "max_weight_fraction": float(w_norm.max()),
        "acceptance_rate": float(accept_rate),
        "n_batches": len(batch_dirs),
        "cv_rmsd_range": [float(rmsd.min()), float(rmsd.max())],
        "cv_pocket_range": [float(pocket.min()), float(pocket.max())],
    }

    stats_path = output_dir / "monitor_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[monitor] Stats saved: {stats_path}")
    print(f"[monitor] {N}/{target_steps} steps ({N/target_steps*100:.1f}%) | "
          f"N_eff={n_eff:.0f} ({n_eff_frac*100:.1f}%) | logw std={logws.std():.3f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="OPES-TPS progress monitor")
    parser.add_argument("--base-dir", type=Path, default=Path(BASE_DIR_DEFAULT))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-steps", type=int, default=12000)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.base_dir / "monitoring")
    generate_dashboard(args.base_dir, output_dir, args.target_steps)


if __name__ == "__main__":
    main()

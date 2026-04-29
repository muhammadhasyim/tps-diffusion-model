"""Matplotlib dashboards for WDSM / OPES evaluation (script helpers).

Used by ``compare_models.py`` and ``evaluate_wdsm_model.py``.  Plotting-only;
keep structured metrics in JSON outputs produced by those scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_training_cvs(training_dir: Path | str, max_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Load CV values and normalized weights from OpenMM OPES WDSM shards."""
    d = Path(training_dir)
    files = sorted(d.glob("wdsm_step_*.npz")) or sorted(d.glob("batch_*.npz"))
    cvs: list[np.ndarray] = []
    logws: list[float] = []
    for f in files:
        data = np.load(f)
        if "cv" in data and "logw" in data:
            if data["cv"].ndim == 1:
                cvs.append(data["cv"])
                logws.append(float(data["logw"]))
            else:
                cvs.extend(data["cv"])
                logws.extend(data["logw"])
        if len(cvs) >= max_samples:
            break
    arr = np.array(cvs[:max_samples])
    logws_arr = np.array(logws[:max_samples], dtype=np.float64)
    w = np.exp(logws_arr - logws_arr.max())
    w /= w.sum()
    return arr, w


def generate_wdsm_evaluation_dashboard(
    baseline_cvs: dict[str, np.ndarray],
    finetuned_cvs: dict[str, np.ndarray],
    output_dir: Path,
    n_baseline: int,
    n_finetuned: int,
) -> dict[str, dict[str, float]]:
    """Baseline vs fine-tuned histogram dashboard (evaluate script)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    cv_names = sorted(set(baseline_cvs.keys()) & set(finetuned_cvs.keys()))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    ks_results: dict[str, dict[str, float]] = {}
    for i, cv in enumerate(cv_names[:6]):
        ax = axes_flat[i]
        b = baseline_cvs[cv]
        f = finetuned_cvs[cv]
        b = b[np.isfinite(b)]
        f = f[np.isfinite(f)]

        if len(b) > 0 and len(f) > 0:
            ks_stat, p_val = ks_2samp(b, f)
            ks_results[cv] = {"ks_statistic": float(ks_stat), "p_value": float(p_val)}

            lo = min(b.min(), f.min())
            hi = max(b.max(), f.max())
            bins = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 40)

            ax.hist(b, bins=bins, alpha=0.6, color="steelblue", label=f"Baseline (n={len(b)})", density=True)
            ax.hist(f, bins=bins, alpha=0.6, color="coral", label=f"Fine-tuned (n={len(f)})", density=True)
            sig = "*" if p_val < 0.05 else ""
            ax.set_title(f"{cv}\nKS={ks_stat:.3f}, p={p_val:.3g}{sig}")
            ax.legend(fontsize=8)
        ax.set_xlabel(cv)
        ax.set_ylabel("Density")

    fig.suptitle(
        f"WDSM Fine-tuned vs Baseline — MEK1+FZC\n"
        f"Baseline: {n_baseline} samples | Fine-tuned: {n_finetuned} samples",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = Path(output_dir) / "evaluation_dashboard.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Dashboard saved: {path}")
    return ks_results


def generate_four_way_dashboard(
    models_data: dict[str, dict[str, np.ndarray]],
    training_cvs: np.ndarray | None,
    training_weights: np.ndarray | None,
    out_dir: Path,
) -> dict[str, Any]:
    """4-way CV / FES figures (compare_models script)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    model_names = list(models_data.keys())
    colors = {
        "Training (reweighted)": "green",
        "Baseline": "blue",
        "Cartesian-trained": "orange",
        "Quotient-trained": "red",
    }

    cv_names = ["rmsd", "rg", "contact_order", "clash_count"]

    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for i, cv in enumerate(cv_names):
        ax = axes_flat[i]
        for name, data in models_data.items():
            vals = data.get(cv, np.array([]))
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.4, density=True, color=colors.get(name, "gray"), label=name)
        ax.set_xlabel(cv)
        ax.set_ylabel("Density")
        ax.set_title(cv)
        ax.legend(fontsize=7)

    if training_cvs is not None and training_cvs.shape[1] >= 1:
        ax = axes_flat[4]
        ax.hist(
            training_cvs[:, 0],
            bins=30,
            weights=training_weights * len(training_weights),
            alpha=0.4,
            density=True,
            color="green",
            label="Training",
        )
        for name in ["Baseline", "Cartesian-trained", "Quotient-trained"]:
            if name in models_data and "ligand_rmsd" in models_data[name]:
                vals = models_data[name]["ligand_rmsd"]
                if np.any(vals != 0):
                    ax.hist(vals, bins=30, alpha=0.4, density=True, color=colors[name], label=name)
        ax.set_xlabel("Ligand RMSD")
        ax.set_title("Ligand RMSD")
        ax.legend(fontsize=7)

    if training_cvs is not None and training_cvs.shape[1] >= 2:
        ax = axes_flat[5]
        ax.hist(
            training_cvs[:, 1],
            bins=30,
            weights=training_weights * len(training_weights),
            alpha=0.4,
            density=True,
            color="green",
            label="Training",
        )
        for name in ["Baseline", "Cartesian-trained", "Quotient-trained"]:
            if name in models_data and "ligand_pocket_dist" in models_data[name]:
                vals = models_data[name]["ligand_pocket_dist"]
                if np.any(vals != 0):
                    ax.hist(vals, bins=30, alpha=0.4, density=True, color=colors[name], label=name)
        ax.set_xlabel("Pocket Dist")
        ax.set_title("Pocket Distance")
        ax.legend(fontsize=7)

    fig1.suptitle("4-Way CV Distribution Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1.savefig(str(out_dir / "cv_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    if training_cvs is not None and training_cvs.shape[1] >= 2:
        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
        rmsd_bins = np.linspace(0, max(training_cvs[:, 0].max(), 5), 35)
        pocket_bins = np.linspace(training_cvs[:, 1].min() - 0.2, training_cvs[:, 1].max() + 0.2, 25)

        hist_h, xe, ye = np.histogram2d(
            training_cvs[:, 0],
            training_cvs[:, 1],
            bins=[rmsd_bins, pocket_bins],
            weights=training_weights * len(training_weights),
        )
        hist_h[hist_h <= 0] = np.nan
        f_train = -np.log(hist_h)
        f_train -= np.nanmin(f_train)
        axes2[0].pcolormesh(xe, ye, f_train.T, cmap="RdYlBu_r", vmin=0, vmax=6, shading="flat")
        axes2[0].set_title("Training (OPES-MD)")
        axes2[0].set_xlabel("Ligand RMSD")
        axes2[0].set_ylabel("Pocket Dist")

        for idx, name in enumerate(["Baseline", "Cartesian-trained", "Quotient-trained"]):
            if name not in models_data:
                continue
            ax = axes2[idx + 1]
            rmsd_vals = models_data[name].get("ligand_rmsd", np.array([]))
            pocket_vals = models_data[name].get("ligand_pocket_dist", np.array([]))
            if len(rmsd_vals) > 0 and np.any(rmsd_vals != 0):
                h2, _, _ = np.histogram2d(rmsd_vals, pocket_vals, bins=[rmsd_bins, pocket_bins])
                h2[h2 <= 0] = np.nan
                f2 = -np.log(h2)
                f2 -= np.nanmin(f2)
                ax.pcolormesh(xe, ye, f2.T, cmap="RdYlBu_r", vmin=0, vmax=6, shading="flat")
            ax.set_title(name)
            ax.set_xlabel("Ligand RMSD")

        fig2.suptitle("2D Free Energy Surfaces", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig2.savefig(str(out_dir / "fes_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    ks_results: dict[str, Any] = {}
    for cv in cv_names:
        ks_results[cv] = {}
        baseline = models_data.get("Baseline", {}).get(cv, np.array([]))
        for name in ["Cartesian-trained", "Quotient-trained"]:
            vals = models_data.get(name, {}).get(cv, np.array([]))
            if len(baseline) > 0 and len(vals) > 0:
                ks_stat, p = ks_2samp(baseline, vals)
                pooled_std = np.sqrt((baseline.var() + vals.var()) / 2)
                cohens_d = (vals.mean() - baseline.mean()) / pooled_std if pooled_std > 0 else 0
                ks_results[cv][name] = {
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p),
                    "cohens_d": float(cohens_d),
                    "mean_shift": float(vals.mean() - baseline.mean()),
                }

    return ks_results
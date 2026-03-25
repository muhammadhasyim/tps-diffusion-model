#!/usr/bin/env python3
"""Validate enhanced sampling by comparing reweighted distributions to vanilla TPS.

This script loads RMSD distributions from:
  1. Vanilla (unbiased) TPS -- from ``watch_rmsd_live.py`` output
  2. Enhanced sampling TPS -- either exponential tilting or OPES

It applies the appropriate reweighting to the enhanced sampling results and
compares the reconstructed distribution to the vanilla baseline using:
  - Visual overlay (histogram + KDE)
  - Kolmogorov-Smirnov test
  - Effective sample size

Usage::

    # Compare tilting + MBAR results against vanilla:
    python scripts/validate_enhanced_sampling.py \\
        --vanilla-json  cofolding_tps_out/cv_rmsd_analysis_live/rmsd_results_live.json \\
        --mbar-json     cofolding_tps_out_tilted/mbar_samples.json \\
        --out-dir       validation_results/

    # Compare OPES-reweighted results against vanilla:
    python scripts/validate_enhanced_sampling.py \\
        --vanilla-json  cofolding_tps_out/cv_rmsd_analysis_live/rmsd_results_live.json \\
        --opes-state    cofolding_tps_out_opes/opes_state.json \\
        --opes-cv-json  cofolding_tps_out_opes/cv_values.json \\
        --out-dir       validation_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_vanilla_rmsds(json_path: Path) -> np.ndarray:
    """Load RMSD values from a watch_rmsd_live.py results JSON."""
    data = json.loads(json_path.read_text())
    rmsds = [
        r["ca_rmsd_angstrom"]
        for r in data
        if r.get("converged") and r.get("ca_rmsd_angstrom") is not None
    ]
    if not rmsds:
        raise ValueError(f"No converged RMSD values in {json_path}")
    return np.array(rmsds, dtype=np.float64)


def ks_test(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test."""
    from scipy.stats import ks_2samp

    stat, p_value = ks_2samp(sample_a, sample_b)
    return {"ks_statistic": float(stat), "p_value": float(p_value)}


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS from normalized weights: (sum w)^2 / sum(w^2)."""
    w = weights / weights.sum()
    return 1.0 / (w ** 2).sum()


def plot_comparison(
    vanilla: np.ndarray,
    enhanced: np.ndarray,
    weights: np.ndarray | None,
    out_path: Path,
    *,
    title: str = "Enhanced Sampling Validation",
    label_enhanced: str = "Enhanced (reweighted)",
) -> None:
    """Overlay vanilla and reweighted enhanced distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_vals = np.concatenate([vanilla, enhanced])
    lo = all_vals.min() - 0.5
    hi = all_vals.max() + 0.5
    bins = np.linspace(lo, hi, 40)

    ax = axes[0]
    ax.hist(vanilla, bins=bins, density=True, alpha=0.5, color="#2196F3",
            edgecolor="white", linewidth=0.5, label="Vanilla TPS")
    if weights is not None:
        ax.hist(enhanced, bins=bins, density=True, weights=weights, alpha=0.5,
                color="#E91E63", edgecolor="white", linewidth=0.5,
                label=label_enhanced)
    else:
        ax.hist(enhanced, bins=bins, density=True, alpha=0.5, color="#E91E63",
                edgecolor="white", linewidth=0.5, label=label_enhanced)

    if len(vanilla) >= 4:
        xs = np.linspace(lo, hi, 300)
        kde_v = gaussian_kde(vanilla, bw_method="scott")
        ax.plot(xs, kde_v(xs), color="#1565C0", linewidth=2, label="Vanilla KDE")
    if len(enhanced) >= 4 and weights is not None:
        try:
            kde_e = gaussian_kde(enhanced, bw_method="scott", weights=weights)
            ax.plot(xs, kde_e(xs), color="#AD1457", linewidth=2, linestyle="--",
                    label="Enhanced KDE")
        except Exception:
            pass

    ax.set_xlabel("Cα-RMSD (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(vanilla, bins=bins, density=True, cumulative=True, histtype="step",
            linewidth=2, color="#2196F3", label="Vanilla CDF")
    if weights is not None:
        sorted_idx = np.argsort(enhanced)
        sorted_cv = enhanced[sorted_idx]
        sorted_w = weights[sorted_idx]
        cdf = np.cumsum(sorted_w) / sorted_w.sum()
        ax.step(sorted_cv, cdf, color="#E91E63", linewidth=2, linestyle="--",
                label="Enhanced CDF")
    else:
        ax.hist(enhanced, bins=bins, density=True, cumulative=True, histtype="step",
                linewidth=2, color="#E91E63", linestyle="--", label="Enhanced CDF")
    ax.set_xlabel("Cα-RMSD (Å)", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title("CDF Comparison", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot → %s", out_path)


def validate_mbar(vanilla_json: Path, mbar_json: Path, out_dir: Path) -> dict:
    """Validate exponential tilting + MBAR against vanilla distribution."""
    from genai_tps.enhanced_sampling.mbar_analysis import MBARDistributionEstimator

    vanilla = load_vanilla_rmsds(vanilla_json)
    logger.info("Vanilla: %d samples, mean=%.3f Å", len(vanilla), vanilla.mean())

    estimator = MBARDistributionEstimator.load_samples(mbar_json)
    logger.info("MBAR: %d states, %d total samples",
                estimator.n_states, estimator.n_total_samples)

    result = estimator.estimate(
        n_bins=40,
        cv_range=(vanilla.min() - 1.0, vanilla.max() + 1.0),
    )

    enhanced_cvs = np.concatenate([
        np.array(s.cv_values) for s in estimator._state_samples
    ])

    plot_comparison(
        vanilla, enhanced_cvs, weights=None,
        out_path=out_dir / "validation_mbar.png",
        title="MBAR Reweighted vs Vanilla TPS",
        label_enhanced="MBAR combined (all lambdas)",
    )

    ks = ks_test(vanilla, enhanced_cvs)
    summary = {
        "method": "exponential_tilting_mbar",
        "n_vanilla": len(vanilla),
        "vanilla_mean": float(vanilla.mean()),
        "vanilla_std": float(vanilla.std()),
        "n_enhanced": len(enhanced_cvs),
        "enhanced_mean": float(enhanced_cvs.mean()),
        "enhanced_std": float(enhanced_cvs.std()),
        "ks_statistic": ks["ks_statistic"],
        "ks_p_value": ks["p_value"],
        "n_states": estimator.n_states,
    }
    return summary


def validate_opes(
    vanilla_json: Path, opes_state: Path, opes_cv_json: Path, out_dir: Path
) -> dict:
    """Validate OPES-reweighted results against vanilla distribution."""
    from genai_tps.enhanced_sampling.opes_bias import OPESBias

    vanilla = load_vanilla_rmsds(vanilla_json)
    logger.info("Vanilla: %d samples, mean=%.3f Å", len(vanilla), vanilla.mean())

    bias = OPESBias.load_state(opes_state)
    logger.info("OPES: %d kernels, counter=%d, zed=%.4g",
                bias.n_kernels, bias.counter, bias.zed)

    cv_data = json.loads(Path(opes_cv_json).read_text())
    enhanced_cvs = np.array(cv_data["cv_values"], dtype=np.float64)
    logger.info("OPES CV values: %d samples", len(enhanced_cvs))

    weights = bias.reweight_samples(enhanced_cvs)
    ess = effective_sample_size(weights)
    logger.info("Effective sample size: %.1f / %d (%.1f%%)",
                ess, len(enhanced_cvs), 100.0 * ess / len(enhanced_cvs))

    plot_comparison(
        vanilla, enhanced_cvs, weights=weights,
        out_path=out_dir / "validation_opes.png",
        title="OPES Reweighted vs Vanilla TPS",
        label_enhanced="OPES (reweighted)",
    )

    weighted_mean = np.average(enhanced_cvs, weights=weights)
    weighted_var = np.average((enhanced_cvs - weighted_mean) ** 2, weights=weights)

    ks = ks_test(vanilla, enhanced_cvs)

    summary = {
        "method": "opes_adaptive",
        "n_vanilla": len(vanilla),
        "vanilla_mean": float(vanilla.mean()),
        "vanilla_std": float(vanilla.std()),
        "n_enhanced": len(enhanced_cvs),
        "enhanced_reweighted_mean": float(weighted_mean),
        "enhanced_reweighted_std": float(np.sqrt(weighted_var)),
        "ks_statistic": ks["ks_statistic"],
        "ks_p_value": ks["p_value"],
        "effective_sample_size": float(ess),
        "ess_fraction": float(ess / len(enhanced_cvs)),
        "n_kernels": bias.n_kernels,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vanilla-json", type=Path, required=True,
        help="Path to rmsd_results_live.json from vanilla TPS.",
    )
    parser.add_argument(
        "--mbar-json", type=Path, default=None,
        help="Path to MBAR samples JSON (for exponential tilting validation).",
    )
    parser.add_argument(
        "--opes-state", type=Path, default=None,
        help="Path to OPES state JSON (for OPES validation).",
    )
    parser.add_argument(
        "--opes-cv-json", type=Path, default=None,
        help="Path to JSON with OPES CV values (for OPES validation).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for validation results.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.mbar_json is not None:
        logger.info("=== Validating Exponential Tilting + MBAR ===")
        results["mbar"] = validate_mbar(
            args.vanilla_json, args.mbar_json, args.out_dir
        )

    if args.opes_state is not None and args.opes_cv_json is not None:
        logger.info("=== Validating OPES Adaptive Bias ===")
        results["opes"] = validate_opes(
            args.vanilla_json, args.opes_state, args.opes_cv_json, args.out_dir
        )

    if not results:
        logger.error("No enhanced sampling data provided. Use --mbar-json or --opes-state + --opes-cv-json.")
        sys.exit(1)

    summary_path = args.out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info("Validation summary → %s", summary_path)

    for method, res in results.items():
        logger.info(
            "%s: KS=%.4f  p=%.4g  vanilla_mean=%.3f  enhanced_mean=%.3f",
            method,
            res.get("ks_statistic", float("nan")),
            res.get("ks_p_value", float("nan")),
            res.get("vanilla_mean", float("nan")),
            res.get("enhanced_reweighted_mean", res.get("enhanced_mean", float("nan"))),
        )


if __name__ == "__main__":
    main()

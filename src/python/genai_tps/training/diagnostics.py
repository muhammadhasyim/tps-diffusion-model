"""Weight diagnostics for importance-weighted training."""

from __future__ import annotations

import numpy as np


def effective_sample_size(logw: np.ndarray) -> float:
    """Kish effective sample size from log importance weights.

    N_eff = (sum w_i)^2 / sum w_i^2 = 1 / sum (w_norm_i)^2
    where w_norm_i = softmax(logw)_i.
    """
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    shifted = logw - logw.max()
    w = np.exp(shifted)
    w_sum = w.sum()
    if w_sum == 0.0:
        return 0.0
    w_norm = w / w_sum
    return float(1.0 / np.sum(w_norm ** 2))


def clip_log_weights(logw: np.ndarray, max_log_ratio: float) -> np.ndarray:
    """Clip extreme log-weights so max(logw) - mean_log <= max_log_ratio.

    Shifts outliers down while preserving relative ordering of non-outlier
    weights.  The mean is computed in log-space via logsumexp for stability.
    """
    logw = np.asarray(logw, dtype=np.float64).copy()
    if logw.size <= 1:
        return logw
    log_mean = np.log(np.mean(np.exp(logw - logw.max()))) + logw.max()
    ceiling = log_mean + max_log_ratio
    np.minimum(logw, ceiling, out=logw)
    return logw


def weight_statistics(logw: np.ndarray) -> dict[str, float]:
    """Summary statistics for a set of log importance weights."""
    logw = np.asarray(logw, dtype=np.float64)
    n = logw.size
    if n == 0:
        return {
            "n_eff": 0.0,
            "n_eff_fraction": 0.0,
            "max_weight_fraction": 0.0,
            "min_logw": 0.0,
            "max_logw": 0.0,
        }
    n_eff = effective_sample_size(logw)
    shifted = logw - logw.max()
    w_norm = np.exp(shifted) / np.exp(shifted).sum()
    return {
        "n_eff": n_eff,
        "n_eff_fraction": n_eff / n,
        "max_weight_fraction": float(w_norm.max()),
        "min_logw": float(logw.min()),
        "max_logw": float(logw.max()),
    }

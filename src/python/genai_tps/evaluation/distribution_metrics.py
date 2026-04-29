"""Distribution-level evaluation metrics for ensemble quality assessment.

Implements the statistical metrics used in Wang et al. 2026 (AnewSampling,
bioRxiv 10.64898/2026.03.10.710952) and related structural biology benchmarks.

All functions operate on plain NumPy arrays and depend only on NumPy/SciPy,
making them usable in any pipeline (analysis notebooks, evaluation scripts)
without requiring PyTorch or simulation libraries.

Metric summary
--------------
+---------------------------+------------------------------+----------------------------+
| Function                  | Measures                     | Source                     |
+===========================+==============================+============================+
| torsion_js_distance       | Distribution overlap on       | AnewSampling § A.1         |
|                           | dihedral angles               |                            |
+---------------------------+------------------------------+----------------------------+
| wasserstein_1d            | 1-D Wasserstein (Earth-mover) | SciPy wrapper              |
|                           | distance on any 1-D samples   |                            |
+---------------------------+------------------------------+----------------------------+
| rmsf_spearman_rmse        | Per-residue flexibility       | AnewSampling § A.1         |
|                           | comparison (Spearman ρ, RMSE) |                            |
+---------------------------+------------------------------+----------------------------+
| wasserstein_2_per_atom    | Per-atom 3-D Gaussian W₂     | AnewSampling Eq. 1-2       |
|                           | (RMWD decomposition)          |                            |
+---------------------------+------------------------------+----------------------------+
| ensemble_rmsf             | Per-atom root-mean-square     | Standard MD analysis       |
|                           | fluctuation from ensemble     |                            |
+---------------------------+------------------------------+----------------------------+

References
----------
Wang Y. et al. (2026). Learning the All-Atom Equilibrium Distribution of
    Biomolecular Interactions at Scale. bioRxiv 10.64898/2026.03.10.710952.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

__all__ = [
    "torsion_js_distance",
    "wasserstein_1d",
    "rmsf_spearman_rmse",
    "wasserstein_2_per_atom",
    "ensemble_rmsf",
]


# ---------------------------------------------------------------------------
# Torsion-angle distribution: Jensen-Shannon distance
# ---------------------------------------------------------------------------

def torsion_js_distance(
    angles_p: np.ndarray,
    angles_q: np.ndarray,
    n_bins: int = 36,
) -> float:
    """Compute the sqrt-Jensen-Shannon divergence between two torsion distributions.

    Matches AnewSampling Appendix A.1: 36 equal-width bins over [-180, 180),
    uniform Dirichlet smoothing (add-one counts) to avoid zero-probability bins.

    Parameters
    ----------
    angles_p:
        1-D array of torsion angles in degrees (predicted / sampled ensemble).
        Values should lie in the range (-180, 180].
    angles_q:
        1-D array of torsion angles in degrees (reference / MD ensemble).
    n_bins:
        Number of histogram bins (default 36, i.e., 10° resolution).

    Returns
    -------
    float
        sqrt(JSD(P || Q)) in [0, 1].  Zero means identical distributions;
        1 means disjoint distributions.

    Notes
    -----
    JSD is symmetric: JSD(P||Q) = JSD(Q||P).
    """
    bin_edges = np.linspace(-180.0, 180.0, n_bins + 1)

    counts_p, _ = np.histogram(angles_p, bins=bin_edges)
    counts_q, _ = np.histogram(angles_q, bins=bin_edges)

    # Add-one (Laplace) smoothing to avoid log(0)
    p = (counts_p + 1.0) / (counts_p.sum() + n_bins)
    q = (counts_q + 1.0) / (counts_q.sum() + n_bins)

    m = 0.5 * (p + q)
    # KL(P || M) + KL(Q || M) = JSD * 2
    kl_pm = np.sum(np.where(p > 0, p * np.log(p / m), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(q / m), 0.0))
    jsd = 0.5 * (kl_pm + kl_qm)

    # Clamp to [0, log(2)] to handle floating-point noise
    jsd = float(np.clip(jsd, 0.0, np.log(2.0)))
    return float(np.sqrt(jsd / np.log(2.0)))  # normalised to [0, 1]


# ---------------------------------------------------------------------------
# 1-D Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein_1d(
    a: np.ndarray,
    b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> float:
    """Compute the 1-D Wasserstein-1 (Earth-Mover) distance.

    Thin wrapper around :func:`scipy.stats.wasserstein_distance`.  Used for
    per-atom pairwise-interaction distance distributions (AnewSampling § A.1)
    and for per-torsion comparisons.

    Parameters
    ----------
    a, b:
        1-D arrays of samples.
    weights_a, weights_b:
        Optional importance weights for each sample (unnormalised).

    Returns
    -------
    float
        W₁ distance in the same units as the input arrays.
    """
    return float(wasserstein_distance(a, b, u_weights=weights_a, v_weights=weights_b))


# ---------------------------------------------------------------------------
# Per-residue RMSF: Spearman correlation and RMSE
# ---------------------------------------------------------------------------

def rmsf_spearman_rmse(
    pred_rmsf: np.ndarray,
    ref_rmsf: np.ndarray,
) -> tuple[float, float]:
    """Compare per-residue RMSF between prediction and reference ensembles.

    AnewSampling § A.1 uses this metric to assess how well the sampled
    ensemble reproduces the per-Cα residue flexibility of a reference MD run.

    Parameters
    ----------
    pred_rmsf:
        (N_residues,) predicted per-residue RMSF in Å.
    ref_rmsf:
        (N_residues,) reference per-residue RMSF in Å.

    Returns
    -------
    rho : float
        Spearman rank correlation coefficient in [-1, 1].
    rmse : float
        Root-mean-square error in the same units as the input (Å).
    """
    if pred_rmsf.shape != ref_rmsf.shape:
        raise ValueError(
            f"Shape mismatch: pred_rmsf {pred_rmsf.shape} != ref_rmsf {ref_rmsf.shape}"
        )
    rho_result = spearmanr(pred_rmsf, ref_rmsf)
    rho = float(rho_result.statistic)
    rmse = float(np.sqrt(np.mean((pred_rmsf - ref_rmsf) ** 2)))
    return rho, rmse


# ---------------------------------------------------------------------------
# Per-atom 3-D Wasserstein-2 (RMWD decomposition)
# ---------------------------------------------------------------------------

def wasserstein_2_per_atom(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Per-atom 3-D Wasserstein-2 under the Gaussian approximation.

    Approximates the per-atom W₂ by fitting 3-D Gaussians to each atom's
    ensemble and computing the closed-form W₂ between them
    (Eq. 1-2 of AnewSampling):

        W₂²(P_n, Q_n) = ||μ_P - μ_Q||² + B(Σ_P, Σ_Q)

    where B(·) is the Bures metric:
        B(A, B) = tr(A) + tr(B) - 2 tr((A^{1/2} B A^{1/2})^{1/2})

    This decomposition separates the contribution of mean displacement (MD
    accuracy) from covariance mismatch (flexibility diversity).

    Parameters
    ----------
    coords_p:
        (T_P, N, 3) atom trajectory of ensemble P (predicted or sampled).
    coords_q:
        (T_Q, N, 3) atom trajectory of ensemble Q (reference).
    eps:
        Regularisation added to diagonal of covariance matrices before
        matrix square root to handle near-singular cases.

    Returns
    -------
    dict with keys:
        - ``"w2_squared"``  (N,) total W₂² per atom
        - ``"mean_sq"``     (N,) squared mean displacement ||μ_P - μ_Q||²
        - ``"bures"``       (N,) Bures metric (covariance mismatch)
        - ``"rmwd"``        (N,) sqrt(w2_squared), analogous to RMSD

    Notes
    -----
    For a single-frame ensemble (T=1) the covariance is zero and W₂ reduces
    to the Euclidean distance between the two mean positions.
    """
    if coords_p.ndim != 3 or coords_q.ndim != 3:
        raise ValueError("coords_p and coords_q must be 3-D arrays (T, N, 3)")
    if coords_p.shape[1] != coords_q.shape[1]:
        raise ValueError(
            f"Atom count mismatch: {coords_p.shape[1]} != {coords_q.shape[1]}"
        )
    if coords_p.shape[2] != 3 or coords_q.shape[2] != 3:
        raise ValueError("Last dimension must be 3 (Cartesian coordinates)")

    N = coords_p.shape[1]

    mu_p = coords_p.mean(axis=0)   # (N, 3)
    mu_q = coords_q.mean(axis=0)   # (N, 3)

    mean_sq = np.sum((mu_p - mu_q) ** 2, axis=-1)   # (N,)

    bures = np.zeros(N, dtype=np.float64)
    if coords_p.shape[0] > 1 or coords_q.shape[0] > 1:
        for n in range(N):
            cov_p = _cov3(coords_p[:, n, :], eps=eps)   # (3, 3)
            cov_q = _cov3(coords_q[:, n, :], eps=eps)   # (3, 3)
            bures[n] = _bures(cov_p, cov_q)

    w2_sq = mean_sq + bures
    return {
        "w2_squared": w2_sq,
        "mean_sq": mean_sq,
        "bures": bures,
        "rmwd": np.sqrt(np.maximum(w2_sq, 0.0)),
    }


def _cov3(pts: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """3x3 covariance matrix of a (T, 3) point set."""
    centred = pts - pts.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / max(len(pts) - 1, 1)
    cov += eps * np.eye(3)
    return cov


def _bures(A: np.ndarray, B: np.ndarray) -> float:
    """Bures metric tr(A) + tr(B) - 2 tr((A^{1/2} B A^{1/2})^{1/2})."""
    try:
        # A^{1/2} via eigendecomposition (symmetric positive definite)
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, 0.0)
        sqrt_A = vecs * np.sqrt(vals) @ vecs.T

        M = sqrt_A @ B @ sqrt_A  # A^{1/2} B A^{1/2}
        vals_m = np.linalg.eigvalsh(M)
        sqrt_trace_M = np.sum(np.sqrt(np.maximum(vals_m, 0.0)))

        bures = np.trace(A) + np.trace(B) - 2.0 * sqrt_trace_M
        return float(max(bures, 0.0))
    except np.linalg.LinAlgError:
        return float(np.trace(A) + np.trace(B))


# ---------------------------------------------------------------------------
# Per-atom RMSF
# ---------------------------------------------------------------------------

def ensemble_rmsf(
    coords: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-atom root-mean-square fluctuation (RMSF) from an ensemble.

    RMSF_n = sqrt(mean over t of ||x_n^(t) - <x_n>||²)

    Parameters
    ----------
    coords:
        (T, N, 3) atom trajectories.
    atom_mask:
        (N,) binary mask (1 = real atom, 0 = padding).  Padded atoms
        are set to NaN in the output.

    Returns
    -------
    rmsf : (N,) array
        Per-atom RMSF in the same units as coords.
    """
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must be shape (T, N, 3)")

    mean_pos = coords.mean(axis=0, keepdims=True)   # (1, N, 3)
    sq_disp = ((coords - mean_pos) ** 2).sum(axis=-1)  # (T, N)
    rmsf = np.sqrt(sq_disp.mean(axis=0))               # (N,)

    if atom_mask is not None:
        rmsf = rmsf.astype(float)
        rmsf[atom_mask == 0] = np.nan

    return rmsf

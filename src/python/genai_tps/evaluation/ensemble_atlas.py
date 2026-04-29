"""ATLAS benchmark suite for protein ensemble quality evaluation.

Implements the full set of evaluation metrics used in the ATLAS benchmark
(Vander Meersche et al. 2024, bioRxiv) and adopted by AnewSampling
(Wang et al. 2026) for comparing protein conformational ensembles.

Metrics
-------
1. **Pairwise RMSD Pearson r** -- Pearson correlation between the lower
   triangles of the predicted and reference pairwise-RMSD matrices.
2. **Per-target RMSF** -- Pearson r and RMSE between predicted and reference
   per-Cα RMSF (wrapped from distribution_metrics.rmsf_spearman_rmse, but
   ATLAS uses Pearson, not Spearman).
3. **PCA geometry (Wasserstein-2)** -- W2 between PCA projections of the
   ensembles onto the top-K principal components of the reference.
4. **Transient contact Jaccard** -- Jaccard index between the sets of
   residue-pair contacts that are transiently formed (0 < f < 1) in both
   ensembles.
5. **SASA exposure** -- Pearson r between per-residue solvent-accessible
   surface area (SASA) from the two ensembles.

All functions operate on NumPy arrays.  SASA computation requires ``mdtraj``
if the input is a file path; if per-frame SASA is provided as an array, no
external dependencies are needed.

References
----------
Vander Meersche Y. et al. (2024). ATLAS: protein flexibility description from
    atomistic molecular dynamics simulations. bioRxiv 2024.
Wang Y. et al. (2026). bioRxiv 10.64898/2026.03.10.710952, Appendix A.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr

from genai_tps.evaluation.distribution_metrics import wasserstein_2_per_atom

__all__ = [
    "pairwise_rmsd_pearson",
    "pca_wasserstein2",
    "transient_contact_jaccard",
    "sasa_pearson",
    "atlas_benchmark",
]


# ---------------------------------------------------------------------------
# 1. Pairwise RMSD Pearson correlation
# ---------------------------------------------------------------------------

def pairwise_rmsd_pearson(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    ca_only: bool = False,
    ca_mask: np.ndarray | None = None,
) -> float:
    """Pearson r between lower-triangular pairwise-RMSD matrices.

    Parameters
    ----------
    coords_p, coords_q:
        (T_P, N, 3) and (T_Q, N, 3) atom coordinate ensembles.
        If ``ca_only=True`` or ``ca_mask`` is provided, non-Cα atoms are
        excluded before computing RMSD.
    ca_only:
        If True and ``ca_mask`` is None, all atoms are assumed to be Cα
        (i.e. no filtering is applied).
    ca_mask:
        Optional (N,) boolean array: True for Cα atoms.  When provided,
        only Cα atoms are used.

    Returns
    -------
    float
        Pearson r in [-1, 1].  A higher value indicates better agreement
        of internal conformational distances.
    """
    if ca_mask is not None:
        coords_p = coords_p[:, ca_mask, :]
        coords_q = coords_q[:, ca_mask, :]

    rmsd_p = _pairwise_rmsd_matrix(coords_p)  # (T_P, T_P)
    rmsd_q = _pairwise_rmsd_matrix(coords_q)  # (T_Q, T_Q)

    # Extract lower triangles
    idx_p = np.tril_indices(len(coords_p), k=-1)
    idx_q = np.tril_indices(len(coords_q), k=-1)
    flat_p = rmsd_p[idx_p]
    flat_q = rmsd_q[idx_q]

    if len(flat_p) < 2 or len(flat_q) < 2:
        return float("nan")

    # Resample to the same size for comparison (use the shorter one as reference)
    if len(flat_p) != len(flat_q):
        rng = np.random.default_rng(0)
        size = min(len(flat_p), len(flat_q))
        flat_p = rng.choice(flat_p, size=size, replace=False)
        flat_q = rng.choice(flat_q, size=size, replace=False)

    r, _ = pearsonr(flat_p, flat_q)
    return float(r)


def _pairwise_rmsd_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute symmetric (T, T) pairwise-RMSD matrix."""
    T, N, _ = coords.shape
    rmsd = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        for j in range(i + 1, T):
            diff = coords[i] - coords[j]
            r = float(np.sqrt((diff ** 2).sum() / N))
            rmsd[i, j] = rmsd[j, i] = r
    return rmsd


# ---------------------------------------------------------------------------
# 2. PCA geometry: Wasserstein-2 between PC projections
# ---------------------------------------------------------------------------

def pca_wasserstein2(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    n_components: int = 2,
    ca_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """W2 distance between PCA projections of two ensembles.

    Fits PCA on the reference ensemble (coords_q) and projects both ensembles
    onto the top-K PCs.  Then computes Wasserstein-2 per PC component.

    Parameters
    ----------
    coords_p, coords_q:
        (T_P, N, 3) and (T_Q, N, 3) coordinate ensembles.
    n_components:
        Number of principal components to use.
    ca_mask:
        Optional boolean (N,) mask for Cα selection.

    Returns
    -------
    dict with keys:
        - ``"per_pc_w2"``  : (n_components,) W2 per PC axis
        - ``"mean_w2"``    : float -- mean over all PCs
        - ``"explained_variance"`` : (n_components,) fraction of variance per PC
    """
    if ca_mask is not None:
        coords_p = coords_p[:, ca_mask, :]
        coords_q = coords_q[:, ca_mask, :]

    T_p, N, _ = coords_p.shape
    T_q, N, _ = coords_q.shape

    # Flatten each frame to a 1-D vector of length 3N
    X_p = coords_p.reshape(T_p, -1).astype(np.float64)  # (T_P, 3N)
    X_q = coords_q.reshape(T_q, -1).astype(np.float64)  # (T_Q, 3N)

    # Center on reference mean
    mean_q = X_q.mean(axis=0, keepdims=True)
    X_q_c = X_q - mean_q
    X_p_c = X_p - mean_q

    # PCA on reference via SVD
    _, S, Vt = np.linalg.svd(X_q_c, full_matrices=False)
    total_var = (S ** 2).sum() + 1e-12
    explained = (S[:n_components] ** 2) / total_var

    V = Vt[:n_components].T   # (3N, n_components)
    proj_p = X_p_c @ V        # (T_P, n_components)
    proj_q = X_q_c @ V        # (T_Q, n_components)

    per_pc_w2 = np.zeros(n_components)
    for k in range(n_components):
        # 1-D Gaussian W2: |mu_p - mu_q| + |sigma_p - sigma_q|
        mu_p, sigma_p = proj_p[:, k].mean(), proj_p[:, k].std()
        mu_q, sigma_q = proj_q[:, k].mean(), proj_q[:, k].std()
        per_pc_w2[k] = abs(mu_p - mu_q) + abs(sigma_p - sigma_q)

    return {
        "per_pc_w2": per_pc_w2,
        "mean_w2": float(per_pc_w2.mean()),
        "explained_variance": explained,
    }


# ---------------------------------------------------------------------------
# 3. Transient contact Jaccard
# ---------------------------------------------------------------------------

def transient_contact_jaccard(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    ca_mask: np.ndarray | None = None,
    cutoff_ang: float = 8.0,
    transient_lo: float = 0.05,
    transient_hi: float = 0.95,
) -> float:
    """Jaccard index between transient residue-pair contact sets.

    A contact between residues i and j is "transient" if the fraction of
    frames in which their Cα distance < ``cutoff_ang`` Å lies in
    (``transient_lo``, ``transient_hi``).

    Parameters
    ----------
    coords_p, coords_q:
        (T, N, 3) coordinate ensembles (Cα only, or use ``ca_mask``).
    ca_mask:
        Optional boolean (N,) mask to select Cα atoms.
    cutoff_ang:
        Distance threshold for defining a contact (default 8 Å).
    transient_lo, transient_hi:
        Occupation fraction bounds for "transient" (default 0.05–0.95).

    Returns
    -------
    float
        Jaccard index J = |A ∩ B| / |A ∪ B|.  Returns 1.0 if both sets
        are empty (no transient contacts in either ensemble).
    """
    if ca_mask is not None:
        coords_p = coords_p[:, ca_mask, :]
        coords_q = coords_q[:, ca_mask, :]

    contacts_p = _transient_contacts(coords_p, cutoff_ang, transient_lo, transient_hi)
    contacts_q = _transient_contacts(coords_q, cutoff_ang, transient_lo, transient_hi)

    if not contacts_p and not contacts_q:
        return 1.0
    intersection = len(contacts_p & contacts_q)
    union = len(contacts_p | contacts_q)
    return float(intersection / union) if union > 0 else 1.0


def _transient_contacts(
    coords: np.ndarray,
    cutoff: float,
    lo: float,
    hi: float,
) -> set[tuple[int, int]]:
    """Return the set of (i, j) residue pairs with transient contact."""
    T, N, _ = coords.shape
    contacts: set[tuple[int, int]] = set()
    for i in range(N):
        for j in range(i + 2, N):  # skip i+1 (neighbouring residues)
            dists = np.linalg.norm(coords[:, i, :] - coords[:, j, :], axis=-1)
            frac = (dists < cutoff).mean()
            if lo < frac < hi:
                contacts.add((i, j))
    return contacts


# ---------------------------------------------------------------------------
# 4. SASA Pearson correlation
# ---------------------------------------------------------------------------

def sasa_pearson(
    sasa_p: np.ndarray,
    sasa_q: np.ndarray,
) -> float:
    """Pearson r between per-residue mean SASA of two ensembles.

    Parameters
    ----------
    sasa_p:
        (T_P, N_res) or (N_res,) per-residue SASA for ensemble P.
        If 2-D, the mean over T is taken first.
    sasa_q:
        (T_Q, N_res) or (N_res,) per-residue SASA for ensemble Q.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    mean_p = sasa_p.mean(axis=0) if sasa_p.ndim == 2 else sasa_p
    mean_q = sasa_q.mean(axis=0) if sasa_q.ndim == 2 else sasa_q
    if len(mean_p) != len(mean_q):
        raise ValueError(
            f"Residue count mismatch: {len(mean_p)} != {len(mean_q)}"
        )
    r, _ = pearsonr(mean_p, mean_q)
    return float(r)


# ---------------------------------------------------------------------------
# Composite: full ATLAS benchmark
# ---------------------------------------------------------------------------

def atlas_benchmark(
    coords_p: np.ndarray,
    coords_q: np.ndarray,
    ca_mask: np.ndarray | None = None,
    sasa_p: np.ndarray | None = None,
    sasa_q: np.ndarray | None = None,
    n_pca: int = 2,
    contact_cutoff: float = 8.0,
) -> dict[str, Any]:
    """Run all ATLAS benchmark metrics and return a summary dict.

    Parameters
    ----------
    coords_p, coords_q:
        (T_P, N, 3) and (T_Q, N, 3) all-atom or Cα coordinate ensembles.
    ca_mask:
        (N,) boolean mask for Cα atoms (required if coords include non-Cα).
    sasa_p, sasa_q:
        Pre-computed per-residue SASA arrays (optional).  Shape (T, N_res)
        or (N_res,).  If None, SASA metrics are omitted.
    n_pca:
        Number of principal components for PCA W2.
    contact_cutoff:
        Cα distance cutoff in Å for transient contact analysis.

    Returns
    -------
    dict with keys:
        - ``"pairwise_rmsd_pearson"``  : float
        - ``"pca_w2"``                 : dict from :func:`pca_wasserstein2`
        - ``"transient_contact_jaccard"`` : float
        - ``"rmsf_pearson"``           : float (Pearson r on per-residue RMSF)
        - ``"sasa_pearson"``           : float (NaN if SASA not provided)
    """
    from genai_tps.evaluation.distribution_metrics import ensemble_rmsf

    result: dict[str, Any] = {}

    result["pairwise_rmsd_pearson"] = pairwise_rmsd_pearson(
        coords_p, coords_q, ca_mask=ca_mask
    )

    result["pca_w2"] = pca_wasserstein2(
        coords_p, coords_q, n_components=n_pca, ca_mask=ca_mask
    )

    result["transient_contact_jaccard"] = transient_contact_jaccard(
        coords_p, coords_q, ca_mask=ca_mask, cutoff_ang=contact_cutoff
    )

    # RMSF comparison
    rmsf_p = ensemble_rmsf(
        coords_p[:, ca_mask, :] if ca_mask is not None else coords_p
    )
    rmsf_q = ensemble_rmsf(
        coords_q[:, ca_mask, :] if ca_mask is not None else coords_q
    )
    if len(rmsf_p) == len(rmsf_q) and len(rmsf_p) >= 2:
        r, _ = pearsonr(rmsf_p, rmsf_q)
        result["rmsf_pearson"] = float(r)
    else:
        result["rmsf_pearson"] = float("nan")

    # SASA
    if sasa_p is not None and sasa_q is not None:
        result["sasa_pearson"] = sasa_pearson(sasa_p, sasa_q)
    else:
        result["sasa_pearson"] = float("nan")

    return result

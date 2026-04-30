"""GPU-friendly geometric collective variables on Boltz snapshots.

Scalar functions on snapshots used for diagnostics, training CVs, and
OpenPathSampling :class:`~openpathsampling.collectivevariable.FunctionCV` factories.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from openpathsampling.collectivevariable import FunctionCV

__all__ = [
    "ca_contact_count",
    "clash_count",
    "contact_order",
    "diffusion_step_index_cv",
    "end_to_end_distance",
    "kabsch_rmsd_aligned",
    "lddt_to_reference",
    "make_boltz_plddt_predictor",
    "make_plddt_proxy_cv",
    "make_rg_cv",
    "make_rmsd_cv",
    "make_sigma_cv",
    "radius_of_gyration",
    "ramachandran_outlier_fraction",
    "rmsd_to_reference",
    "shape_acylindricity",
    "shape_kappa2",
]


def _coords_torch(snapshot) -> torch.Tensor:
    if getattr(snapshot, "_tensor_coords_gpu", None) is not None:
        return snapshot._tensor_coords_gpu
    tc = getattr(snapshot, "tensor_coords", None)
    if tc is not None:
        c = tc
        if c.dim() == 2:
            c = c.unsqueeze(0)
        return c
    c = torch.as_tensor(snapshot.coordinates, dtype=torch.float32)
    if c.dim() == 2:
        c = c.unsqueeze(0)
    return c


def radius_of_gyration(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    """Mass-unweighted :math:`R_g` for the first batch element (Angstrom units)."""
    x = _coords_torch(snapshot)[0]
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        sel = m.bool()
        pts = x[sel]
    else:
        pts = x
    c = pts.mean(dim=0)
    rg = torch.sqrt(((pts - c) ** 2).sum(dim=-1).mean() + 1e-12)
    return float(rg.detach().cpu())


def _masked_points(snapshot, atom_mask: torch.Tensor | None) -> torch.Tensor:
    """First-batch coordinates, optionally filtered by ``atom_mask``.

    Parameters
    ----------
    snapshot:
        Object with ``tensor_coords`` / ``coordinates`` (see :func:`_coords_torch`).
    atom_mask:
        Boolean mask ``(N,)`` or ``(1, N)``; same convention as :func:`radius_of_gyration`.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)``, dtype and device from the snapshot.
    """
    x = _coords_torch(snapshot)[0]
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        return x[m.bool()]
    return x


def end_to_end_distance(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    """Distance between the first and last points after masking (Å).

    Parameters
    ----------
    snapshot:
        Coordinate source; see :func:`_coords_torch`.
    atom_mask:
        Optional mask ``(N,)`` or ``(1, N)``; same as :func:`radius_of_gyration`.

    Returns
    -------
    float
        End-to-end distance, or ``0.0`` if fewer than two points remain.
    """
    pts = _masked_points(snapshot, atom_mask)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        d = torch.linalg.norm(pts[0] - pts[n - 1])
        return float(d.detach().cpu())


def ca_contact_count(
    snapshot,
    *,
    seq_sep: int = 6,
    dist_threshold: float = 8.0,
    atom_mask: torch.Tensor | None = None,
) -> float:
    """Number of long-range contacts (unordered pairs).

    A pair ``(i, j)``, ``i < j``, counts if ``|i - j| >= seq_sep`` and
    ``||x_i - x_j|| < dist_threshold``. Same separation rule as
    :func:`contact_order`; this function returns a count, not a fraction.

    Parameters
    ----------
    snapshot:
        Coordinate source; see :func:`_coords_torch`.
    seq_sep:
        Minimum sequence index separation for an eligible pair.
    dist_threshold:
        Contact cutoff in Å.
    atom_mask:
        Optional mask ``(N,)`` or ``(1, N)``.
    """
    x = _masked_points(snapshot, atom_mask)
    n = x.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]
        idx = torch.arange(n, device=x.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        upper = torch.triu(
            (sep >= seq_sep) & (dists < dist_threshold), diagonal=1
        )
        return float(upper.sum().item())


def _gyration_eigenvalues_descending(pts: torch.Tensor) -> torch.Tensor:
    r"""Eigenvalues of the unweighted centroid gyration tensor, largest first (Å²).

    Uses :math:`\mathbf{S} = n^{-1} \sum_i (\mathbf{r}_i - \mathbf{r}_{\mathrm{cm}})
    (\mathbf{r}_i - \mathbf{r}_{\mathrm{cm}})^{\mathsf T}` with eigenvalues
    :math:`\lambda_1 \geq \lambda_2 \geq \lambda_3`.
    """
    n = pts.shape[0]
    if n < 1:
        return torch.zeros(3, device=pts.device, dtype=pts.dtype)
    r = pts - pts.mean(dim=0, keepdim=True)
    s = (r.T @ r) / float(n)
    w = torch.linalg.eigvalsh(s)
    return w.flip(0)


def shape_kappa2(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    r"""Relative shape anisotropy :math:`\kappa^2` from gyration eigenvalues (dimensionless).

    .. math::

        \kappa^2 = 1 - \frac{3(\lambda_1\lambda_2 + \lambda_2\lambda_3 +
        \lambda_3\lambda_1)}{(\lambda_1 + \lambda_2 + \lambda_3)^2}

    Isotropic cloud :math:`\to \kappa^2 = 0`; straight filament :math:`\to 1`.
    The result is clamped to :math:`[0, 1]`. Eigenvalues are from
    :func:`_gyration_eigenvalues_descending`.

    Parameters
    ----------
    snapshot:
        Coordinate source; see :func:`_coords_torch`.
    atom_mask:
        Optional mask ``(N,)`` or ``(1, N)``.

    Returns
    -------
    float
        ``0.0`` if fewer than two points remain or if the eigenvalue trace is
        negligible (numerical degeneracy).
    """
    pts = _masked_points(snapshot, atom_mask)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        lam = _gyration_eigenvalues_descending(pts)
        t = lam.sum()
        if float(t.item()) < 1e-12:
            return 0.0
        i2 = lam[0] * lam[1] + lam[1] * lam[2] + lam[2] * lam[0]
        k2 = 1.0 - 3.0 * (i2 / (t * t))
        return float(torch.clamp(k2, 0.0, 1.0).item())


def shape_acylindricity(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    r"""Triaxial acylindricity :math:`\lambda_2 - \lambda_3` in Å².

    :math:`\lambda_i` are descending eigenvalues of the unweighted centroid
    gyration tensor (see :func:`_gyration_eigenvalues_descending`). Vanishes for
    axisymmetric objects (e.g. ideal cylinder symmetry).

    Parameters
    ----------
    snapshot:
        Coordinate source; see :func:`_coords_torch`.
    atom_mask:
        Optional mask ``(N,)`` or ``(1, N)``.

    Returns
    -------
    float
        ``0.0`` if fewer than two points remain after masking.
    """
    pts = _masked_points(snapshot, atom_mask)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        lam = _gyration_eigenvalues_descending(pts)
        return float((lam[1] - lam[2]).detach().cpu())


def rmsd_to_reference(
    snapshot,
    reference: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> float:
    """RMSD (no alignment) between snapshot coords and ``reference`` (M, 3)."""
    x = _coords_torch(snapshot)[0]
    ref = reference.to(device=x.device, dtype=x.dtype)
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        sel = m.bool()
        d = (x[sel] - ref[sel]).pow(2).sum(dim=-1).mean()
    else:
        d = (x - ref).pow(2).sum(dim=-1).mean()
    return float(torch.sqrt(d + 1e-12).detach().cpu())


def kabsch_rmsd_aligned(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> float:
    """Kabsch-superposed RMSD between two coordinate sets (N, 3) in any consistent unit.

    Optimally superimposes ``coords_a`` onto ``coords_b`` via a proper rotation
    (det R = +1) and then computes the root-mean-square deviation.

    Algorithm (Kabsch 1976):
      1. Centre both sets at the origin.
      2. Compute the cross-covariance matrix H = P^T Q.
      3. SVD: H = U S V^T.
      4. Correct for reflections: d = sign(det(V U^T)).
      5. Optimal rotation: R = V diag(1, 1, d) U^T.
      6. RMSD = sqrt(mean ||R P_i - Q_i||^2).

    Args:
        coords_a: (N, 3) array — coordinates to rotate (e.g. pre-minimization Cα).
        coords_b: (N, 3) array — reference coordinates (e.g. post-minimization Cα).

    Returns:
        RMSD in the same units as the input coordinates.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    if coords_a.shape != coords_b.shape or coords_a.ndim != 2 or coords_a.shape[1] != 3:
        raise ValueError(
            f"Both arrays must have shape (N, 3); got {coords_a.shape} and {coords_b.shape}"
        )
    P = coords_a - coords_a.mean(axis=0)
    Q = coords_b - coords_b.mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(((P_rot - Q) ** 2).sum(axis=-1).mean()))


def diffusion_step_index_cv(snapshot) -> float:
    return float(getattr(snapshot, "step_index", 0))


def make_rg_cv(name: str = "Rg", atom_mask: torch.Tensor | None = None) -> FunctionCV:
    return FunctionCV(name, _rg_callable, atom_mask=atom_mask)


def _rg_callable(snap, atom_mask=None):
    return radius_of_gyration(snap, atom_mask)


def make_rmsd_cv(
    name: str,
    reference: np.ndarray | torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> FunctionCV:
    ref = torch.as_tensor(reference, dtype=torch.float32)
    return FunctionCV(name, _rmsd_callable, reference=ref, atom_mask=atom_mask)


def _rmsd_callable(snap, reference=None, atom_mask=None):
    return rmsd_to_reference(snap, reference, atom_mask)


def make_sigma_cv(name: str = "sigma") -> FunctionCV:
    return FunctionCV(
        name,
        lambda snap: float(snap.sigma) if getattr(snap, "sigma", None) is not None else 0.0,
    )


def make_plddt_proxy_cv(
    name: str = "pLDDT_proxy",
    predictor: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
) -> FunctionCV:
    """pLDDT from denoised coords; supply ``predictor`` for real Boltz confidence.

    Parameters
    ----------
    name:
        OPS CV name.
    predictor:
        Callable that takes ``(B, N, 3)`` coordinates and returns a scalar
        pLDDT value (0–100).

    Raises
    ------
    ValueError
        If ``predictor`` is ``None``.  A real predictor must be supplied via
        :func:`make_boltz_plddt_predictor`.

    Notes
    -----
    To wire up the real Boltz confidence head, use
    :func:`make_boltz_plddt_predictor` to create the predictor from a loaded
    ``Boltz2`` model instance before constructing this CV.
    """
    if predictor is None:
        raise ValueError(
            "make_plddt_proxy_cv requires a real predictor callable. "
            "Use make_boltz_plddt_predictor() to create one from a loaded Boltz2 model, "
            "or pass a custom function that maps (B, N, 3) coords -> scalar pLDDT."
        )

    def _pred(snap):
        x = _coords_torch(snap)
        with torch.no_grad():
            out = predictor(x)
        return float(out.detach().cpu().reshape(-1)[0])

    return FunctionCV(name, _pred)


def make_boltz_plddt_predictor(
    model: "Any",
    network_condition_kwargs: "dict[str, Any]",
) -> "Callable[[torch.Tensor], torch.Tensor]":
    """Create a pLDDT predictor callable from a loaded Boltz2 model.

    Extracts the confidence module (``model.confidence_module``) and returns a
    callable that runs a forward pass on denoised coordinates to obtain the
    mean per-residue pLDDT (0–100 scale).

    The confidence module follows Algorithm 31 of the Boltz-2 paper:
    it requires trunk single representations (s, s_inputs), pair
    representations (z), predicted coordinates (x_pred), features dict (feats),
    and predicted distogram logits — NOT raw coordinates alone.

    Parameters
    ----------
    model:
        A loaded ``Boltz2`` model instance (from ``Boltz2.load_from_checkpoint``).
        Must have a ``confidence_module`` attribute.
    network_condition_kwargs:
        The same kwargs dict used for the score network.  Must contain:
        ``s_inputs``, ``s``, ``z``, ``feats``, ``pred_distogram_logits``, and
        ``multiplicity``.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A function that takes ``(B, N, 3)`` coords and returns a scalar pLDDT
        float in [0, 100].

    Raises
    ------
    AttributeError
        If the model does not have a ``confidence_module``.
    KeyError
        If ``network_condition_kwargs`` is missing required keys for the
        confidence forward pass.
    """
    if not hasattr(model, "confidence_module"):
        raise AttributeError(
            "make_boltz_plddt_predictor: model has no 'confidence_module'. "
            "Ensure the Boltz2 checkpoint was loaded with confidence_prediction=True."
        )
    confidence_module = model.confidence_module

    required_keys = {"s_inputs", "s", "z", "feats", "pred_distogram_logits", "multiplicity"}
    missing = required_keys - set(network_condition_kwargs.keys())
    if missing:
        raise KeyError(
            f"make_boltz_plddt_predictor: network_condition_kwargs missing keys "
            f"required by the Boltz2 confidence module: {missing}. "
            f"These are produced by the trunk forward pass (Boltz2.forward)."
        )

    from boltz.model.layers.confidence_utils import compute_aggregated_metric  # noqa: PLC0415

    def _predictor(coords: torch.Tensor) -> torch.Tensor:
        """Run Boltz2 confidence module on denoised coords -> mean pLDDT."""
        with torch.no_grad():
            out = confidence_module(
                s_inputs=network_condition_kwargs["s_inputs"].detach(),
                s=network_condition_kwargs["s"].detach(),
                z=network_condition_kwargs["z"].detach(),
                x_pred=coords.detach(),
                feats=network_condition_kwargs["feats"],
                pred_distogram_logits=network_condition_kwargs["pred_distogram_logits"].detach(),
                multiplicity=network_condition_kwargs["multiplicity"],
            )

        plddt_logits = out.get("plddt", None)
        if plddt_logits is None:
            raise RuntimeError(
                "Boltz2 confidence module returned no 'plddt' key. "
                "Available keys: " + str(list(out.keys()))
            )
        plddt = compute_aggregated_metric(plddt_logits)
        return plddt.mean()

    return _predictor


# ---------------------------------------------------------------------------
# Diagnostic CVs: contact order, clash count, lDDT, Ramachandran outliers
# ---------------------------------------------------------------------------

def contact_order(
    snapshot,
    seq_sep: int = 6,
    dist_threshold: float = 8.0,
    atom_mask: torch.Tensor | None = None,
) -> float:
    """Fraction of Cα pairs with sequence separation >= ``seq_sep`` and distance < ``dist_threshold``.

    Parameters
    ----------
    snapshot:
        OPS snapshot (or any object with ``tensor_coords`` or ``coordinates``).
    seq_sep:
        Minimum sequence separation to consider a pair (default: 6).
    dist_threshold:
        Distance threshold in Ångström for counting a contact (default: 8.0).
    atom_mask:
        Optional boolean/float mask of shape (N,) or (1, N) selecting which atoms to use.

    Returns
    -------
    float
        Fraction of eligible pairs that are in contact. 0.0 when there are no eligible pairs.
    """
    x = _coords_torch(snapshot)[0]  # (N, 3)
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        x = x[m.bool()]
    n = x.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]  # (N, N)
        idx = torch.arange(n, device=x.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        eligible = sep >= seq_sep
        n_eligible = int(eligible.sum().item())
        if n_eligible == 0:
            return 0.0
        contacts = (dists < dist_threshold) & eligible
        return float(contacts.sum().item() / n_eligible)


def clash_count(
    snapshot,
    min_dist: float = 3.0,
    min_seq_sep: int = 3,
    atom_mask: torch.Tensor | None = None,
) -> int:
    """Number of Cα pairs with distance < ``min_dist`` and sequence separation >= ``min_seq_sep``.

    Bonded/near-bonded neighbours (|i-j| < ``min_seq_sep``) are excluded.

    Parameters
    ----------
    snapshot:
        OPS snapshot (or any object with ``tensor_coords`` or ``coordinates``).
    min_dist:
        Clash distance threshold in Ångström (default: 3.0).
    min_seq_sep:
        Minimum sequence separation below which pairs are not counted (default: 3).
    atom_mask:
        Optional boolean/float mask of shape (N,) or (1, N).

    Returns
    -------
    int
        Number of clashing pairs (each pair counted once).
    """
    x = _coords_torch(snapshot)[0]  # (N, 3)
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        x = x[m.bool()]
    n = x.shape[0]
    if n < 2:
        return 0
    with torch.no_grad():
        dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]
        idx = torch.arange(n, device=x.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        clashes = (dists < min_dist) & (sep >= min_seq_sep)
        # Each pair (i,j) and (j,i) both flagged; divide by 2
        return int(clashes.sum().item() // 2)


def lddt_to_reference(
    snapshot,
    reference: torch.Tensor,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    atom_mask: torch.Tensor | None = None,
) -> float:
    """Alignment-free local distance difference test (lDDT) to a reference structure.

    Mariani et al. (2013): for each pair of residues within ``inclusion_radius``
    in the reference, computes the fraction of distance differences |d_query - d_ref|
    below each threshold in ``thresholds``, then averages across pairs and thresholds.

    Parameters
    ----------
    snapshot:
        OPS snapshot.
    reference:
        (N, 3) reference coordinates (Ångström).
    inclusion_radius:
        Maximum pairwise distance in the reference to include a pair (default: 15.0 Å).
    thresholds:
        Distance-difference thresholds in Ångström (default: (0.5, 1.0, 2.0, 4.0)).
    atom_mask:
        Optional boolean/float mask of shape (N,) or (1, N).

    Returns
    -------
    float
        lDDT score in [0, 1]. Returns 0.0 when no eligible pairs exist.
    """
    x = _coords_torch(snapshot)[0]  # (N, 3)
    ref = reference.to(device=x.device, dtype=x.dtype)
    if ref.dim() == 3:
        ref = ref[0]
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        m_bool = m.bool()
        x = x[m_bool]
        ref = ref[m_bool]
    n = x.shape[0]
    if n < 2:
        return 0.0
    with torch.no_grad():
        d_query = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]
        d_ref = torch.cdist(ref.unsqueeze(0), ref.unsqueeze(0))[0]
        eye = torch.eye(n, device=x.device, dtype=torch.bool)
        pair_mask = (d_ref < inclusion_radius) & (~eye)
        if pair_mask.sum() == 0:
            return 0.0
        diff = (d_query - d_ref).abs()
        scores: list[torch.Tensor] = []
        for t in thresholds:
            scores.append((diff[pair_mask] < t).float().mean())
        return float(torch.stack(scores).mean().item())


def ramachandran_outlier_fraction(
    snapshot,
    backbone_indices: torch.Tensor | None = None,
) -> float:
    """Fraction of residues with (phi, psi) dihedral angles outside the allowed region.

    For each residue with defined phi and psi angles, classifies whether the
    (phi, psi) pair falls inside the simplified allowed Ramachandran region:

        Allowed regions (Lovell et al. 2003 approximation):
          - Alpha helix:     phi in [-160, -20],  psi in [-120, 50]
          - Beta sheet:      phi in [-160, -40],  psi in [90, 180] or [-180, -150]
          - Left-handed alpha: phi in [40, 100], psi in [20, 100]

        Any (phi, psi) not in any allowed region is classified as an outlier.

    Parameters
    ----------
    snapshot:
        OPS snapshot with backbone atoms in order (N, CA, C, N, CA, C, ...).
    backbone_indices:
        (n_res, 3) integer tensor mapping residue index to the indices of
        (N, CA, C) atoms in the coordinate array.  If None, returns 0.0.

    Returns
    -------
    float
        Fraction of residues that are Ramachandran outliers.  Returns 0.0 when
        fewer than 2 residues are available (phi/psi undefined).
    """
    if backbone_indices is None:
        return 0.0
    x = _coords_torch(snapshot)[0]  # (N_atoms, 3)
    n_res = backbone_indices.shape[0]
    if n_res < 2:
        return 0.0

    def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> float:
        """Dihedral angle a-b-c-d in degrees."""
        b1 = b - a
        b2 = c - b
        b3 = d - c
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        m1 = torch.linalg.cross(n1, b2 / (b2.norm() + 1e-10))
        x_ = (n1 * n2).sum()
        y_ = (m1 * n2).sum()
        angle_rad = torch.atan2(y_, x_)
        return float(angle_rad.item() * 180.0 / torch.pi)

    def _in_allowed(phi: float, psi: float) -> bool:
        """Return True if (phi, psi) is in the simplified allowed Ramachandran region."""
        # Core alpha-helix region
        if -160 <= phi <= -20 and -120 <= psi <= 50:
            return True
        # Beta-sheet / upper region
        if -160 <= phi <= -40 and (90 <= psi <= 180 or -180 <= psi <= -150):
            return True
        # Left-handed helix region
        if 40 <= phi <= 100 and 20 <= psi <= 100:
            return True
        return False

    outliers = 0
    n_classified = 0
    indices = backbone_indices.long()
    for i in range(1, n_res - 1):
        try:
            # phi: C(i-1) - N(i) - CA(i) - C(i)
            c_prev = x[indices[i - 1, 2]]
            n_i = x[indices[i, 0]]
            ca_i = x[indices[i, 1]]
            c_i = x[indices[i, 2]]
            # psi: N(i) - CA(i) - C(i) - N(i+1)
            n_next = x[indices[i + 1, 0]]

            phi = _dihedral(c_prev, n_i, ca_i, c_i)
            psi = _dihedral(n_i, ca_i, c_i, n_next)

            if not (math.isnan(phi) or math.isnan(psi)):
                n_classified += 1
                if not _in_allowed(phi, psi):
                    outliers += 1
        except Exception:
            continue

    if n_classified == 0:
        return 0.0
    return float(outliers / n_classified)


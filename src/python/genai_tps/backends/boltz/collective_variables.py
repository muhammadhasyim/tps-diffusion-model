"""GPU-friendly collective variables; OPS CVs wrap scalar functions on snapshots."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from openpathsampling.collectivevariable import FunctionCV

# Boltz chain_type_ids: {"PROTEIN": 0, "DNA": 1, "RNA": 2, "NONPOLYMER": 3}
_PROTEIN_MOL_TYPE: int = 0
_NONPOLYMER_MOL_TYPE: int = 3


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
    """Return ``(n, 3)`` coordinates after optional ``atom_mask`` (same convention as ``radius_of_gyration``)."""
    x = _coords_torch(snapshot)[0]
    if atom_mask is not None:
        m = atom_mask[0] if atom_mask.dim() > 1 else atom_mask
        return x[m.bool()]
    return x


def end_to_end_distance(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    """Euclidean distance between first and last atom in the masked list (Å).

    Intended for Cα traces: after masking, uses indices ``0`` and ``n-1``.
    Returns ``0.0`` when ``n < 2`` (stable for OPES).
    """
    pts = _masked_points(snapshot, atom_mask)
    n = int(pts.shape[0])
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
    """Count long-range pairs with distance below ``dist_threshold`` (not normalized).

    Same eligibility rule as :func:`contact_order` (``|i-j| >= seq_sep``) but returns
    the raw number of contacting pairs (each unordered pair once).
    """
    x = _masked_points(snapshot, atom_mask)
    n = int(x.shape[0])
    if n < 2:
        return 0.0
    with torch.no_grad():
        dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))[0]
        idx = torch.arange(n, device=x.device)
        sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        upper = torch.triu(
            (sep >= int(seq_sep)) & (dists < float(dist_threshold)), diagonal=1
        )
        return float(upper.sum().item())


def _gyration_eigenvalues_descending(pts: torch.Tensor) -> torch.Tensor:
    """Eigenvalues ``λ1 >= λ2 >= λ3`` of mass-weighted gyration tensor (Å²)."""
    n = int(pts.shape[0])
    if n < 1:
        return torch.zeros(3, device=pts.device, dtype=pts.dtype)
    r = pts - pts.mean(dim=0, keepdim=True)
    # S = (1/n) sum_i r_i r_i^T  ->  (3,3)
    s = (r.T @ r) / float(max(n, 1))
    w = torch.linalg.eigvalsh(s)  # ascending
    return w.flip(0)


def shape_kappa2(snapshot, atom_mask: torch.Tensor | None = None) -> float:
    r"""Relative shape anisotropy :math:`\kappa^2 \in [0,1]` from gyration eigenvalues.

    :math:`\kappa^2 = 1 - 3(\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_3\lambda_1)
    / (\lambda_1+\lambda_2+\lambda_3)^2`.

    Sphere: :math:`\kappa^2=0`; straight rod (two non-zero dims degenerate): :math:`\kappa^2=1`.
    """
    pts = _masked_points(snapshot, atom_mask)
    n = int(pts.shape[0])
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
    r"""Acylindricity :math:`\lambda_2 - \lambda_3` (Å²) from the gyration tensor eigenvalues."""
    pts = _masked_points(snapshot, atom_mask)
    n = int(pts.shape[0])
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
    predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> FunctionCV:
    """pLDDT from denoised coords; supply ``predictor`` for real Boltz confidence.

    Parameters
    ----------
    name:
        OPS CV name.
    predictor:
        Callable that takes ``(B, N, 3)`` coordinates and returns a scalar
        pLDDT value (0–100).  If ``None``, returns the stub value 50.0
        (used when the confidence head is not available).

    Notes
    -----
    To wire up the real Boltz confidence head, use
    :func:`make_boltz_plddt_predictor` to create the predictor from a loaded
    ``Boltz2`` model instance before constructing this CV.
    """

    def _pred(snap):
        x = _coords_torch(snap)
        if predictor is None:
            return 50.0 + 0.0 * float(x.detach().cpu().mean())
        with torch.no_grad():
            out = predictor(x)
        return float(out.detach().cpu().reshape(-1)[0])

    return FunctionCV(name, _pred)


def make_boltz_plddt_predictor(
    model: "Any",
    network_condition_kwargs: "dict[str, Any]",
) -> "Callable[[torch.Tensor], torch.Tensor]":
    """Create a pLDDT predictor callable from a loaded Boltz2 model.

    Extracts the confidence head (``model.confidence_head``) and returns a
    callable that runs a forward pass on denoised coordinates to obtain the
    mean per-residue pLDDT (0–100 scale).

    Parameters
    ----------
    model:
        A loaded ``Boltz2`` model instance (from ``Boltz2.load_from_checkpoint``).
    network_condition_kwargs:
        The same kwargs dict used for the score network (contains trunk features,
        feats, etc.).  The confidence head reuses the same conditioning tensors.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A function that takes ``(B, N, 3)`` coords and returns a scalar pLDDT
        float in [0, 100].  Returns 50.0 on error.
    """
    try:
        confidence_head = model.confidence_head
    except AttributeError:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "make_boltz_plddt_predictor: model has no confidence_head; "
            "returning stub predictor (50.0)"
        )

        def _stub(coords: torch.Tensor) -> torch.Tensor:
            return torch.tensor(50.0)

        return _stub

    def _predictor(coords: torch.Tensor) -> torch.Tensor:
        try:
            from boltz.model.modules.confidence_utils import (  # noqa: PLC0415
                compute_aggregated_metric,
            )
            with torch.no_grad():
                out = confidence_head(coords, **network_condition_kwargs)
            plddt_logits = out.get("plddt", None)
            if plddt_logits is not None:
                plddt = compute_aggregated_metric(plddt_logits)
                return plddt.mean()
            return torch.tensor(50.0)
        except Exception as exc:
            import logging  # noqa: PLC0415
            logging.getLogger(__name__).warning(
                "make_boltz_plddt_predictor._predictor: error %s; returning 50.0", exc
            )
            return torch.tensor(50.0)

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


# ---------------------------------------------------------------------------
# Protein-ligand pose quality CVs
# ---------------------------------------------------------------------------

def _kabsch_rotation(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Kabsch rotation matrix and centres for optimal superposition.

    Computes the proper rotation R (det R = +1) and the centres of mass
    ``c_mob`` and ``c_ref`` such that::

        mobile_aligned = (mobile - c_mob) @ R.T + c_ref

    minimises the RMSD between ``mobile_aligned`` and ``reference``.

    Parameters
    ----------
    mobile:
        (N, 3) array — coordinates to rotate.
    reference:
        (N, 3) array — target/reference coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(R, c_mob, c_ref)`` where R is (3, 3), c_mob and c_ref are (3,).
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    c_mob = mobile.mean(axis=0)
    c_ref = reference.mean(axis=0)
    P = mobile - c_mob
    Q = reference - c_ref
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    return R, c_mob, c_ref


class PoseCVIndexer:
    """Pre-computes atom index arrays from a Boltz StructureV2 topology.

    Used by the four protein-ligand pose quality CVs to look up which atoms
    are protein, ligand, pocket, and potential H-bond donors/acceptors without
    re-parsing the topology at every snapshot.

    Scientific conventions
    ----------------------
    - *Ligand*: all atoms in chains with ``mol_type == NONPOLYMER (3)``.
    - *Pocket*: protein Cα atoms (name "CA") within ``pocket_radius`` of the
      initial ligand centre-of-mass; **also** all protein heavy atoms within
      the same radius (used for contact counting).
    - *H-bond proxies*: atoms whose PDB name (stripped) starts with "N" or "O",
      following the standard that nitrogen and oxygen are the biologically
      relevant donors/acceptors (MDAnalysis / GROMACS convention).

    Parameters
    ----------
    structure:
        Boltz ``StructureV2`` topology object (``load_topo`` return value).
    ref_coords:
        (N_atoms, 3) Ångström coordinates of the first TPS snapshot — used as
        the RMSD reference pose and to define the binding pocket.
    pocket_radius:
        Protein heavy atoms and Cα atoms within this radius (Å) of the initial
        ligand COM form the pocket definition (default 6.0 Å).
    """

    def __init__(
        self,
        structure: "Any",
        ref_coords: np.ndarray,
        pocket_radius: float = 6.0,
    ) -> None:
        ref = np.asarray(ref_coords, dtype=np.float64)
        chains = structure.chains
        atoms = structure.atoms

        protein_idx_list: list[int] = []
        ligand_idx_list: list[int] = []

        for chain in chains:
            start = int(chain["atom_idx"])
            end = start + int(chain["atom_num"])
            mol_type = int(chain["mol_type"])
            if mol_type == _PROTEIN_MOL_TYPE:
                protein_idx_list.extend(range(start, end))
            elif mol_type == _NONPOLYMER_MOL_TYPE:
                ligand_idx_list.extend(range(start, end))

        self.protein_idx: np.ndarray = np.array(protein_idx_list, dtype=np.int64)
        self.ligand_idx: np.ndarray = np.array(ligand_idx_list, dtype=np.int64)

        # Identify protein Cα atoms (name == "CA" after stripping whitespace)
        if len(self.protein_idx) > 0:
            prot_names = np.array(
                [str(atoms[i]["name"]).strip() for i in self.protein_idx]
            )
            ca_mask = prot_names == "CA"
            self.protein_ca_idx: np.ndarray = self.protein_idx[ca_mask]
        else:
            self.protein_ca_idx = np.array([], dtype=np.int64)

        # Pocket: protein atoms within pocket_radius of initial ligand COM
        if len(self.ligand_idx) > 0:
            ligand_com = ref[self.ligand_idx].mean(axis=0)
        else:
            ligand_com = np.zeros(3)
        self._ligand_com_ref: np.ndarray = ligand_com

        if len(self.protein_ca_idx) > 0:
            ca_coords = ref[self.protein_ca_idx]
            ca_dists = np.linalg.norm(ca_coords - ligand_com, axis=1)
            self.pocket_ca_idx: np.ndarray = self.protein_ca_idx[ca_dists <= pocket_radius]
        else:
            self.pocket_ca_idx = np.array([], dtype=np.int64)

        if len(self.protein_idx) > 0:
            prot_coords = ref[self.protein_idx]
            prot_dists = np.linalg.norm(prot_coords - ligand_com, axis=1)
            self.pocket_heavy_idx: np.ndarray = self.protein_idx[prot_dists <= pocket_radius]
        else:
            self.pocket_heavy_idx = np.array([], dtype=np.int64)

        # H-bond proxy: N/O atoms on ligand and pocket protein
        def _no_mask(idx_arr: np.ndarray) -> np.ndarray:
            names = [str(atoms[i]["name"]).strip() for i in idx_arr]
            return np.array(
                [n.startswith("N") or n.startswith("O") for n in names], dtype=bool
            )

        if len(self.ligand_idx) > 0:
            self.ligand_no_idx: np.ndarray = self.ligand_idx[_no_mask(self.ligand_idx)]
        else:
            self.ligand_no_idx = np.array([], dtype=np.int64)

        if len(self.pocket_heavy_idx) > 0:
            self.pocket_no_idx: np.ndarray = self.pocket_heavy_idx[
                _no_mask(self.pocket_heavy_idx)
            ]
        else:
            self.pocket_no_idx = np.array([], dtype=np.int64)

        # Reference coordinates stored for RMSD calculation
        if len(self.protein_ca_idx) > 0:
            self.ref_protein_ca: np.ndarray = ref[self.protein_ca_idx].copy()
        else:
            self.ref_protein_ca = np.empty((0, 3), dtype=np.float64)

        if len(self.ligand_idx) > 0:
            self.ref_ligand: np.ndarray = ref[self.ligand_idx].copy()
        else:
            self.ref_ligand = np.empty((0, 3), dtype=np.float64)


# --- Ligand pose RMSD (BPMD/PLUMED TYPE=OPTIMAL convention) ----------------

def ligand_pose_rmsd(snapshot, indexer: PoseCVIndexer) -> float:
    """RMSD of ligand heavy atoms from initial pose, after Kabsch-aligning protein Cα.

    Implements the PLUMED ``RMSD TYPE=OPTIMAL`` convention used in Binding Pose
    Metadynamics (BPMD): protein Cα atoms serve as the alignment frame
    (``occupancy=1/beta=0``); ligand heavy atoms are the RMSD group
    (``occupancy=0/beta=1``).  The reference is the **first TPS snapshot** —
    the raw Boltz docked pose before any minimization.

    Parameters
    ----------
    snapshot:
        OPS snapshot.
    indexer:
        Pre-built :class:`PoseCVIndexer` for this system.

    Returns
    -------
    float
        Ligand RMSD in Ångström.  Returns 0.0 when no ligand atoms are present.
    """
    if len(indexer.ligand_idx) == 0:
        return 0.0

    x = _coords_torch(snapshot)[0].detach().cpu().numpy().astype(np.float64)
    lig_cur = x[indexer.ligand_idx]  # (L, 3)

    if len(indexer.protein_ca_idx) >= 3:
        ca_cur = x[indexer.protein_ca_idx]  # (M, 3)
        R, c_mob, c_ref = _kabsch_rotation(ca_cur, indexer.ref_protein_ca)
        lig_aligned = (lig_cur - c_mob) @ R.T + c_ref
    else:
        # No protein Cα available: skip alignment (raw RMSD against reference)
        lig_aligned = lig_cur

    diff_sq = ((lig_aligned - indexer.ref_ligand) ** 2).sum(axis=-1)
    return float(np.sqrt(diff_sq.mean() + 1e-24))


def make_ligand_pose_rmsd_cv(name: str, indexer: PoseCVIndexer) -> FunctionCV:
    """Create an OPS ``FunctionCV`` for :func:`ligand_pose_rmsd`."""
    return FunctionCV(name, _ligand_pose_rmsd_callable, indexer=indexer)


def _ligand_pose_rmsd_callable(snap, indexer: PoseCVIndexer) -> float:
    return ligand_pose_rmsd(snap, indexer)


# --- Ligand-pocket distance (COM-COM, funnel metadynamics style) ------------

def ligand_pocket_distance(snapshot, indexer: PoseCVIndexer) -> float:
    """Euclidean distance between the ligand COM and the pocket Cα COM.

    The pocket is defined at construction time as the protein Cα atoms within
    ``pocket_radius`` of the initial ligand centre-of-mass.  Both COMs are
    recomputed from the current snapshot coordinates at each call, so the
    distance tracks the instantaneous ligand position relative to the binding
    site without requiring explicit alignment.  This is conceptually analogous
    to the ``fps.lp`` projection in PLUMED's ``FUNNEL_PS`` colvar.

    Parameters
    ----------
    snapshot:
        OPS snapshot.
    indexer:
        Pre-built :class:`PoseCVIndexer` for this system.

    Returns
    -------
    float
        Distance in Ångström.  Returns 0.0 when the pocket is empty.
    """
    if len(indexer.ligand_idx) == 0 or len(indexer.pocket_ca_idx) == 0:
        return 0.0

    x = _coords_torch(snapshot)[0].detach().cpu().numpy().astype(np.float64)
    lig_com = x[indexer.ligand_idx].mean(axis=0)
    pocket_com = x[indexer.pocket_ca_idx].mean(axis=0)
    return float(np.linalg.norm(lig_com - pocket_com))


def make_ligand_pocket_distance_cv(name: str, indexer: PoseCVIndexer) -> FunctionCV:
    """Create an OPS ``FunctionCV`` for :func:`ligand_pocket_distance`."""
    return FunctionCV(name, _ligand_pocket_distance_callable, indexer=indexer)


def _ligand_pocket_distance_callable(snap, indexer: PoseCVIndexer) -> float:
    return ligand_pocket_distance(snap, indexer)


# --- Protein-ligand contact count (PLUMED COORDINATION style) ---------------

def protein_ligand_contacts(
    snapshot,
    indexer: PoseCVIndexer,
    r0: float = 3.5,
) -> float:
    """Sum of rational switching functions over pocket-protein × ligand heavy atom pairs.

    Implements the canonical PLUMED ``COORDINATION`` definition::

        s_ij = (1 − (r_ij/r0)^6) / (1 − (r_ij/r0)^12)

    with ``n=6, m=12`` (rational function, continuous derivatives).  At
    ``r = r0`` the switching function equals 0.5; it approaches 1 for
    ``r << r0`` and 0 for ``r >> r0``.  The default ``r0 = 3.5 Å`` covers
    typical heavy-atom contact distances for protein-ligand complexes.

    Parameters
    ----------
    snapshot:
        OPS snapshot.
    indexer:
        Pre-built :class:`PoseCVIndexer` for this system.
    r0:
        Switching-function half-maximum distance in Ångström (default 3.5).

    Returns
    -------
    float
        Sum of switching function values (continuous contact count).
    """
    if len(indexer.ligand_idx) == 0 or len(indexer.pocket_heavy_idx) == 0:
        return 0.0

    x = _coords_torch(snapshot)[0]
    prot_coords = x[torch.as_tensor(indexer.pocket_heavy_idx, dtype=torch.long, device=x.device)]
    lig_coords = x[torch.as_tensor(indexer.ligand_idx, dtype=torch.long, device=x.device)]

    with torch.no_grad():
        # (P, L) pairwise distance matrix
        dists = torch.cdist(
            prot_coords.unsqueeze(0), lig_coords.unsqueeze(0)
        )[0]  # (P, L)
        ratio = dists / r0
        r6 = ratio.pow(6)
        r12 = ratio.pow(12)
        num = 1.0 - r6
        den = 1.0 - r12
        # At r = r0, num = den = 0; L'Hôpital gives limit = n/m = 6/12 = 0.5.
        # Detect this degenerate case and substitute the analytic limit.
        degenerate = den.abs() < 1e-9
        s = torch.where(degenerate, torch.full_like(num, 0.5), num / (den + 1e-30))
        # Clamp to [0, 1] for numerical safety at r << r0
        s = s.clamp(0.0, 1.0)
        return float(s.sum().item())


def make_protein_ligand_contacts_cv(
    name: str,
    indexer: PoseCVIndexer,
    r0: float = 3.5,
) -> FunctionCV:
    """Create an OPS ``FunctionCV`` for :func:`protein_ligand_contacts`."""
    return FunctionCV(
        name, _protein_ligand_contacts_callable, indexer=indexer, r0=r0
    )


def _protein_ligand_contacts_callable(
    snap, indexer: PoseCVIndexer, r0: float = 3.5
) -> float:
    return protein_ligand_contacts(snap, indexer, r0=r0)


# --- H-bond count (distance-only proxy, no explicit H) ---------------------

def protein_ligand_hbond_count(
    snapshot,
    indexer: PoseCVIndexer,
    cutoff: float = 3.5,
) -> float:
    """Number of putative protein-ligand H-bonds detected by heavy-atom proximity.

    Standard geometric H-bond criteria require the donor-acceptor distance
    r_DA < 3.5 Å and the D-H···A angle > 150° (MDAnalysis default: 3.0 Å +
    angle; GROMACS: 3.5 Å).  Because Boltz-2 outputs contain no explicit
    hydrogen atoms, this CV uses a *distance-only* proxy: it counts N/O···N/O
    pairs (one atom from the pocket protein, one from the ligand) with
    heavy-atom distance < ``cutoff``.  Using 3.5 Å accounts for the slightly
    longer donor-acceptor distance that arises when the hydrogen position is
    not resolved.

    Parameters
    ----------
    snapshot:
        OPS snapshot.
    indexer:
        Pre-built :class:`PoseCVIndexer` for this system.
    cutoff:
        Heavy-atom distance threshold in Ångström (default 3.5).

    Returns
    -------
    float
        Integer count of proximal N/O···N/O pairs, returned as float for
        compatibility with OPS CV infrastructure.
    """
    if len(indexer.ligand_no_idx) == 0 or len(indexer.pocket_no_idx) == 0:
        return 0.0

    x = _coords_torch(snapshot)[0]
    prot_no = x[torch.as_tensor(indexer.pocket_no_idx, dtype=torch.long, device=x.device)]
    lig_no = x[torch.as_tensor(indexer.ligand_no_idx, dtype=torch.long, device=x.device)]

    with torch.no_grad():
        dists = torch.cdist(
            prot_no.unsqueeze(0), lig_no.unsqueeze(0)
        )[0]  # (P_no, L_no)
        return float((dists < cutoff).sum().item())


def make_protein_ligand_hbond_count_cv(
    name: str,
    indexer: PoseCVIndexer,
    cutoff: float = 3.5,
) -> FunctionCV:
    """Create an OPS ``FunctionCV`` for :func:`protein_ligand_hbond_count`."""
    return FunctionCV(
        name, _protein_ligand_hbond_count_callable, indexer=indexer, cutoff=cutoff
    )


def _protein_ligand_hbond_count_callable(
    snap, indexer: PoseCVIndexer, cutoff: float = 3.5
) -> float:
    return protein_ligand_hbond_count(snap, indexer, cutoff=cutoff)


def make_training_sucos_pocket_qcov_traj_fn(
    scorer: Any,
    structure: Any,
    n_struct: int,
):
    """Build ``f(traj) -> float`` for Škrinjar-style training similarity on trajectories.

    Wraps :class:`genai_tps.analysis.skrinjar_similarity.IncrementalSkrinjarScorer`
    with Boltz last-frame PDB export.  The CLI entry point is
    ``scripts/run_opes_tps.py --bias-cv training_sucos_pocket_qcov`` (coordinate
    hash caching lives on *scorer*).

    Parameters
    ----------
    scorer:
        Configured :class:`~genai_tps.analysis.skrinjar_similarity.IncrementalSkrinjarScorer`.
    structure:
        Boltz :class:`~boltz.data.types.StructureV2` from ``load_topo``.
    n_struct:
        Number of heavy atoms in *structure* (first *n_struct* rows are scored).

    Returns
    -------
    Callable
        ``Callable[[Trajectory], float]`` suitable for OPES bias hooks.
    """
    from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy  # noqa: PLC0415

    n = int(n_struct)

    def _fn(traj) -> float:
        snap = traj[-1]
        coords = snapshot_frame_numpy_copy(snap)[:n]
        return float(scorer.score_coords(coords, structure, n))

    return _fn

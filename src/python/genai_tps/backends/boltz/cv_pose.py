"""Protein–ligand pose collective variables and :class:`PoseCVIndexer`.

Kabsch alignment, pocket definition, and ligand-centric CVs used by TPS/OPES.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from openpathsampling.collectivevariable import FunctionCV

from genai_tps.backends.boltz.constants import NONPOLYMER_MOL_TYPE, PROTEIN_MOL_TYPE
from genai_tps.backends.boltz.cv_geometric import _coords_torch

_PROTEIN_MOL_TYPE: int = PROTEIN_MOL_TYPE
_NONPOLYMER_MOL_TYPE: int = NONPOLYMER_MOL_TYPE

__all__ = [
    "PoseCVIndexer",
    "_binding_site_touching_protein_atoms",
    "ligand_pose_rmsd",
    "ligand_pocket_distance",
    "make_ligand_pose_rmsd_cv",
    "make_ligand_pocket_distance_cv",
    "make_protein_ligand_contacts_cv",
    "make_protein_ligand_hbond_count_cv",
    "protein_ligand_contacts",
    "protein_ligand_hbond_count",
]


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
    if mobile.shape[0] < 3 or reference.shape[0] < 3:
        raise ValueError("Kabsch alignment needs at least three points.")
    c_mob = mobile.mean(axis=0)
    c_ref = reference.mean(axis=0)
    P = mobile - c_mob
    Q = reference - c_ref
    H = P.T @ Q
    if not np.isfinite(H).all():
        raise ValueError("Kabsch: non-finite covariance (check MD coordinates).")
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        U, _, Vt = np.linalg.svd(H + 1e-12 * np.eye(3, dtype=np.float64))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    return R, c_mob, c_ref


def _binding_site_touching_protein_atoms(
    ref: np.ndarray,
    ligand_idx: np.ndarray,
    protein_idx: np.ndarray,
    *,
    cutoff_angstrom: float,
) -> np.ndarray:
    """Return sorted global indices of protein atoms within *cutoff* of any ligand atom.

    The binding envelope is residue-centric in downstream use: residues that touch
    the ligand (minimum protein–ligand atom distance ``≤ cutoff``) define the lining
    of the pocket. This matches docking-school definitions where “active/passive”
    residues satisfy **heavy-atom proximity** between receptor and ligand (see e.g.
    HADDOCK / Bonvin‑lab docking tutorials citing **~3.5 Å** cutoffs); *cutoff* here
    is left as an explicit Å parameter so MD workflows may use coordination-shell
    choices such as ~6 Å (Colvars coordination-style cutoffs).

    Raises
    ------
    ValueError
        When *cutoff_angstrom* is not finite or not positive.

    References
    ----------
    Bonvin *et al.* biomolecular docking tutorials defining binding-site residues via
    **any atom–atom distance** thresholds; PDB‑LIG / pocket centroid literature using
    **5–8 Å** merges for clustered ligand geometries (Nature Communications‑style pocket
    sets cited in PocketVec / PDB‑LIG analyses).
    """
    r = float(cutoff_angstrom)
    if not math.isfinite(r) or r <= 0:
        raise ValueError("Binding-site cutoff must be a finite positive Å length.")

    lg = np.asarray(ligand_idx, dtype=np.int64)
    pr = np.asarray(protein_idx, dtype=np.int64)
    if lg.size == 0 or pr.size == 0:
        return np.array([], dtype=np.int64)

    lig = np.asarray(ref[lg], dtype=np.float64)
    pcs = np.asarray(ref[pr], dtype=np.float64)
    cutoff2 = r * r

    diff = pcs[:, np.newaxis, :] - lig[np.newaxis, :, :]
    d2_ij = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
    d2_closest = np.min(d2_ij, axis=1)
    hit = d2_closest <= cutoff2
    return np.sort(pr[np.nonzero(hit)[0]]).astype(np.int64)


class PoseCVIndexer:
    """Pre-computes atom index arrays from a Boltz StructureV2 topology.

    Used by the four protein-ligand pose quality CVs to look up which atoms
    are protein, ligand, pocket, and potential H-bond donors/acceptors without
    re-parsing the topology at every snapshot.

    Scientific conventions
    ----------------------
    - *Ligand*: all atoms in chains with ``mol_type == NONPOLYMER (3)``.
    - *Pocket*: protein residues that **touch** the ligand at the reference
      coordinates: any protein atom lies within ``pocket_radius`` Å of **any**
      ligand atom (minimum distance test).  This follows receptor–ligand **active
      residue** shells from docking tutorials (see class doc ref: Bonvin lab /
      HADDOCK distance-to-ligand rules) and coordination-style cutoffs used in
      Colvars / pocket-centroid literature.  All protein atoms in each touching
      residue are recorded as ``pocket_heavy_idx``; ``pocket_ca_idx`` lists the
      corresponding Cα atoms for alignment and COM-based pocket distance CVs.
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
        Maximum protein–ligand atom–atom distance (Å) for a residue to count as
        binding-site lining at the reference pose (default 6.0 Å).
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

        # Ligand COM (diagnostics / legacy consumers)
        if len(self.ligand_idx) > 0:
            ligand_com = ref[self.ligand_idx].mean(axis=0)
        else:
            ligand_com = np.zeros(3)
        self._ligand_com_ref: np.ndarray = ligand_com

        # Map each global atom index → protein residue id (-1 if not protein)
        atom_residue = np.full(ref.shape[0], -1, dtype=np.int32)
        residues = structure.residues
        for chain in chains:
            if int(chain["mol_type"]) != _PROTEIN_MOL_TYPE:
                continue
            r0 = int(chain["res_idx"])
            rn = int(chain["res_num"])
            for rid in range(r0, r0 + rn):
                res = residues[rid]
                a0 = int(res["atom_idx"])
                ana = int(res["atom_num"])
                atom_residue[a0 : a0 + ana] = rid

        touching = _binding_site_touching_protein_atoms(
            ref,
            self.ligand_idx,
            self.protein_idx,
            cutoff_angstrom=float(pocket_radius),
        )
        pocket_res_ids = sorted(
            {int(atom_residue[i]) for i in touching.tolist() if atom_residue[i] >= 0}
        )

        pocket_ca_list: list[int] = []
        pocket_heavy_list: list[int] = []
        for rid in pocket_res_ids:
            res = residues[rid]
            a0 = int(res["atom_idx"])
            ana = int(res["atom_num"])
            for kk in range(ana):
                ai = a0 + kk
                pocket_heavy_list.append(ai)
                if str(atoms[ai]["name"]).strip() == "CA":
                    pocket_ca_list.append(ai)

        self.pocket_ca_idx = np.asarray(pocket_ca_list, dtype=np.int64)
        self.pocket_heavy_idx = np.asarray(pocket_heavy_list, dtype=np.int64)

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

        if len(self.pocket_ca_idx) > 0:
            self.ref_pocket_ca: np.ndarray = ref[self.pocket_ca_idx].copy()
        else:
            self.ref_pocket_ca = np.empty((0, 3), dtype=np.float64)

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
    if not np.isfinite(x).all():
        return 1e3
    lig_cur = x[indexer.ligand_idx]  # (L, 3)

    if len(indexer.pocket_ca_idx) >= 3:
        ca_cur = x[indexer.pocket_ca_idx]
        ca_ref = indexer.ref_pocket_ca
    elif len(indexer.protein_ca_idx) >= 3:
        ca_cur = x[indexer.protein_ca_idx]
        ca_ref = indexer.ref_protein_ca
    else:
        ca_cur = None
        ca_ref = None

    if ca_cur is not None and len(ca_cur) >= 3 and np.isfinite(ca_cur).all():
        try:
            R, c_mob, c_ref = _kabsch_rotation(ca_cur, ca_ref)
            lig_aligned = (lig_cur - c_mob) @ R.T + c_ref
        except (np.linalg.LinAlgError, ValueError):
            lig_aligned = lig_cur
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

    The pocket residue set is fixed at construction from **protein residues that
    touch the ligand** at the reference pose (minimum protein–ligand atom distance
    ``≤ pocket_radius``, then full residue shells). Pocket COM uses the pocket Cα
    subset only. Both COMs are recomputed from the current snapshot coordinates
    at each call, so the distance tracks the instantaneous ligand position relative
    to the binding site without requiring explicit alignment.  This is conceptually
    analogous to the ``fps.lp`` projection in PLUMED's ``FUNNEL_PS`` colvar.

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
        Returns ``1e3`` when coordinates are non-finite (same sentinel as
        :func:`ligand_pose_rmsd`).
    """
    if len(indexer.ligand_idx) == 0 or len(indexer.pocket_ca_idx) == 0:
        return 0.0

    x = _coords_torch(snapshot)[0].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(x).all():
        # Match ``ligand_pose_rmsd`` sentinel when coordinates are unusable.
        return 1e3

    lig_com = x[indexer.ligand_idx].mean(axis=0)
    pocket_com = x[indexer.pocket_ca_idx].mean(axis=0)
    if not (np.isfinite(lig_com).all() and np.isfinite(pocket_com).all()):
        return 1e3
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

"""GPU-friendly collective variables; OPS CVs wrap scalar functions on snapshots.

Implementation is split into :mod:`genai_tps.backends.boltz.cv_geometric` and
:mod:`genai_tps.backends.boltz.cv_pose`; this module re-exports the public surface
for backward compatibility.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

_cv_geometric = importlib.import_module("genai_tps.backends.boltz.cv_geometric")
_cv_pose = importlib.import_module("genai_tps.backends.boltz.cv_pose")

globals().update({k: getattr(_cv_geometric, k) for k in _cv_geometric.__all__})
globals().update({k: getattr(_cv_pose, k) for k in _cv_pose.__all__})

__all__ = list(_cv_geometric.__all__) + list(_cv_pose.__all__)


def make_training_sucos_pocket_qcov_traj_fn(
    scorer: Any,
    structure: Any,
    n_struct: int,
):
    """Build ``f(traj) -> float`` for Škrinjar-style training similarity on trajectories.

    Wraps :class:`genai_tps.evaluation.skrinjar_similarity.IncrementalSkrinjarScorer`
    with Boltz last-frame PDB export.  The CLI entry point is
    ``scripts/run_opes_tps.py --bias-cv training_sucos_pocket_qcov`` (coordinate
    hash caching lives on *scorer*).

    Parameters
    ----------
    scorer:
        Configured :class:`~genai_tps.evaluation.skrinjar_similarity.IncrementalSkrinjarScorer`.
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


def _normalize_atom_mask_for_compute_cvs(
    atom_mask_np: np.ndarray | None,
    *,
    n_atoms_coords: int,
    n_struct_topo: int,
) -> np.ndarray | None:
    """Return mask of shape ``(1, n_atoms_coords)`` or ``None``.

    RMSD / Rg helpers expect a batch row mask matching the coordinate width *M*.
    A legacy mistake used ``mask_np[:1]`` on a 1D array of length *n_s*,
    producing shape ``(1,)`` instead of ``(1, M)``.
    """
    if atom_mask_np is None:
        return None
    am = np.asarray(atom_mask_np, dtype=np.float32)
    M = int(n_atoms_coords)
    n_s = int(n_struct_topo)
    if am.ndim == 1:
        if am.shape[0] == M:
            return am.reshape(1, M)
        if am.shape[0] == n_s and M >= n_s:
            row = np.zeros((1, M), dtype=np.float32)
            row[0, :n_s] = am
            return row
        raise ValueError(
            "compute_cvs atom_mask_np 1-D: expected length "
            f"n_atoms={M} or n_struct={n_s}, got {am.shape[0]}."
        )
    if am.ndim == 2:
        if am.shape[1] != M:
            raise ValueError(
                "compute_cvs atom_mask_np 2-D: expected width "
                f"n_atoms={M}, got {am.shape[1]}."
            )
        return np.asarray(am[:1], dtype=np.float32)
    raise ValueError(
        f"compute_cvs atom_mask_np must be 1-D or 2-D, got ndim={am.ndim}."
    )


def compute_cvs(
    structures: np.ndarray,
    reference_coords: np.ndarray,
    atom_mask_np: np.ndarray | None,
    topo_npz: str | Path,
    *,
    pocket_radius: float = 6.0,
) -> dict[str, np.ndarray]:
    """Compute geometry and pose CVs using Boltz topology from *topo_npz*.

    Parameters
    ----------
    structures
        Array of shape ``(n_samples, n_atoms, 3)``.
    reference_coords
        Reference structure for RMSD, shape ``(n_atoms, 3)`` or larger (padded).
    atom_mask_np
        Optional per-atom mask for protein RMSD / Rg: either 1-D ``(n_atoms,)``
        aligned with *structures*' second axis, 1-D ``(n_struct,)`` for the first
        *n_struct* real atoms (padded coords: pad columns get weight 0), or 2-D
        ``(batch, n_atoms)`` where only the first row is used for every frame.
    topo_npz
        Path to Boltz processed-structure NPZ for :class:`PoseCVIndexer` topology.
    pocket_radius
        Å radius defining the binding pocket for ligand pose CVs (passed to
        :class:`PoseCVIndexer`).
    """
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot  # noqa: PLC0415
    from genai_tps.io.boltz_npz_export import load_topo  # noqa: PLC0415

    structures = np.asarray(structures, dtype=np.float64)
    if structures.ndim != 3:
        raise ValueError(f"structures must be (N, n_atoms, 3), got shape {structures.shape}")
    n_atoms_coords = int(structures.shape[1])

    structure, n_struct = load_topo(Path(topo_npz))
    n_s = int(n_struct)
    ref_np = reference_coords[:n_s].astype(np.float32)
    ref_t = torch.tensor(reference_coords, dtype=torch.float32)

    mask_arr = _normalize_atom_mask_for_compute_cvs(
        atom_mask_np,
        n_atoms_coords=n_atoms_coords,
        n_struct_topo=n_s,
    )
    mask_t = torch.tensor(mask_arr, dtype=torch.float32) if mask_arr is not None else None

    indexer = PoseCVIndexer(structure, ref_np, pocket_radius=float(pocket_radius))

    results: dict[str, list[float]] = {
        "rmsd": [],
        "rg": [],
        "contact_order": [],
        "clash_count": [],
        "ligand_rmsd": [],
        "ligand_pocket_dist": [],
    }

    for i, coords in enumerate(structures):
        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(coords_t, step_index=0, defer_numpy_coords=True)

        results["rmsd"].append(float(rmsd_to_reference(snap, ref_t, mask_t)))
        results["rg"].append(float(radius_of_gyration(snap, mask_t)))
        results["contact_order"].append(float(contact_order(snap)))
        results["clash_count"].append(float(clash_count(snap)))

        try:
            results["ligand_rmsd"].append(float(ligand_pose_rmsd(snap, indexer)))
            results["ligand_pocket_dist"].append(float(ligand_pocket_distance(snap, indexer)))
        except Exception:
            results["ligand_rmsd"].append(float("nan"))
            results["ligand_pocket_dist"].append(float("nan"))

        if (i + 1) % 100 == 0:
            print(f"  CVs computed for {i + 1}/{len(structures)}", flush=True)

    return {k: np.array(v) for k, v in results.items()}


def compute_simple_cvs(structures: np.ndarray, ref_coords: np.ndarray) -> dict[str, np.ndarray]:
    """Lightweight CVs without full Boltz topology (ligand channels zeroed)."""
    from genai_tps.backends.boltz.snapshot import BoltzSnapshot  # noqa: PLC0415

    ref_t = torch.tensor(ref_coords, dtype=torch.float32)
    results: dict[str, list[float]] = {
        cv: []
        for cv in ["rmsd", "rg", "contact_order", "clash_count", "ligand_rmsd", "ligand_pocket_dist"]
    }
    for coords in structures:
        ct = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        snap = BoltzSnapshot.from_gpu_batch(ct, step_index=0, defer_numpy_coords=True)
        results["rmsd"].append(float(rmsd_to_reference(snap, ref_t)))
        results["rg"].append(float(radius_of_gyration(snap)))
        results["contact_order"].append(float(contact_order(snap)))
        results["clash_count"].append(float(clash_count(snap)))
        results["ligand_rmsd"].append(0.0)
        results["ligand_pocket_dist"].append(0.0)
    return {k: np.array(v) for k, v in results.items()}


__all__.extend(
    [
        "make_training_sucos_pocket_qcov_traj_fn",
        "_normalize_atom_mask_for_compute_cvs",
        "compute_cvs",
        "compute_simple_cvs",
    ]
)

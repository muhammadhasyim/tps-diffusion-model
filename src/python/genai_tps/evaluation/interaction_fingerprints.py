"""Protein-ligand interaction fingerprint (IFP) analysis using ProLIF.

Computes per-interaction-type distance distributions from a molecular dynamics
or generative-sampling trajectory, then compares them with a reference ensemble
via the Wasserstein-1 distance (AnewSampling Appendix A.1).

Interaction types supported by ProLIF (≥2.0):
    HBDonor, HBAcceptor, Hydrophobic, PiStacking, PiCation, CationPi,
    Anionic, Cationic, MetalCoordination, VdWContact, ...

Usage example
-------------
>>> from genai_tps.evaluation.interaction_fingerprints import (
...     compute_interaction_distances,
...     interaction_ws_distances,
... )
>>> # traj_pred, traj_ref: MDAnalysis-compatible or ProLIF trajectory objects
>>> fp_pred = compute_interaction_distances(traj_pred, lig_sel="resname LIG")
>>> fp_ref  = compute_interaction_distances(traj_ref,  lig_sel="resname LIG")
>>> ws_result = interaction_ws_distances(fp_pred, fp_ref)

Dependencies
------------
- ``prolif>=2.0`` — install: ``pip install 'prolif>=2.0'`` or ``conda install -c conda-forge prolif``
- ``MDAnalysis>=2.2`` (pulled in by ProLIF)

References
----------
Wang Y. et al. (2026). bioRxiv 10.64898/2026.03.10.710952, Appendix A.1.
Bouysset C. & Fiorucci S. (2021). ProLIF: a library to encode molecular
    interactions as fingerprints. J. Cheminformatics 13, 72.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import prolif
import MDAnalysis as mda

from genai_tps.evaluation.distribution_metrics import wasserstein_1d

__all__ = [
    "compute_interaction_distances",
    "interaction_ws_distances",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_interaction_distances(
    traj: Any,
    lig_sel: str = "resname LIG",
    protein_sel: str = "protein",
    count_fingerprint: bool = False,
) -> dict[str, np.ndarray]:
    """Compute per-frame interaction fingerprints using ProLIF.

    Runs ``prolif.Fingerprint`` over a trajectory and returns per-interaction
    per-frame binary presence (or count) arrays.

    Parameters
    ----------
    traj:
        Trajectory accepted by ProLIF: an ``MDAnalysis.Universe`` or a ProLIF
        ``Trajectory`` object.  The universe must already have coordinates
        loaded (e.g. via ``Universe.load_new(...)``).
    lig_sel:
        MDAnalysis selection string for the ligand residue(s).
    protein_sel:
        MDAnalysis selection string for the protein.
    count_fingerprint:
        If True, use ``prolif.CountFingerprint`` to count repeated contacts
        per frame.  Default is binary (present/absent).

    Returns
    -------
    dict[str, np.ndarray]
        Maps interaction-label strings (e.g. ``"A2.HBAcceptor"``) to 1-D
        arrays of length T (number of frames) containing 0.0/1.0 (binary)
        or counts (when ``count_fingerprint=True``).

    Raises
    ------
    ValueError
        If the ligand or protein selection yields no atoms.
    """

    if isinstance(traj, mda.Universe):
        u = traj
    else:
        u = mda.Universe(traj)

    lig_atoms = u.select_atoms(lig_sel)
    prot_atoms = u.select_atoms(protein_sel)

    if len(lig_atoms) == 0:
        raise ValueError(f"Ligand selection {lig_sel!r} matched 0 atoms")
    if len(prot_atoms) == 0:
        raise ValueError(f"Protein selection {protein_sel!r} matched 0 atoms")

    fp_cls = prolif.CountFingerprint if count_fingerprint else prolif.Fingerprint
    fp = fp_cls()
    fp.run(u.trajectory, lig_atoms, prot_atoms)

    df = fp.to_dataframe()

    result: dict[str, np.ndarray] = {}
    for col in df.columns:
        label = ".".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
        result[label] = df[col].to_numpy().astype(float)

    return result


def interaction_ws_distances(
    fp_pred: dict[str, np.ndarray],
    fp_ref: dict[str, np.ndarray],
    only_shared: bool = True,
) -> dict[str, Any]:
    """Compute per-interaction Wasserstein-1 distances between two fingerprint sets.

    For each interaction type that appears in at least one ensemble, the W1
    distance between predicted and reference per-frame indicator arrays is
    computed.  For a binary 0/1 indicator this reduces to the absolute
    difference in occupation probability.

    Parameters
    ----------
    fp_pred:
        Output of :func:`compute_interaction_distances` for the predicted
        ensemble.
    fp_ref:
        Output of :func:`compute_interaction_distances` for the reference
        ensemble.
    only_shared:
        If True (default), only compute distances for interactions present in
        both ensembles.  If False, missing interactions are treated as always
        absent (all-zero arrays).

    Returns
    -------
    dict with keys:
        - ``"per_interaction_w1"`` : dict[str, float] -- W1 per interaction label
        - ``"mean_w1"``            : float -- mean over all (shared) interactions
        - ``"n_interactions"``     : int -- number of interactions compared
        - ``"labels"``             : list[str] -- sorted interaction labels

    Notes
    -----
    For binary fingerprints, W1 = |f_pred - f_ref| where f is the occupation
    fraction.  For count fingerprints it represents the distributional shift
    of interaction counts.
    """
    if only_shared:
        labels = sorted(set(fp_pred.keys()) & set(fp_ref.keys()))
    else:
        labels = sorted(set(fp_pred.keys()) | set(fp_ref.keys()))

    if not labels:
        return {
            "per_interaction_w1": {},
            "mean_w1": float("nan"),
            "n_interactions": 0,
            "labels": [],
        }

    per_w1: dict[str, float] = {}
    for label in labels:
        arr_p = fp_pred.get(label, np.zeros(1))
        arr_r = fp_ref.get(label, np.zeros(1))
        per_w1[label] = wasserstein_1d(arr_p, arr_r)

    values = np.array(list(per_w1.values()))
    return {
        "per_interaction_w1": per_w1,
        "mean_w1": float(values.mean()) if len(values) > 0 else float("nan"),
        "n_interactions": len(labels),
        "labels": labels,
    }

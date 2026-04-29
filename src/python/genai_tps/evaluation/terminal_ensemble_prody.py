"""ProDy-based analysis of TPS terminal-structure ensembles.

Given a collection of single-frame PDB files (one per accepted TPS path),
this module:

1. Parses all structures with ProDy and selects Cα atoms.
2. Builds a ``PDBEnsemble`` with iterative least-squares superposition.
3. Runs **PCA** (essential dynamics) on the aligned Cα coordinates.
4. Optionally runs **ANM** or **GNM** on the ensemble mean structure for
   comparison of empirical vs. theoretical soft modes.
5. Returns a :class:`EnsembleAnalysisResult` dataclass with arrays and
   metadata suitable for JSON serialisation and plotting.

This module has **no** hard dependency on Boltz or OpenMM.  All it needs is
``prody>=2.0`` and ``numpy``.  The caller is responsible for converting NPZ
checkpoints to PDB files first (see
:mod:`genai_tps.io.boltz_npz_export`).

Example::

    from pathlib import Path
    from genai_tps.evaluation.terminal_ensemble_prody import run_ensemble_analysis

    result = run_ensemble_analysis(
        pdb_dir=Path("cofolding_tps_out/cv_rmsd_analysis/last_frame_pdbs"),
        n_pcs=10,
        run_anm=True,
    )
    print(result.explained_variance_ratio[:3])
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EnsembleAnalysisResult:
    """Output of :func:`run_ensemble_analysis`.

    Attributes
    ----------
    n_structures:
        Number of structures successfully parsed and included in the ensemble.
    n_ca_atoms:
        Number of Cα atoms used (all structures share the same topology).
    eigenvalues:
        Array of shape ``(n_pcs,)`` — PCA eigenvalues (variance along each PC)
        in units of Å².
    explained_variance_ratio:
        Array of shape ``(n_pcs,)`` — fraction of total variance captured by
        each PC.
    cumulative_variance_ratio:
        Array of shape ``(n_pcs,)`` — cumulative variance explained.
    projections:
        Array of shape ``(n_structures, n_pcs)`` — per-structure projections
        onto the principal components (PC scores), in Å.
    labels:
        List of per-structure label strings (typically the PDB file stem or
        ``mc_step`` value extracted from the filename).
    mean_coords:
        Array of shape ``(n_ca_atoms, 3)`` — Cα mean coordinates after
        superposition, in Å.
    anm_eigenvalues:
        Array of ANM eigenvalues ``(n_modes,)`` for the mean structure, or
        ``None`` if ANM was not requested.
    gnm_eigenvalues:
        Array of GNM eigenvalues ``(n_modes,)`` for the mean structure, or
        ``None`` if GNM was not requested.
    failed_pdbs:
        List of PDB file stems that could not be parsed or had incompatible
        topology.
    """

    n_structures: int = 0
    n_ca_atoms: int = 0
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    explained_variance_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_variance_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    projections: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    labels: list[str] = field(default_factory=list)
    mean_coords: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 3))
    anm_eigenvalues: Optional[np.ndarray] = None
    gnm_eigenvalues: Optional[np.ndarray] = None
    failed_pdbs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        d = asdict(self)
        for key in (
            "eigenvalues",
            "explained_variance_ratio",
            "cumulative_variance_ratio",
            "projections",
            "mean_coords",
            "anm_eigenvalues",
            "gnm_eigenvalues",
        ):
            val = d[key]
            d[key] = val.tolist() if isinstance(val, np.ndarray) else val
        return d


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def run_ensemble_analysis(
    pdb_dir: Path,
    *,
    n_pcs: int = 10,
    run_anm: bool = False,
    run_gnm: bool = False,
    n_enm_modes: int = 20,
    selection: str = "calpha",
    glob: str = "*.pdb",
    label_from_stem: bool = True,
) -> EnsembleAnalysisResult:
    """Analyse a set of single-frame PDB files as a structural ensemble.

    Parameters
    ----------
    pdb_dir:
        Directory containing single-frame PDB files (one per accepted TPS
        path terminal structure).
    n_pcs:
        Number of principal components to retain.
    run_anm:
        Compute ANM on the ensemble mean structure and store eigenvalues.
    run_gnm:
        Compute GNM on the ensemble mean structure and store eigenvalues.
    n_enm_modes:
        Number of non-trivial modes to compute for ANM/GNM.
    selection:
        ProDy atom selection string applied to every parsed structure
        (default ``"calpha"``).
    glob:
        Glob pattern for selecting PDB files within *pdb_dir*.
    label_from_stem:
        If ``True``, use the file stem as the per-structure label and attempt
        to extract a ``mc_step`` integer from names matching
        ``tps_mc_step_NNNNNNNN``.

    Returns
    -------
    EnsembleAnalysisResult
        Populated result dataclass.  On failure (< 2 parseable structures),
        the eigenvalue arrays are empty and a warning is logged.
    """
    try:
        import prody as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ProDy is required for ensemble analysis. "
            "Install it with:  pip install 'prody>=2.0'"
        ) from exc

    pdb_paths = sorted(pdb_dir.glob(glob))
    if not pdb_paths:
        raise FileNotFoundError(f"No PDB files matching '{glob}' found in {pdb_dir}")

    result = EnsembleAnalysisResult()

    # ── 1. Parse all structures ──────────────────────────────────────────────
    # Suppress ProDy's chatty log output during parsing
    pd.confProDy(verbosity="none")

    atom_groups: list = []
    labels: list[str] = []
    failed: list[str] = []

    ref_n_atoms: Optional[int] = None

    for pdb_path in pdb_paths:
        try:
            ag = pd.parsePDB(str(pdb_path), model=1)
            if ag is None:
                raise ValueError("parsePDB returned None")
            ca = ag.select(selection)
            if ca is None or len(ca) == 0:
                raise ValueError(f"No atoms matched selection '{selection}'")
            n = len(ca)
            if ref_n_atoms is None:
                ref_n_atoms = n
            elif n != ref_n_atoms:
                raise ValueError(
                    f"Topology mismatch: expected {ref_n_atoms} Cα atoms, "
                    f"got {n}"
                )
            atom_groups.append(ca)
            labels.append(pdb_path.stem if label_from_stem else str(pdb_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", pdb_path.name, exc)
            failed.append(pdb_path.stem)

    result.failed_pdbs = failed
    n_ok = len(atom_groups)
    logger.info("Parsed %d / %d structures successfully.", n_ok, len(pdb_paths))

    if n_ok < 2:
        warnings.warn(
            f"Only {n_ok} structure(s) could be parsed; "
            "PCA requires at least 2.  Returning empty result.",
            RuntimeWarning,
            stacklevel=2,
        )
        return result

    # ── 2. Build ensemble with iterative superposition ───────────────────────
    ens = pd.PDBEnsemble("terminal_ensemble")
    ref_ag = atom_groups[0]
    ens.setAtoms(ref_ag)
    ens.setCoords(ref_ag.getCoords())

    for ag, label in zip(atom_groups, labels):
        ens.addCoordset(ag.getCoords(), label=label)

    ens.iterpose()

    result.n_structures = n_ok
    result.n_ca_atoms = ref_n_atoms
    result.labels = labels

    mean_coords = ens.getCoordsets().mean(axis=0)
    result.mean_coords = np.asarray(mean_coords, dtype=np.float64)

    # ── 3. PCA ───────────────────────────────────────────────────────────────
    pca = pd.PCA("terminal_pca")
    pca.buildCovariance(ens)
    n_pcs_actual = min(n_pcs, n_ok - 1, ref_n_atoms * 3 - 6)  # dof upper bound
    pca.calcModes(n_pcs_actual)

    eigenvalues = np.array([pca[i].getEigval() for i in range(n_pcs_actual)])
    total_var = eigenvalues.sum()
    if total_var > 0:
        evr = eigenvalues / total_var
    else:
        evr = np.zeros_like(eigenvalues)

    result.eigenvalues = eigenvalues
    result.explained_variance_ratio = evr
    result.cumulative_variance_ratio = np.cumsum(evr)

    # Per-structure projections (n_structures × n_pcs)
    proj = pd.calcProjection(ens, pca[:n_pcs_actual])
    result.projections = np.asarray(proj, dtype=np.float64)

    logger.info(
        "PCA: first 3 PCs explain %.1f%% of variance.",
        100.0 * result.cumulative_variance_ratio[min(2, n_pcs_actual - 1)],
    )

    # ── 4. ANM on mean structure ─────────────────────────────────────────────
    if run_anm:
        try:
            mean_ag = atom_groups[0].copy()
            mean_ag.setCoords(mean_coords)
            anm, _ = pd.calcANM(mean_ag, selstr=selection, n_modes=n_enm_modes)
            result.anm_eigenvalues = np.array(
                [anm[i].getEigval() for i in range(len(anm))]
            )
            logger.info("ANM: computed %d modes on mean structure.", len(anm))
        except Exception as exc:
            logger.warning("ANM failed: %s", exc)

    # ── 5. GNM on mean structure ─────────────────────────────────────────────
    if run_gnm:
        try:
            mean_ag = atom_groups[0].copy()
            mean_ag.setCoords(mean_coords)
            gnm, _ = pd.calcGNM(mean_ag, selstr=selection, n_modes=n_enm_modes)
            result.gnm_eigenvalues = np.array(
                [gnm[i].getEigval() for i in range(len(gnm))]
            )
            logger.info("GNM: computed %d modes on mean structure.", len(gnm))
        except Exception as exc:
            logger.warning("GNM failed: %s", exc)

    return result

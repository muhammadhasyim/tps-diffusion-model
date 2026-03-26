"""OpenMM-based collective variable: Cα-RMSD to local energy minimum.

This module provides :class:`OpenMMLocalMinRMSD`, which computes the same CV
used by ``watch_rmsd_live.py`` and ``compute_cv_rmsd.py``:

    Kabsch-aligned Cα-RMSD between:
        (a) the raw Boltz-generated structure (last frame of a TPS trajectory)
        (b) its own AMBER14/GBn2 energy-minimized local minimum

The CV is **per-structure** -- each snapshot is minimized independently.
This is expensive (~1-10 s per call on GPU depending on system size), so a
coordinate-hash LRU cache is included.  Identical coordinate arrays return
the cached value without re-running OpenMM.

The cache is particularly effective in the TPS loop because:
  - ``cv_old`` at step N is always the last-accepted trajectory's last frame.
  - ``cv_new`` at step N is the trial trajectory's last frame.
  - When a trial is *accepted*, ``cv_new`` at step N becomes ``cv_old`` at
    step N+1 -- so only one new minimization is needed per accepted step,
    and rejected trials cost at most one extra minimization.

Usage::

    from genai_tps.enhanced_sampling.openmm_cv import OpenMMLocalMinRMSD

    cv = OpenMMLocalMinRMSD(
        topo_npz=Path("boltz_results/processed/structures/foo.npz"),
        platform="CUDA",
        max_iter=500,
        cache_size=256,
    )

    # cv_function compatible with run_tps_path_sampling:
    rmsd = cv(trajectory)   # returns float, Ångström

Requirements:
    openmm, pdbfixer (or pdbfixer fallback path in compute_cv_rmsd.py),
    boltz (for topology loading and PDB writing).
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OpenMMLocalMinRMSD:
    """Cα-RMSD to AMBER14/GBn2 local minimum -- the canonical TPS validation CV.

    Matches the pipeline in ``watch_rmsd_live.py`` and ``compute_cv_rmsd.py``:

    1. Extract the **last frame** of the trajectory (fully denoised structure).
    2. Write coordinates to a temporary PDB using the Boltz topology.
    3. Run ``minimize_pdb`` (AMBER14 + GBn2 + OpenMM L-BFGS).
    4. Return Kabsch-aligned Cα-RMSD in Ångström.

    Results are cached by coordinate hash to avoid redundant minimizations.

    Parameters
    ----------
    topo_npz:
        Path to the Boltz ``processed/structures/*.npz`` file.  Used to
        write PDB files with the correct atom/chain topology.
    platform:
        OpenMM platform name (``"CUDA"``, ``"OpenCL"``, or ``"CPU"``).
    max_iter:
        Maximum L-BFGS iterations per minimization.  500 is sufficient for
        most structures (convergence checked by OpenMM).
    cache_size:
        Number of coordinate hash → RMSD pairs to retain in the LRU cache.
        With the TPS accept/reject pattern, 64--256 is typically sufficient.
    fallback_value:
        Value returned when minimization fails (default: ``float('nan')``).
        Use a large value (e.g. 999.0) to bias away from failed structures.
    mol_dir:
        Path to the Boltz CCD molecule directory (``~/.boltz/mols``).  When
        provided, SMILES for each NONPOLYMER chain are read from
        ``{mol_dir}/{ccd}.pkl`` and passed to ``minimize_pdb`` as
        ``ligand_smiles``, enabling GAFF2 parameterisation of ligands.
        If ``None``, ligand chains are not parameterised (AMBER14-only mode).
    """

    def __init__(
        self,
        topo_npz: Path | str,
        platform: str = "CUDA",
        max_iter: int = 500,
        cache_size: int = 256,
        fallback_value: float = float("nan"),
        mol_dir: Path | str | None = None,
    ):
        self.topo_npz = Path(topo_npz)
        self.platform = platform
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.fallback_value = fallback_value
        self.mol_dir = Path(mol_dir) if mol_dir is not None else None

        self._topo = None
        self._n_struct: int = 0
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._n_calls = 0
        self._n_cache_hits = 0
        self._n_failures = 0

    def _load_topo(self) -> None:
        """Lazy-load the Boltz topology (avoids import at module level)."""
        if self._topo is None:
            from genai_tps.analysis.boltz_npz_export import load_topo
            self._topo, self._n_struct = load_topo(self.topo_npz)
            logger.info(
                "OpenMMLocalMinRMSD: loaded topology from %s (%d atoms)",
                self.topo_npz, self._n_struct,
            )

    def _extract_ligand_smiles(self) -> dict[str, str]:
        """Return a chain-ID → SMILES mapping for all NONPOLYMER chains.

        Iterates the Boltz ``StructureV2`` topology, finds every chain whose
        ``mol_type`` equals ``const.chain_type_ids["NONPOLYMER"]``, reads the
        first residue name (the CCD code, e.g. ``"FZC"``, ``"ATP"``, ``"MG"``)
        and loads the corresponding RDKit Mol from ``{mol_dir}/{ccd}.pkl``.

        Returns
        -------
        dict[str, str]
            Maps PDB chain letter to SMILES string.  Returns an empty dict when
            ``mol_dir`` is ``None``, the topology is not yet loaded, or no
            NONPOLYMER chains are found.

        Notes
        -----
        Chains whose pkl file is missing or cannot be read are silently skipped
        (a WARNING is logged).  The caller should pass the result directly to
        ``minimize_pdb(ligand_smiles=...)``, which also handles missing entries
        gracefully.
        """
        if self.mol_dir is None or self._topo is None:
            return {}

        import pickle  # noqa: PLC0415

        try:
            from boltz.data import const  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "_extract_ligand_smiles: boltz not importable; "
                "cannot resolve ligand SMILES."
            )
            return {}

        nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
        result: dict[str, str] = {}

        for chain in self._topo.chains:
            if int(chain["mol_type"]) != nonpolymer_id:
                continue
            chain_name = str(chain["name"]).strip()
            res_idx = int(chain["res_idx"])
            ccd_code = str(self._topo.residues[res_idx]["name"]).strip()

            pkl_path = self.mol_dir / f"{ccd_code}.pkl"
            if not pkl_path.is_file():
                logger.warning(
                    "_extract_ligand_smiles: pkl for CCD '%s' not found at %s; "
                    "chain '%s' will use protein-only fallback.",
                    ccd_code, pkl_path, chain_name,
                )
                continue

            try:
                with pkl_path.open("rb") as fh:
                    mol = pickle.load(fh)  # noqa: S301 — trusted local file
                from rdkit.Chem import MolToSmiles  # noqa: PLC0415
                smiles = MolToSmiles(mol)
                result[chain_name] = smiles
                logger.info(
                    "_extract_ligand_smiles: chain '%s' CCD='%s' SMILES='%.40s'",
                    chain_name, ccd_code, smiles,
                )
            except Exception as exc:
                logger.warning(
                    "_extract_ligand_smiles: failed to get SMILES for chain '%s' "
                    "CCD='%s': %s",
                    chain_name, ccd_code, exc,
                )

        return result

    def _coords_to_pdb_string(self, coords_angstrom: np.ndarray) -> str:
        """Convert a (n_atoms, 3) Å coordinate array to a PDB string.

        Uses the Boltz topology to assign atom names, residues, and chains.
        ``coords_angstrom`` must have exactly ``self._n_struct`` rows.

        Parameters
        ----------
        coords_angstrom:
            Shape ``(n_atoms, 3)`` in Ångström.

        Returns
        -------
        str
            PDB-format string.
        """
        from boltz.data.types import Coords, Interface  # noqa: PLC0415
        from boltz.data.write.pdb import to_pdb  # noqa: PLC0415
        from dataclasses import replace

        n = self._n_struct
        fc = np.asarray(coords_angstrom[:n], dtype=np.float32)

        atoms = self._topo.atoms.copy()
        atoms["coords"] = fc
        atoms["is_present"] = True
        residues = self._topo.residues.copy()
        residues["is_present"] = True
        coord_arr = np.array([(x,) for x in fc], dtype=Coords)
        interfaces = np.array([], dtype=Interface)

        new_s = replace(
            self._topo,
            atoms=atoms,
            residues=residues,
            interfaces=interfaces,
            coords=coord_arr,
        )
        return to_pdb(new_s, plddts=None, boltz2=True)

    @staticmethod
    def _coord_hash(coords: np.ndarray) -> str:
        """Stable MD5 hash of a float32 coordinate array for cache keying."""
        arr = np.asarray(coords, dtype=np.float32)
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _minimize_coords(self, coords_angstrom: np.ndarray) -> float:
        """Write a temp PDB and run OpenMM minimization; return Cα-RMSD (Å)."""
        import sys
        scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from compute_cv_rmsd import minimize_pdb  # type: ignore[import]

        pdb_str = self._coords_to_pdb_string(coords_angstrom)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False, prefix="opes_cv_"
        ) as f:
            f.write(pdb_str)
            tmp_path = Path(f.name)

        ligand_smiles = self._extract_ligand_smiles() or None

        try:
            result = minimize_pdb(
                tmp_path,
                max_iter=self.max_iter,
                platform_name=self.platform,
                ligand_smiles=ligand_smiles,
            )
            if result["converged"] and result["ca_rmsd_angstrom"] is not None:
                return float(result["ca_rmsd_angstrom"])
            else:
                logger.warning(
                    "OpenMMLocalMinRMSD: minimization failed: %s", result.get("error")
                )
                self._n_failures += 1
                return self.fallback_value
        finally:
            tmp_path.unlink(missing_ok=True)

    def _get_last_frame_coords(self, trajectory) -> np.ndarray | None:
        """Extract last-frame Cα coordinates from a TPS Trajectory.

        Supports ``BoltzSnapshot`` (with ``tensor_coords``) and generic OPS
        snapshots (with ``.coordinates``).

        Returns
        -------
        np.ndarray or None
            Shape ``(n_atoms, 3)`` in Ångström, or None if extraction fails.
        """
        snap = trajectory[-1]

        tc = getattr(snap, "tensor_coords", None)
        if tc is not None:
            try:
                arr = tc.detach().cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[0]
                return arr.astype(np.float32)
            except Exception as exc:
                logger.warning("OpenMMLocalMinRMSD: tensor_coords extraction failed: %s", exc)

        coords = getattr(snap, "coordinates", None)
        if coords is not None:
            try:
                arr = np.asarray(coords, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                return arr
            except Exception as exc:
                logger.warning("OpenMMLocalMinRMSD: coordinates extraction failed: %s", exc)

        return None

    def __call__(self, trajectory) -> float:
        """Compute Cα-RMSD-to-local-minimum for the last frame of *trajectory*.

        Parameters
        ----------
        trajectory:
            An OpenPathSampling ``Trajectory`` whose last snapshot contains
            Boltz coordinates.

        Returns
        -------
        float
            Kabsch Cα-RMSD in Ångström.  Returns ``self.fallback_value`` on
            failure (e.g. minimization diverged or topology mismatch).
        """
        self._load_topo()
        self._n_calls += 1

        coords = self._get_last_frame_coords(trajectory)
        if coords is None:
            logger.warning("OpenMMLocalMinRMSD: could not extract coordinates from last frame.")
            return self.fallback_value

        key = self._coord_hash(coords)
        if key in self._cache:
            self._n_cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        value = self._minimize_coords(coords)

        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        if self._n_calls % 50 == 0:
            hit_rate = 100.0 * self._n_cache_hits / self._n_calls
            logger.info(
                "OpenMMLocalMinRMSD: %d calls, %.1f%% cache hit rate, %d failures",
                self._n_calls, hit_rate, self._n_failures,
            )

        return value

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of calls served from cache (0.0 -- 1.0)."""
        if self._n_calls == 0:
            return 0.0
        return self._n_cache_hits / self._n_calls

    def clear_cache(self) -> None:
        """Discard the coordinate cache (frees memory, forces recomputation)."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return a summary of call statistics."""
        return {
            "n_calls": self._n_calls,
            "n_cache_hits": self._n_cache_hits,
            "n_failures": self._n_failures,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size_current": len(self._cache),
        }

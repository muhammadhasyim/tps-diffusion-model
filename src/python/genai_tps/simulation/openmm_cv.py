"""OpenMM-based collective variables for Boltz-generated structures.

Two CVs are provided:

:class:`OpenMMLocalMinRMSD`
    Kabsch-aligned Cα-RMSD between the raw Boltz structure and its own
    AMBER14 + explicit TIP3P energy-minimized local minimum.  Expensive (~1–10 s per
    call on GPU) but captures how far the generated geometry is from a
    nearby physical minimum.

:class:`OpenMMEnergy`
    Raw AMBER14 + explicit TIP3P single-point potential energy (kJ/mol) evaluated at
    the Boltz geometry **without any minimization**.  Faster than RMSD-to-
    minimum because it skips L-BFGS, and directly exposes high-energy
    (strained / clashing) configurations as large positive values.

Both classes use a coordinate-hash LRU cache, which is particularly
effective in TPS accept/reject cycling where adjacent steps share frames.

Usage::

    from genai_tps.simulation.openmm_cv import (
        OpenMMLocalMinRMSD, OpenMMEnergy,
    )

    # RMSD-to-minimum CV (existing):
    rmsd_cv = OpenMMLocalMinRMSD(topo_npz=..., platform="CUDA")
    rmsd = rmsd_cv(trajectory)   # Ångström

    # Raw energy CV (new):
    energy_cv = OpenMMEnergy(topo_npz=..., platform="CUDA")
    energy = energy_cv(trajectory)   # kJ/mol

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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def ligand_smiles_dict_for_boltz_structure(
    structure: Any,
    mol_dir: Path | str | None,
) -> dict[str, str]:
    """Return PDB chain ID → SMILES for each NONPOLYMER chain (Boltz CCD pickles).

    Looks up ``{mol_dir}/{CCD}.pkl`` (RDKit mol) for each ligand/cofactor chain,
    same convention as :class:`OpenMMLocalMinRMSD`.
    """
    if mol_dir is None:
        return {}
    mdir = Path(mol_dir)

    import pickle  # noqa: PLC0415

    from boltz.data import const  # noqa: PLC0415
    from rdkit.Chem import MolToSmiles  # noqa: PLC0415

    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]
    result: dict[str, str] = {}
    for chain in structure.chains:
        if int(chain["mol_type"]) != nonpolymer_id:
            continue
        chain_name = str(chain["name"]).strip()
        res_idx = int(chain["res_idx"])
        ccd_code = str(structure.residues[res_idx]["name"]).strip()
        pkl_path = mdir / f"{ccd_code}.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(
                f"CCD '{ccd_code}' molecule pickle not found at {pkl_path}. "
                "Point mol_dir at the Boltz cache mols directory (~/.boltz/mols)."
            )
        with pkl_path.open("rb") as fh:
            mol = pickle.load(fh)  # noqa: S301 — trusted local Boltz cache file
        smiles = MolToSmiles(mol)
        result[chain_name] = smiles
        logger.info(
            "ligand_smiles_dict: chain '%s' CCD='%s' SMILES='%.40s'",
            chain_name,
            ccd_code,
            smiles,
        )
    return result


def _detect_ligand_smiles(
    structure: Any,
    mol_dir: Path | str | None,
) -> dict[str, str]:
    """Used by :mod:`genai_tps.simulation.openmm_md_runner` with ``--mol-dir``."""
    return ligand_smiles_dict_for_boltz_structure(structure, mol_dir)


class OpenMMLocalMinRMSD:
    """Cα-RMSD to AMBER14 + explicit-TIP3P local minimum -- the canonical TPS validation CV.

    Matches the pipeline in ``watch_rmsd_live.py`` and ``compute_cv_rmsd.py``:

    1. Extract the **last frame** of the trajectory (fully denoised structure).
    2. Write coordinates to a temporary PDB using the Boltz topology.
    3. Run ``minimize_pdb`` (AMBER14 + explicit TIP3P + OpenMM L-BFGS).
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
        Value returned when minimization fails (default: ``999.0`` Å).
        Large finite values bias OPES away from failed structures without
        poisoning numerical updates; avoid ``nan`` for enhanced sampling.
    mol_dir:
        Path to the Boltz CCD molecule directory (``~/.boltz/mols``).  When
        provided, SMILES for each NONPOLYMER chain are read from
        ``{mol_dir}/{ccd}.pkl`` and passed to ``minimize_pdb`` as
        ``ligand_smiles``, enabling GAFF2 parameterisation of ligands.
        If ``None``, ligand chains are not parameterised (AMBER14-only mode).
    openmm_device_index:
        Optional CUDA/OpenCL ordinal passed to OpenMM as ``DeviceIndex`` when
        *platform* is ``CUDA`` or ``OpenCL``.
    """

    def __init__(
        self,
        topo_npz: Path | str,
        platform: str = "CUDA",
        max_iter: int = 500,
        cache_size: int = 256,
        fallback_value: float = 999.0,
        mol_dir: Path | str | None = None,
        openmm_device_index: int | None = None,
    ):
        self.topo_npz = Path(topo_npz)
        self.platform = platform
        self.openmm_device_index = openmm_device_index
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
        # Persistent OpenMM context: built once on first minimization call
        self._simulation = None
        self._omm_topology = None
        self._omm_ca_indices: list[int] | None = None
        self._ligand_smiles_cache: dict[str, str] | None = None

    def _load_topo(self) -> None:
        """Lazy-load the Boltz topology (avoids import at module level)."""
        if self._topo is None:
            from genai_tps.io.boltz_npz_export import load_topo
            self._topo, self._n_struct = load_topo(self.topo_npz)
            logger.info(
                "OpenMMLocalMinRMSD: loaded topology from %s (%d atoms)",
                self.topo_npz, self._n_struct,
            )

    def _extract_ligand_smiles(self) -> dict[str, str]:
        """Return chain-ID → SMILES for NONPOLYMER chains (Boltz CCD ``.pkl``)."""
        if self.mol_dir is None or self._topo is None:
            return {}
        return ligand_smiles_dict_for_boltz_structure(self._topo, self.mol_dir)

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
        """Run OpenMM minimization using a cached ligand SMILES; return Cα-RMSD (Å).

        The ligand SMILES dictionary is extracted once from the topology and
        cached on the instance, eliminating the per-call RDKit/pickle overhead.
        The OpenMM context itself is rebuilt per call (topology is constant),
        but the ``GAFFTemplateGenerator`` construction is skipped on repeat calls
        via the ``_ligand_smiles_cache`` attribute.
        """
        import sys
        scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from compute_cv_rmsd import minimize_pdb  # type: ignore[import]
        from genai_tps.utils.compute_device import openmm_device_index_properties

        pdb_str = self._coords_to_pdb_string(coords_angstrom)

        # Cache ligand SMILES once (avoids per-call RDKit/pickle overhead)
        if self._ligand_smiles_cache is None:
            self._ligand_smiles_cache = self._extract_ligand_smiles() or {}
        ligand_smiles = self._ligand_smiles_cache or None
        omm_props = openmm_device_index_properties(
            self.platform, self.openmm_device_index
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False, prefix="opes_cv_"
        ) as f:
            f.write(pdb_str)
            tmp_path = Path(f.name)

        try:
            result = minimize_pdb(
                tmp_path,
                max_iter=self.max_iter,
                platform_name=self.platform,
                ligand_smiles=ligand_smiles,
                platform_properties=omm_props if omm_props else None,
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

    def _get_last_frame_coords(self, trajectory) -> np.ndarray:
        """Extract last-frame coordinates from a TPS Trajectory.

        Supports ``BoltzSnapshot`` (with ``tensor_coords``) and generic OPS
        snapshots (with ``.coordinates``).

        Returns
        -------
        np.ndarray
            Shape ``(n_atoms, 3)`` in Ångström.

        Raises
        ------
        TypeError
            When the snapshot has neither ``tensor_coords`` nor ``coordinates``.
        """
        snap = trajectory[-1]

        tc = getattr(snap, "tensor_coords", None)
        if tc is not None:
            arr = tc.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            return arr.astype(np.float32)

        coords = getattr(snap, "coordinates", None)
        if coords is not None:
            arr = np.asarray(coords, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            return arr

        raise TypeError(
            f"OpenMMLocalMinRMSD: snapshot {type(snap).__name__} has neither "
            "'tensor_coords' nor 'coordinates' attribute."
        )

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
    def n_calls(self) -> int:
        return self._n_calls

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


class OpenMMEnergy:
    """Single-point AMBER14 + explicit-TIP3P potential energy (kJ/mol) — no minimization.

    For each evaluation the class:

    1. Converts Boltz heavy-atom coordinates to a PDB string.
    2. Adds hydrogens via PDBFixer (or Modeller fallback).
    3. Builds an AMBER14 + explicit TIP3P (PME) OpenMM system.
    4. Returns the potential energy at the **raw** (un-minimized) geometry.

    Steps 2–3 are repeated per unique coordinate set because PDBFixer
    determines hydrogen placement from the heavy-atom geometry, and we
    must not reuse stale H positions across structures.  An LRU cache
    keyed on the coordinate hash prevents redundant evaluations for
    trajectories that share frames (common in TPS accept/reject cycling).

    Bad / strained configurations from the diffusion model will produce
    very high energies, making this CV ideal for detecting physically
    implausible tails in the generative distribution.

    Parameters
    ----------
    topo_npz:
        Path to the Boltz ``processed/structures/*.npz`` file.
    platform:
        OpenMM platform name (``"CUDA"``, ``"OpenCL"``, ``"CPU"``).
    fallback_value:
        Energy returned when evaluation fails (default: ``1e8`` kJ/mol).
    cache_size:
        LRU cache entries (coordinate hash → energy).
    mol_dir:
        Path to Boltz CCD molecule directory for ligand parameterization.
    openmm_device_index:
        Optional CUDA/OpenCL ordinal for OpenMM ``DeviceIndex``.
    """

    def __init__(
        self,
        topo_npz: Path | str,
        platform: str = "CUDA",
        fallback_value: float = 1e8,
        cache_size: int = 256,
        mol_dir: Path | str | None = None,
        openmm_device_index: int | None = None,
    ):
        self.topo_npz = Path(topo_npz)
        self.platform = platform
        self.openmm_device_index = openmm_device_index
        self.fallback_value = float(fallback_value)
        self.cache_size = cache_size
        self.mol_dir = Path(mol_dir) if mol_dir is not None else None

        self._topo = None
        self._n_struct: int = 0
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._n_calls = 0
        self._n_cache_hits = 0
        self._n_failures = 0
        self._ligand_smiles_cache: dict[str, str] | None = None

    def _load_topo(self) -> None:
        if self._topo is None:
            from genai_tps.io.boltz_npz_export import load_topo
            self._topo, self._n_struct = load_topo(self.topo_npz)

    def _extract_ligand_smiles(self) -> dict[str, str]:
        """Chain-ID → SMILES for NONPOLYMER chains (mirrors OpenMMLocalMinRMSD)."""
        if self.mol_dir is None or self._topo is None:
            return {}
        return ligand_smiles_dict_for_boltz_structure(self._topo, self.mol_dir)

    def _coords_to_pdb_string(self, coords_angstrom: np.ndarray) -> str:
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
            self._topo, atoms=atoms, residues=residues,
            interfaces=interfaces, coords=coord_arr,
        )
        return to_pdb(new_s, plddts=None, boltz2=True)

    def _evaluate_energy(self, coords_angstrom: np.ndarray) -> float:
        """Delegate to ``minimize_pdb`` with ``max_iter=1`` and return the energy.

        Reuses the full ligand-handling pipeline in ``compute_cv_rmsd.minimize_pdb``
        (PDBFixer on protein-only, GAFF2 for multi-atom cofactors, tip3p ions,
        OpenFF conformer insertion) so that systems with ``LIG`` / ``HETATM``
        residues are parameterized correctly — identical to ``OpenMMLocalMinRMSD``.

        With ``max_iter=1`` the L-BFGS takes a single step, so the returned
        energy is essentially the raw single-point energy of the Boltz geometry
        (the perturbation from one gradient step is negligible compared to the
        large energy differences between good and bad structures).
        """
        import sys
        scripts_dir = Path(__file__).resolve().parents[4] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from compute_cv_rmsd import minimize_pdb  # type: ignore[import]
        from genai_tps.utils.compute_device import openmm_device_index_properties

        pdb_str = self._coords_to_pdb_string(coords_angstrom)

        if self._ligand_smiles_cache is None:
            self._ligand_smiles_cache = self._extract_ligand_smiles() or {}
        ligand_smiles = self._ligand_smiles_cache or None
        omm_props = openmm_device_index_properties(
            self.platform, self.openmm_device_index
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False, prefix="omm_energy_"
        ) as f:
            f.write(pdb_str)
            tmp_path = Path(f.name)

        try:
            result = minimize_pdb(
                tmp_path,
                max_iter=1,
                platform_name=self.platform,
                ligand_smiles=ligand_smiles,
                platform_properties=omm_props if omm_props else None,
            )
            if result["converged"] and result["energy_kj_mol"] is not None:
                return float(result["energy_kj_mol"])
            else:
                logger.warning(
                    "OpenMMEnergy: minimize_pdb failed: %s", result.get("error")
                )
                self._n_failures += 1
                return self.fallback_value
        finally:
            tmp_path.unlink(missing_ok=True)

    def _get_last_frame_coords(self, trajectory) -> np.ndarray:
        """Extract last-frame coordinates (Å) from a TPS trajectory.

        Raises
        ------
        TypeError
            When the snapshot has neither ``tensor_coords`` nor ``coordinates``.
        """
        snap = trajectory[-1]
        tc = getattr(snap, "tensor_coords", None)
        if tc is not None:
            arr = tc.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            return arr.astype(np.float32)
        coords = getattr(snap, "coordinates", None)
        if coords is not None:
            arr = np.asarray(coords, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            return arr
        raise TypeError(
            f"OpenMMEnergy: snapshot {type(snap).__name__} has neither "
            "'tensor_coords' nor 'coordinates' attribute."
        )

    @staticmethod
    def _coord_hash(coords: np.ndarray) -> str:
        arr = np.asarray(coords, dtype=np.float32)
        return hashlib.md5(arr.tobytes()).hexdigest()

    def __call__(self, trajectory) -> float:
        """Return AMBER14 + explicit-TIP3P potential energy (kJ/mol) of the last frame.

        High values indicate strained / physically implausible configurations.
        No energy minimization is performed — this is the raw single-point
        energy of the Boltz-generated structure with template-placed hydrogens.
        """
        self._load_topo()
        self._n_calls += 1

        coords = self._get_last_frame_coords(trajectory)

        key = self._coord_hash(coords)
        if key in self._cache:
            self._n_cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        value = self._evaluate_energy(coords)

        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        if self._n_calls % 50 == 0:
            hit_rate = 100.0 * self._n_cache_hits / self._n_calls
            logger.info(
                "OpenMMEnergy: %d calls, %.1f%% cache hit, %d failures",
                self._n_calls, hit_rate, self._n_failures,
            )

        return value

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def stats(self) -> dict:
        return {
            "n_calls": self._n_calls,
            "n_cache_hits": self._n_cache_hits,
            "n_failures": self._n_failures,
            "cache_hit_rate": (
                self._n_cache_hits / self._n_calls if self._n_calls > 0 else 0.0
            ),
            "cache_size_current": len(self._cache),
        }

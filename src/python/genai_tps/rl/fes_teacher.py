"""OpenMM + OPES ``teacher`` for FES-guided RL (target CV distribution)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from genai_tps.backends.boltz.collective_variables import (
    PoseCVIndexer,
    ligand_pocket_distance,
    ligand_pose_rmsd,
)
from genai_tps.enhanced_sampling.opes_bias import OPESBias

__all__ = [
    "OpenMMTeacher",
    "boltz_terminal_pose_cv_numpy",
    "build_openmm_indices_for_boltz_atoms",
    "load_build_md_simulation_from_pdb",
]


def load_build_md_simulation_from_pdb() -> Callable[..., tuple[Any, dict]]:
    """Load :func:`build_md_simulation_from_pdb` from ``scripts/compute_cv_rmsd.py``."""

    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "scripts" / "compute_cv_rmsd.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Expected OpenMM helper script at {script_path}")
    spec = importlib.util.spec_from_file_location("compute_cv_rmsd", script_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load compute_cv_rmsd module spec.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "build_md_simulation_from_pdb", None)
    if fn is None:
        raise AttributeError("compute_cv_rmsd.build_md_simulation_from_pdb missing.")
    return fn


def _openmm_residue_sequence_number(residue: Any) -> int:
    rid = residue.id
    if isinstance(rid, tuple):
        return int(rid[0])
    return int(rid)


def _boltz_atom_pdb_key(structure: Any, atom_idx: int) -> tuple[str, int, str]:
    """Return ``(chain_name, pdb_residue_number, atom_name)`` for a Boltz atom index."""
    atoms = structure.atoms
    residues = structure.residues
    chains = structure.chains
    for chain in chains:
        a0 = int(chain["atom_idx"])
        a1 = a0 + int(chain["atom_num"])
        if not (a0 <= atom_idx < a1):
            continue
        chain_name = str(chain["name"]).strip()
        rs = int(chain["res_idx"])
        re_end = rs + int(chain["res_num"])
        for ri in range(rs, re_end):
            res = residues[ri]
            r0 = int(res["atom_idx"])
            r1 = r0 + int(res["atom_num"])
            if r0 <= atom_idx < r1:
                pdb_resnum = int(res["res_idx"]) + 1  # matches Boltz ``to_pdb``
                aname = str(atoms[atom_idx]["name"]).strip().upper()
                return chain_name, pdb_resnum, aname
    raise ValueError(f"atom_idx {atom_idx} not found in Boltz structure.")


def build_openmm_indices_for_boltz_atoms(structure: Any, h_topology: Any) -> np.ndarray:
    """Map each Boltz heavy-atom index to an OpenMM atom index in *h_topology*.

    Matching uses PDB semantics ``(chain_id, residue_sequence_number, atom_name)``
    for non-hydrogen atoms.  The reference PDB used to build *h_topology* should
    come from the same topology as *structure* (e.g. :func:`npz_to_pdb
    <genai_tps.analysis.boltz_npz_export.npz_to_pdb>`).
    """
    n = int(structure.atoms.shape[0])
    openmm_keys_to_idx: dict[tuple[str, int, str], int] = {}
    for atom in h_topology.atoms():
        if atom.element is not None and atom.element.symbol == "H":
            continue
        chain_id = atom.residue.chain.id.strip()
        seq = _openmm_residue_sequence_number(atom.residue)
        name = atom.name.strip().upper()
        key = (chain_id, seq, name)
        if key in openmm_keys_to_idx:
            raise ValueError(f"Duplicate OpenMM heavy-atom key: {key}")
        openmm_keys_to_idx[key] = int(atom.index)

    out = np.empty(n, dtype=np.int64)
    missing: list[tuple[int, tuple[str, int, str]]] = []
    for i in range(n):
        key = _boltz_atom_pdb_key(structure, i)
        j = openmm_keys_to_idx.get(key)
        if j is None:
            missing.append((i, key))
            out[i] = -1
        else:
            out[i] = j
    if missing:
        raise ValueError(
            "Could not map some Boltz atoms to OpenMM topology; "
            f"first misses: {missing[:5]!r}"
        )
    return out


def boltz_terminal_pose_cv_numpy(
    x_next: torch.Tensor,
    n_struct: int,
    indexer: PoseCVIndexer,
) -> np.ndarray:
    """2-D CV vector: ligand pose RMSD (Å), ligand–pocket Cα COM distance (Å)."""
    x = x_next[:, : int(n_struct), :].detach().float()
    snap = SimpleNamespace(_tensor_coords_gpu=x)
    rmsd = ligand_pose_rmsd(snap, indexer)
    dist = ligand_pocket_distance(snap, indexer)
    return np.array([rmsd, dist], dtype=np.float64)


class OpenMMTeacher:
    """Persistent OpenMM Langevin simulation + :class:`OPESBias` on CV space.

    Collective variables on the OpenMM side are computed by mapping OpenMM
    positions onto Boltz-order heavy-atom coordinates (Å) and evaluating the
    same pose CVs as the Boltz student.

    Parameters
    ----------
    pdb_path:
        Heavy-atom reference PDB (same layout as Boltz ``structures/*.npz``).
    structure:
        Boltz :class:`StructureV2` from :func:`~genai_tps.analysis.boltz_npz_export.load_topo`.
    pose_indexer:
        :class:`PoseCVIndexer` built with reference coordinates aligned with the
        student (typically first-frame / docked pose).
    ligand_smiles:
        Optional chain ID → SMILES (passed to ``build_md_simulation_from_pdb``).
    platform_name:
        OpenMM platform preference.
    temperature_k:
        Langevin temperature (K).
    opes:
        Pre-constructed :class:`OPESBias` instance (owns ``ndim``, ``kbt``, …).
    logp_kbt:
        Thermal energy used only for scaling ``log_p_target = -V / logp_kbt``
        (often match *opes*.``kbt`` in kJ/mol).
    """

    def __init__(
        self,
        pdb_path: Path,
        structure: Any,
        pose_indexer: PoseCVIndexer,
        opes: OPESBias,
        *,
        ligand_smiles: dict[str, str] | None = None,
        platform_name: str = "CUDA",
        temperature_k: float = 300.0,
        logp_kbt: float = 2.494,
        minimize_steps: int = 0,
    ) -> None:
        self.structure = structure
        self.indexer = pose_indexer
        self.opes = opes
        self.logp_kbt = float(logp_kbt)
        self._n_boltz = int(structure.atoms.shape[0])

        build_md = load_build_md_simulation_from_pdb()
        self.sim, meta = build_md(
            Path(pdb_path),
            platform_name=platform_name,
            temperature_k=temperature_k,
            ligand_smiles=ligand_smiles,
        )
        self.platform_used: str = str(meta["platform_used"])
        h_topology = self.sim.topology
        self._omm_idx = build_openmm_indices_for_boltz_atoms(structure, h_topology)
        self._md_counter = 0

        if minimize_steps > 0:
            self.sim.minimizeEnergy(maxIterations=int(minimize_steps))

    def _positions_boltz_order_angstrom(self) -> np.ndarray:
        state = self.sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)  # nm, shape (N_omm, 3)
        out = np.zeros((self._n_boltz, 3), dtype=np.float64)
        for i in range(self._n_boltz):
            j = int(self._omm_idx[i])
            out[i] = pos[j] * 10.0
        return out

    def current_cv_numpy(self) -> np.ndarray:
        """Pose CV vector from the current OpenMM coordinates."""
        coords = self._positions_boltz_order_angstrom()
        x = torch.from_numpy(coords).float().unsqueeze(0)
        snap = SimpleNamespace(_tensor_coords_gpu=x)
        return np.array(
            [ligand_pose_rmsd(snap, self.indexer), ligand_pocket_distance(snap, self.indexer)],
            dtype=np.float64,
        )

    def set_positions_from_boltz(
        self,
        coords_angstrom: np.ndarray,
        *,
        minimize_steps: int = 200,
    ) -> None:
        """Warm-start OpenMM from Boltz heavy-atom coordinates.

        Maps Boltz-order heavy-atom positions (Angstroms) onto the full OpenMM
        topology via ``self._omm_idx`` (hydrogen positions are left unchanged),
        sets the context positions, and runs a short energy minimisation.

        Parameters
        ----------
        coords_angstrom:
            Shape ``(N_boltz, 3)`` in Angstroms (same layout as Boltz
            ``structures/*.npz`` atom coordinates).
        minimize_steps:
            ``minimizeEnergy`` iterations after setting positions.  Set to 0
            to skip minimisation entirely.
        """
        import openmm.unit as unit  # deferred to avoid hard dep at import time

        coords = np.asarray(coords_angstrom, dtype=np.float64)
        if coords.ndim == 3:
            coords = coords.squeeze(0)
        if coords.shape[0] < self._n_boltz:
            raise ValueError(
                f"Expected at least {self._n_boltz} atoms, got {coords.shape[0]}."
            )
        coords = coords[: self._n_boltz]

        state = self.sim.context.getState(getPositions=True)
        cur_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        new_pos = cur_pos.copy()
        for i in range(self._n_boltz):
            j = int(self._omm_idx[i])
            new_pos[j] = coords[i] / 10.0  # Angstrom -> nm

        self.sim.context.setPositions(new_pos * unit.nanometers)
        if minimize_steps > 0:
            self.sim.minimizeEnergy(maxIterations=int(minimize_steps))

    def run_md_burst(
        self,
        n_steps: int,
        deposit_pace: int,
    ) -> list[tuple[np.ndarray, float]]:
        """Run *n_steps* Langevin steps; deposit OPES kernels every *deposit_pace* steps.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            ``(cv, weight)`` pairs for each deposit event (weight is ``1.0`` placeholder).
        """
        if deposit_pace < 1:
            raise ValueError("deposit_pace must be >= 1.")
        out: list[tuple[np.ndarray, float]] = []
        for _ in range(int(n_steps)):
            self.sim.step(1)
            self._md_counter += 1
            if self._md_counter % deposit_pace == 0:
                cv = self.current_cv_numpy()
                self.opes.update(cv, self._md_counter)
                out.append((cv.copy(), 1.0))
        return out

    def log_p_target(self, cv: np.ndarray | Sequence[float]) -> float:
        """Surrogate target log-density: ``-V(\\mathbf{s}) / (k_B T)``.

        Here :math:`V` is the OPES bias from :meth:`OPESBias.evaluate`.  This is a
        practical control signal for RL, not a fully normalised Boltzmann log-density.
        """
        v = float(self.opes.evaluate(cv))
        return -v / self.logp_kbt

    def fes_on_grid(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        *,
        i_dim: int = 0,
        j_dim: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``-kT * log p`` surrogate (OPES bias *V*) on a 2-D grid (diagnostics)."""
        g1, g2 = np.meshgrid(s1, s2, indexing="ij")
        z = np.zeros_like(g1, dtype=np.float64)
        for a in range(g1.shape[0]):
            for b in range(g1.shape[1]):
                vec = np.zeros(self.opes.ndim, dtype=np.float64)
                vec[i_dim] = g1[a, b]
                vec[j_dim] = g2[a, b]
                z[a, b] = self.opes.evaluate(vec)
        return g1, g2, z

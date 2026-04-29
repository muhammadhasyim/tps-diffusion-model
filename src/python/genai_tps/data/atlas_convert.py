"""Convert cached ATLAS trajectories into WDSM fine-tuning arrays."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class AtomMapError(ValueError):
    """Raised when ATLAS trajectory atoms cannot be mapped to Boltz atom order."""


@dataclass(frozen=True)
class TrajectoryAtomRecord:
    """Atom identity from an ATLAS trajectory topology."""

    chain_id: str
    residue_index: int
    residue_name: str
    atom_name: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        """Strict atom-mapping key."""
        return (
            self.chain_id.strip() or "_",
            int(self.residue_index),
            self.residue_name.strip().upper(),
            self.atom_name.strip().upper(),
        )


@dataclass(frozen=True)
class BoltzAtomRecord:
    """Atom identity in Boltz processed atom order."""

    chain_id: str
    residue_index: int
    residue_name: str
    atom_name: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        """Strict atom-mapping key."""
        return (
            self.chain_id.strip() or "_",
            int(self.residue_index),
            self.residue_name.strip().upper(),
            self.atom_name.strip().upper(),
        )


@dataclass(frozen=True)
class WdsmArrays:
    """WDSM-compatible arrays plus conversion metadata."""

    coords: np.ndarray
    logw: np.ndarray
    atom_mask: np.ndarray
    frame_indices: np.ndarray


def build_atom_index_map(
    trajectory_atoms: Sequence[TrajectoryAtomRecord],
    boltz_atoms: Sequence[BoltzAtomRecord],
) -> np.ndarray:
    """Return indices that reorder trajectory coordinates into Boltz atom order."""
    lookup: dict[tuple[str, int, str, str], int] = {}
    duplicates: list[tuple[str, int, str, str]] = []
    for idx, atom in enumerate(trajectory_atoms):
        key = atom.key
        if key in lookup:
            duplicates.append(key)
        lookup[key] = idx
    if duplicates:
        preview = ", ".join(map(str, duplicates[:5]))
        raise AtomMapError(f"ATLAS trajectory contains duplicate atom keys: {preview}")

    index_map = []
    missing = []
    for atom in boltz_atoms:
        key = atom.key
        if key not in lookup:
            missing.append(key)
            continue
        index_map.append(lookup[key])

    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise AtomMapError(
            f"ATLAS trajectory is missing {len(missing)} atoms required by Boltz order: {preview}"
        )
    return np.asarray(index_map, dtype=np.int64)


def sample_frame_indices(
    n_frames: int,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Select deterministic frame indices from a trajectory."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    candidates = np.arange(0, n_frames, stride, dtype=np.int64)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")
        if len(candidates) > max_frames:
            rng = np.random.default_rng(seed)
            candidates = np.sort(rng.choice(candidates, size=max_frames, replace=False))
    return candidates


def wdsm_arrays_from_trajectory(
    trajectory_coords: np.ndarray,
    index_map: np.ndarray,
    *,
    frame_indices: np.ndarray | None = None,
) -> WdsmArrays:
    """Reorder trajectory coordinates and attach uniform WDSM weights."""
    coords = np.asarray(trajectory_coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"trajectory_coords must be (T, N, 3), got {coords.shape}")
    selected = coords[:, np.asarray(index_map, dtype=np.int64), :]
    n_samples, n_atoms, _ = selected.shape
    frames = (
        np.arange(n_samples, dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64)
    )
    if len(frames) != n_samples:
        raise ValueError("frame_indices length must match selected coordinates.")
    return WdsmArrays(
        coords=np.asarray(selected, dtype=np.float32),
        logw=np.zeros(n_samples, dtype=np.float64),
        atom_mask=np.ones((n_samples, n_atoms), dtype=np.float32),
        frame_indices=frames,
    )


def write_wdsm_npz(
    path: Path,
    *,
    coords: np.ndarray,
    logw: np.ndarray,
    atom_mask: np.ndarray,
) -> None:
    """Write arrays in ``ReweightedStructureDataset`` format."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coords=np.asarray(coords, dtype=np.float32),
        logw=np.asarray(logw, dtype=np.float64),
        atom_mask=np.asarray(atom_mask, dtype=np.float32),
    )


def structure_with_coordinate_ensemble(structure, coords: np.ndarray):
    """Return a ``StructureV2`` copy whose coordinate ensemble is ``coords``.

    Parameters
    ----------
    structure:
        Boltz ``StructureV2``-like object.
    coords:
        ``(T, N, 3)`` coordinates in Angstrom, already ordered like
        ``structure.atoms``.
    """
    coords_array = np.asarray(coords, dtype=np.float32)
    if coords_array.ndim != 3 or coords_array.shape[2] != 3:
        raise ValueError(f"coords must be (T, N, 3), got {coords_array.shape}")
    n_frames, n_atoms, _ = coords_array.shape
    if n_atoms != len(structure.atoms):
        raise ValueError(
            f"Coordinate atom count ({n_atoms}) does not match StructureV2 atoms "
            f"({len(structure.atoms)})."
        )
    coord_dtype = structure.coords.dtype
    ensemble_dtype = structure.ensemble.dtype
    flat_coords = np.array([(xyz,) for xyz in coords_array.reshape(-1, 3)], dtype=coord_dtype)
    ensemble = np.array(
        [(frame_idx * n_atoms, n_atoms) for frame_idx in range(n_frames)],
        dtype=ensemble_dtype,
    )
    atoms = structure.atoms.copy()
    atoms["coords"] = coords_array[0]
    atoms["is_present"] = True
    return replace(structure, atoms=atoms, coords=flat_coords, ensemble=ensemble)


def dump_structure_with_coordinate_ensemble(
    structure,
    coords: np.ndarray,
    output_npz: Path,
) -> None:
    """Write a Boltz ``StructureV2`` NPZ with ATLAS frames as conformers."""
    updated = structure_with_coordinate_ensemble(structure, coords)
    output_npz = Path(output_npz).expanduser()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    updated.dump(output_npz)


def write_frame_map(
    path: Path,
    *,
    record_id: str,
    n_frames: int,
    logw: Sequence[float] | None = None,
) -> None:
    """Write a JSON frame map consumed by multi-protein WDSM datasets."""
    weights = [0.0] * n_frames if logw is None else [float(x) for x in logw]
    if len(weights) != n_frames:
        raise ValueError("logw length must match n_frames.")
    payload = {
        "samples": [
            {"record_id": record_id, "frame_idx": idx, "logw": weights[idx]}
            for idx in range(n_frames)
        ]
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_conversion_metadata(path: Path, payload: dict[str, object]) -> None:
    """Write a deterministic JSON sidecar describing an ATLAS conversion."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_atlas_zip(zip_path: Path, extract_dir: Path, *, overwrite: bool = False) -> Path:
    """Extract one ATLAS ZIP and return the extraction directory."""
    zip_path = Path(zip_path).expanduser()
    extract_dir = Path(extract_dir).expanduser()
    marker = extract_dir / ".extracted"
    if marker.is_file() and not overwrite:
        return extract_dir
    if not zip_path.is_file():
        raise FileNotFoundError(f"ATLAS ZIP not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extractall(zf, extract_dir)
    marker.write_text(str(zip_path) + "\n", encoding="utf-8")
    return extract_dir


def find_atlas_trajectory_files(extracted_dir: Path, atlas_id: str) -> list[tuple[str, Path, Path]]:
    """Find ATLAS replicate XTC/PDB pairs in an extracted protein bundle.

    Returns
    -------
    list of tuple
        ``(replicate_name, xtc_path, topology_pdb_path)`` entries sorted by
        replicate name.
    """
    extracted_dir = Path(extracted_dir).expanduser()
    pdb_candidates = sorted(extracted_dir.rglob("*.pdb"))
    if not pdb_candidates:
        raise FileNotFoundError(f"No PDB topology found under {extracted_dir}")
    topology_pdb = _prefer_starting_pdb(pdb_candidates, atlas_id)
    entries: list[tuple[str, Path, Path]] = []
    for xtc in sorted(extracted_dir.rglob("*.xtc")):
        stem = xtc.stem
        replicate = _replicate_name(stem)
        entries.append((replicate, xtc, topology_pdb))
    if not entries:
        raise FileNotFoundError(f"No XTC trajectories found under {extracted_dir}")
    return entries


def load_mdtraj_frames(
    xtc_path: Path,
    topology_path: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    seed: int = 0,
):
    """Load sampled frames from an ATLAS XTC as an ``mdtraj.Trajectory``."""
    try:
        import mdtraj as md  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("ATLAS conversion requires mdtraj.") from exc

    traj = md.load(str(xtc_path), top=str(topology_path), stride=stride)
    if max_frames is not None and traj.n_frames > max_frames:
        indices = sample_frame_indices(traj.n_frames, stride=1, max_frames=max_frames, seed=seed)
        traj = traj[indices]
    return traj


def trajectory_atom_records_from_mdtraj(topology) -> list[TrajectoryAtomRecord]:
    """Build strict atom records from an mdtraj topology.

    Residue indices are one-based ordinal positions within each chain.  This
    avoids mismatches from PDB author numbering/insertion codes when Boltz has
    reconstructed a normalized topology from FASTA/YAML input.
    """
    records: list[TrajectoryAtomRecord] = []
    for chain_idx, chain in enumerate(topology.chains):
        chain_id = _chain_id(chain, chain_idx)
        residues = list(chain.residues)
        residue_ordinals = {residue: i + 1 for i, residue in enumerate(residues)}
        for residue in residues:
            residue_index = residue_ordinals[residue]
            for atom in residue.atoms:
                records.append(
                    TrajectoryAtomRecord(
                        chain_id=chain_id,
                        residue_index=residue_index,
                        residue_name=str(residue.name),
                        atom_name=str(atom.name),
                    )
                )
    return records


def boltz_atom_records_from_structure(structure) -> list[BoltzAtomRecord]:
    """Build atom records from a Boltz ``StructureV2``-like object."""
    records: list[BoltzAtomRecord] = []
    chains = structure.chains
    residues = structure.residues
    atoms = structure.atoms

    for chain_idx, chain in enumerate(chains):
        chain_id = _array_scalar(chain, "name", default=str(chain_idx))
        res_start = int(_array_scalar(chain, "res_idx"))
        res_count = int(_array_scalar(chain, "res_num"))
        for residue_ordinal, residue in enumerate(residues[res_start : res_start + res_count], start=1):
            residue_name = str(_array_scalar(residue, "name"))
            atom_start = int(_array_scalar(residue, "atom_idx"))
            atom_count = int(_array_scalar(residue, "atom_num"))
            for atom in atoms[atom_start : atom_start + atom_count]:
                records.append(
                    BoltzAtomRecord(
                        chain_id=chain_id,
                        residue_index=residue_ordinal,
                        residue_name=residue_name,
                        atom_name=str(_array_scalar(atom, "name")),
                    )
                )
    return records


def convert_mdtraj_trajectory_to_wdsm(traj, boltz_structure) -> WdsmArrays:
    """Convert a loaded mdtraj trajectory to Boltz-ordered WDSM arrays."""
    trajectory_atoms = trajectory_atom_records_from_mdtraj(traj.topology)
    boltz_atoms = boltz_atom_records_from_structure(boltz_structure)
    index_map = build_atom_index_map(trajectory_atoms, boltz_atoms)
    # mdtraj stores nanometers; WDSM/Boltz code expects Angstrom.
    coords_angstrom = np.asarray(traj.xyz, dtype=np.float32) * 10.0
    return wdsm_arrays_from_trajectory(coords_angstrom, index_map)


def concatenate_wdsm_arrays(arrays: Sequence[WdsmArrays]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate several WDSM array bundles."""
    if not arrays:
        raise ValueError("No WDSM arrays to concatenate.")
    coords = np.concatenate([a.coords for a in arrays], axis=0)
    logw = np.concatenate([a.logw for a in arrays], axis=0)
    atom_mask = np.concatenate([a.atom_mask for a in arrays], axis=0)
    return coords, logw, atom_mask


def metadata_for_records(records: Sequence[object]) -> list[dict[str, object]]:
    """Convert dataclass records to metadata dictionaries."""
    return [asdict(record) for record in records]


def _prefer_starting_pdb(candidates: Sequence[Path], atlas_id: str) -> Path:
    normalized = atlas_id.lower()
    for candidate in candidates:
        stem = candidate.stem.lower()
        if stem == normalized:
            return candidate
    for candidate in candidates:
        stem = candidate.stem.lower()
        if "end" not in stem:
            return candidate
    return candidates[0]


def _safe_extractall(zf: zipfile.ZipFile, extract_dir: Path) -> None:
    root = extract_dir.resolve()
    for member in zf.infolist():
        target = (extract_dir / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe path in ATLAS ZIP: {member.filename}")
    zf.extractall(extract_dir)


def _replicate_name(stem: str) -> str:
    upper = stem.upper()
    for token in ("R1", "R2", "R3"):
        if token in upper:
            return token
    return stem


def _chain_id(chain, chain_idx: int) -> str:
    for attr in ("chain_id", "id"):
        value = getattr(chain, attr, None)
        if value not in (None, ""):
            return str(value)
    return str(chain_idx)


def _array_scalar(row, field: str, *, default: object | None = None):
    names = getattr(getattr(row, "dtype", None), "names", None)
    if names is not None and field in names:
        return row[field].item() if hasattr(row[field], "item") else row[field]
    if default is not None:
        return default
    raise KeyError(f"Field {field!r} not present in structured row.")

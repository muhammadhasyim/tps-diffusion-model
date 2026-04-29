"""Assemble per-step WDSM NPZ shards into a single training NPZ."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genai_tps.training.diagnostics import clip_log_weights, weight_statistics


@dataclass(frozen=True)
class AssembledWdsmDataset:
    """Stacked coords / log weights / atom mask."""

    coords: np.ndarray
    logw: np.ndarray
    atom_mask: np.ndarray


def assemble_wdsm_from_directory(
    wdsm_dir: Path,
    *,
    topo_npz: Path | None = None,
    max_log_ratio: float | None = None,
    min_step: int = 0,
    max_step: int | None = None,
) -> AssembledWdsmDataset:
    """Load ``wdsm_step_*.npz`` files and return stacked arrays."""
    pattern = re.compile(r"wdsm_step_(\d+)\.npz$")
    files: list[tuple[int, Path]] = []
    for f in sorted(wdsm_dir.glob("wdsm_step_*.npz")):
        m = pattern.search(f.name)
        if m:
            step = int(m.group(1))
            if step >= min_step and (max_step is None or step <= max_step):
                files.append((step, f))
    files.sort(key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(f"No wdsm_step_*.npz files found in {wdsm_dir}")

    n_struct: int | None = None
    if topo_npz is not None:
        from genai_tps.io.boltz_npz_export import load_topo

        _, n_struct = load_topo(Path(topo_npz))
        n_struct = int(n_struct)

    coords_list = []
    logw_list = []

    for _step, fpath in files:
        data = np.load(fpath)
        coords_list.append(data["coords"].astype(np.float32))
        logw_list.append(float(data["logw"]))

    coords_array = np.stack(coords_list, axis=0)
    logw_array = np.array(logw_list, dtype=np.float64)
    n_samples, n_atoms, _three = coords_array.shape

    if n_struct is not None and n_struct < n_atoms:
        atom_mask = np.zeros((n_samples, n_atoms), dtype=np.float32)
        atom_mask[:, :n_struct] = 1.0
    else:
        atom_mask = np.ones((n_samples, n_atoms), dtype=np.float32)

    if max_log_ratio is not None:
        logw_array = clip_log_weights(logw_array, max_log_ratio)

    return AssembledWdsmDataset(coords=coords_array, logw=logw_array, atom_mask=atom_mask)


def save_assembled_npz(path: Path, assembled: AssembledWdsmDataset) -> None:
    """Write ``coords``, ``logw``, ``atom_mask`` keys compatible with :class:`~genai_tps.training.dataset.ReweightedStructureDataset`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coords=assembled.coords, logw=assembled.logw, atom_mask=assembled.atom_mask)


def print_assembly_stats(assembled: AssembledWdsmDataset) -> None:
    """Log weight statistics (stdout)."""
    stats = weight_statistics(assembled.logw)
    n_samples = assembled.coords.shape[0]
    print(
        f"[assemble] Dataset: {n_samples} samples; "
        f"max_weight_fraction={stats['max_weight_fraction']:.4f}"
    )

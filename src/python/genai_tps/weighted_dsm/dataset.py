"""Dataset for importance-weighted structure data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genai_tps.weighted_dsm.diagnostics import effective_sample_size


class ReweightedStructureDataset(Dataset):
    """Dataset of (coords, logw, atom_mask) tuples for weighted DSM training.

    Parameters
    ----------
    coords:
        (N, M, 3) atom coordinates in Angstrom.
    logw:
        (N,) log importance weights from enhanced sampling reweighting.
    atom_mask:
        (N, M) binary mask (1 = real atom, 0 = padding).
    """

    def __init__(
        self,
        coords: np.ndarray,
        logw: np.ndarray,
        atom_mask: np.ndarray,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float32)
        self.logw = np.asarray(logw, dtype=np.float64)
        self.atom_mask = np.asarray(atom_mask, dtype=np.float32)
        assert self.coords.ndim == 3, f"coords must be (N, M, 3), got {self.coords.shape}"
        assert self.logw.ndim == 1, f"logw must be (N,), got {self.logw.shape}"
        assert len(self.coords) == len(self.logw)
        self.n_eff = effective_sample_size(self.logw)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "coords": self.coords[idx],
            "logw": self.logw[idx],
            "atom_mask": self.atom_mask[idx],
        }

    @classmethod
    def from_npz(cls, path: str | Path) -> ReweightedStructureDataset:
        """Load from NPZ with keys 'coords', 'logw', optionally 'atom_mask'."""
        data = np.load(path)
        coords = data["coords"]
        logw = data["logw"]
        if "atom_mask" in data:
            atom_mask = data["atom_mask"]
        else:
            atom_mask = np.ones(coords.shape[:2], dtype=np.float32)
        ds = cls(coords, logw, atom_mask)
        print(
            f"[ReweightedStructureDataset] Loaded {len(ds)} samples from {Path(path).name}, "
            f"N_eff={ds.n_eff:.1f} ({ds.n_eff / len(ds) * 100:.1f}%)"
        )
        return ds

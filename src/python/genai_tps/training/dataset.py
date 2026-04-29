"""Dataset for importance-weighted structure data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from genai_tps.training.diagnostics import effective_sample_size


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


def split_wdsm_npz_train_val(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split assembled NPZ into ``train.npz`` / ``val.npz`` with reproducible shuffle."""
    from genai_tps.training.diagnostics import weight_statistics

    data_path = Path(data_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    coords = data["coords"]
    logw = data["logw"]
    atom_mask = data["atom_mask"] if "atom_mask" in data else np.ones(coords.shape[:2], dtype=np.float32)

    n_samples = len(logw)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    split = int((1.0 - val_fraction) * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_out = output_dir / "train.npz"
    val_out = output_dir / "val.npz"
    np.savez(train_out, coords=coords[train_idx], logw=logw[train_idx], atom_mask=atom_mask[train_idx])
    np.savez(val_out, coords=coords[val_idx], logw=logw[val_idx], atom_mask=atom_mask[val_idx])

    for name, idx in [("Full", np.arange(n_samples)), ("Train", train_idx), ("Val", val_idx)]:
        lw = logw[idx]
        stats = weight_statistics(lw)
        n_eff = effective_sample_size(lw)
        print(
            f"[split] {name:5s}: N={len(idx):6d}  N_eff={n_eff:7.1f} ({n_eff/len(idx)*100:5.1f}%)  "
            f"logw=[{lw.min():.3f}, {lw.max():.3f}]  std={lw.std():.3f}"
        )

    print(f"\n[split] Saved {train_out} ({len(train_idx)} samples)")
    print(f"[split] Saved {val_out} ({len(val_idx)} samples)")
    return train_out, val_out

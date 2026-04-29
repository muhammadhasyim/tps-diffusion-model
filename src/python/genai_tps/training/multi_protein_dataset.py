"""Multi-protein WDSM dataset built on Boltz-2's v2 feature pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameSample:
    """One WDSM supervision sample from a Boltz ``StructureV2`` ensemble."""

    record_id: str
    frame_idx: int
    logw: float = 0.0


def read_frame_map(path: Path) -> list[FrameSample]:
    """Read a multi-protein frame map JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Frame map must contain a 'samples' list.")
    samples = [
        FrameSample(
            record_id=str(item["record_id"]),
            frame_idx=int(item["frame_idx"]),
            logw=float(item.get("logw", 0.0)),
        )
        for item in raw_samples
    ]
    if not samples:
        raise ValueError(f"No frame samples found in {path}")
    return samples


def structure_single_frame(structure, frame_idx: int):
    """Return a ``StructureV2`` copy containing one selected conformer."""
    n_atoms = len(structure.atoms)
    n_frames = len(structure.ensemble)
    if frame_idx < 0 or frame_idx >= n_frames:
        raise IndexError(f"frame_idx={frame_idx} outside StructureV2 ensemble size {n_frames}.")
    start = int(structure.ensemble[frame_idx]["atom_coord_idx"])
    count = int(structure.ensemble[frame_idx]["atom_num"])
    if count != n_atoms:
        raise ValueError(
            f"Frame {frame_idx} has {count} atoms but StructureV2 atom table has {n_atoms}."
        )
    frame_coords = np.asarray(
        structure.coords[start : start + count]["coords"],
        dtype=np.float32,
    )
    atoms = structure.atoms.copy()
    atoms["coords"] = frame_coords
    atoms["is_present"] = True
    coords = np.array([(xyz,) for xyz in frame_coords], dtype=structure.coords.dtype)
    ensemble = np.array([(0, n_atoms)], dtype=structure.ensemble.dtype)
    return replace(structure, atoms=atoms, coords=coords, ensemble=ensemble)


class MultiProteinWdsmDataset(Dataset):
    """ATLAS/WDSM samples featurized by Boltz-2's v2 inference pipeline.

    The dataset intentionally reuses the Boltz submodule instead of copying it.
    Each item loads a ``StructureV2`` record, selects one coordinate frame from
    the structure ensemble, then runs ``Boltz2Tokenizer`` and ``Boltz2Featurizer``.
    """

    def __init__(
        self,
        *,
        manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Path,
        frame_samples: Sequence[FrameSample],
        constraints_dir: Path | None = None,
        template_dir: Path | None = None,
        extra_mols_dir: Path | None = None,
        max_atoms: int | None = None,
        max_tokens: int | None = None,
        max_seqs: int = 1024,
        seed: int = 0,
        load_input_fn: Callable[..., Any] | None = None,
        tokenizer: Any | None = None,
        featurizer: Any | None = None,
        molecule_loader: Callable[[Any, Any], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest = manifest
        self.target_dir = Path(target_dir)
        self.msa_dir = Path(msa_dir)
        self.mol_dir = Path(mol_dir)
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.extra_mols_dir = extra_mols_dir
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = int(seed)
        self.frame_samples = list(frame_samples)
        self._records_by_id = {record.id: record for record in manifest.records}
        missing = sorted({sample.record_id for sample in self.frame_samples} - set(self._records_by_id))
        if missing:
            raise KeyError(f"Frame map references records missing from manifest: {missing[:10]}")

        if load_input_fn is None:
            from boltz.data.module.inferencev2 import load_input as load_input_fn  # noqa: PLC0415
        if tokenizer is None:
            from boltz.data.tokenize.boltz2 import Boltz2Tokenizer  # noqa: PLC0415

            tokenizer = Boltz2Tokenizer()
        if featurizer is None:
            from boltz.data.feature.featurizerv2 import Boltz2Featurizer  # noqa: PLC0415

            featurizer = Boltz2Featurizer()
        if molecule_loader is None:
            molecule_loader = _load_boltz_molecules
        self.load_input_fn = load_input_fn
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.molecule_loader = molecule_loader

    def __len__(self) -> int:
        return len(self.frame_samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.frame_samples[idx]
        record = self._records_by_id[sample.record_id]
        input_data = self.load_input_fn(
            record=record,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
            template_dir=self.template_dir,
            extra_mols_dir=self.extra_mols_dir,
            affinity=False,
        )
        input_data = replace(
            input_data,
            structure=structure_single_frame(input_data.structure, sample.frame_idx),
        )
        tokenized = self.tokenizer.tokenize(input_data)
        molecules = self.molecule_loader(self.mol_dir, tokenized)
        random = np.random.default_rng(self.seed + idx)
        features = self.featurizer.process(
            tokenized,
            molecules=molecules,
            random=random,
            training=True,
            max_atoms=self.max_atoms,
            max_tokens=self.max_tokens,
            max_seqs=self.max_seqs,
            pad_to_max_seqs=False,
            single_sequence_prop=0.0,
            compute_frames=True,
            compute_constraint_features=True,
            num_ensembles=1,
            fix_single_ensemble=True,
        )
        features["logw"] = torch.tensor(float(sample.logw), dtype=torch.float32)
        features["atlas_frame_idx"] = torch.tensor(int(sample.frame_idx), dtype=torch.long)
        features["wdsm_record_id"] = sample.record_id
        return features


def multi_protein_collate(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate Boltz features while preserving WDSM metadata."""
    record_ids = [item.pop("wdsm_record_id") for item in data]
    try:
        from boltz.data.module.inferencev2 import collate as boltz_collate  # noqa: PLC0415
    except Exception:
        boltz_collate = _fallback_collate
    batch = boltz_collate(data)
    batch["wdsm_record_id"] = record_ids
    return batch


def _fallback_collate(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Small tensor collate used only when Boltz's import-heavy collate is unavailable."""
    keys = data[0].keys()
    collated: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in data]
        if all(torch.is_tensor(v) for v in values):
            shapes = [tuple(v.shape) for v in values]
            if len(set(shapes)) == 1:
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = _pad_and_stack(values)
        else:
            collated[key] = values
    return collated


def _pad_and_stack(values: list[torch.Tensor]) -> torch.Tensor:
    max_shape = tuple(max(v.shape[dim] for v in values) for dim in range(values[0].ndim))
    padded = []
    for value in values:
        out = value.new_zeros(max_shape)
        slices = tuple(slice(0, size) for size in value.shape)
        out[slices] = value
        padded.append(out)
    return torch.stack(padded, dim=0)


def _load_boltz_molecules(mol_dir: Path, tokenized) -> dict[str, Any]:
    from boltz.data.mol import load_canonicals, load_molecules  # noqa: PLC0415

    molecules = {}
    molecules.update(load_canonicals(Path(mol_dir)))
    if tokenized.extra_mols:
        molecules.update(tokenized.extra_mols)
    mol_names = set(tokenized.tokens["res_name"].tolist())
    missing = mol_names - set(molecules.keys())
    molecules.update(load_molecules(Path(mol_dir), missing))
    return molecules

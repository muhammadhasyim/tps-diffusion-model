"""Tests for multi-protein WDSM dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from genai_tps.training.multi_protein_dataset import (
    FrameSample,
    MultiProteinWdsmDataset,
    multi_protein_collate,
    read_frame_map,
    structure_single_frame,
)


@dataclass(frozen=True)
class FakeStructure:
    atoms: np.ndarray
    coords: np.ndarray
    ensemble: np.ndarray


@dataclass(frozen=True)
class FakeInput:
    structure: FakeStructure


def _fake_structure() -> FakeStructure:
    atom_dtype = [
        ("name", np.dtype("<U4")),
        ("coords", np.dtype("3f4")),
        ("is_present", np.dtype("?")),
        ("bfactor", np.dtype("f4")),
        ("plddt", np.dtype("f4")),
    ]
    coord_dtype = [("coords", np.dtype("3f4"))]
    ensemble_dtype = [("atom_coord_idx", np.dtype("i4")), ("atom_num", np.dtype("i4"))]
    atoms = np.zeros(3, dtype=atom_dtype)
    atoms["name"] = ["N", "CA", "C"]
    atoms["is_present"] = True
    frame0 = np.zeros((3, 3), dtype=np.float32)
    frame1 = np.ones((3, 3), dtype=np.float32) * 5.0
    coords = np.array([(xyz,) for xyz in np.concatenate([frame0, frame1], axis=0)], dtype=coord_dtype)
    ensemble = np.array([(0, 3), (3, 3)], dtype=ensemble_dtype)
    return FakeStructure(atoms=atoms, coords=coords, ensemble=ensemble)


def test_read_frame_map(tmp_path):
    frame_map = tmp_path / "frame_map.json"
    frame_map.write_text(
        '{"samples": [{"record_id": "p1", "frame_idx": 2, "logw": 1.5}]}',
        encoding="utf-8",
    )

    samples = read_frame_map(frame_map)

    assert samples == [FrameSample(record_id="p1", frame_idx=2, logw=1.5)]


def test_structure_single_frame_selects_requested_conformer():
    structure = _fake_structure()

    selected = structure_single_frame(structure, 1)

    assert len(selected.ensemble) == 1
    np.testing.assert_allclose(selected.coords["coords"], np.ones((3, 3), dtype=np.float32) * 5.0)
    np.testing.assert_allclose(selected.atoms["coords"], np.ones((3, 3), dtype=np.float32) * 5.0)


def test_multi_protein_dataset_featurizes_selected_frame():
    records = [SimpleNamespace(id="p1")]
    manifest = SimpleNamespace(records=records)
    structure = _fake_structure()

    def load_input_fn(**_kwargs):
        return FakeInput(structure=structure)

    class Tokenizer:
        def tokenize(self, input_data):
            return SimpleNamespace(structure=input_data.structure)

    class Featurizer:
        def process(self, data, **_kwargs):
            coords = torch.from_numpy(data.structure.coords["coords"].copy()).unsqueeze(0)
            return {
                "coords": coords,
                "atom_pad_mask": torch.ones(coords.shape[1], dtype=torch.float32),
                "token_index": torch.arange(1),
            }

    ds = MultiProteinWdsmDataset(
        manifest=manifest,
        target_dir=".",
        msa_dir=".",
        mol_dir=".",
        frame_samples=[FrameSample(record_id="p1", frame_idx=1, logw=0.25)],
        load_input_fn=load_input_fn,
        tokenizer=Tokenizer(),
        featurizer=Featurizer(),
        molecule_loader=lambda _mol_dir, _tokenized: {},
    )

    item = ds[0]

    assert item["logw"].item() == 0.25
    assert item["atlas_frame_idx"].item() == 1
    assert item["wdsm_record_id"] == "p1"
    assert item["coords"].shape == (1, 3, 3)
    assert torch.all(item["coords"] == 5.0)


def test_multi_protein_collate_preserves_record_ids():
    items = [
        {
            "coords": torch.zeros(1, 3, 3),
            "atom_pad_mask": torch.ones(3),
            "logw": torch.tensor(0.0),
            "atlas_frame_idx": torch.tensor(0),
            "wdsm_record_id": "p1",
        },
        {
            "coords": torch.ones(1, 3, 3),
            "atom_pad_mask": torch.ones(3),
            "logw": torch.tensor(0.0),
            "atlas_frame_idx": torch.tensor(1),
            "wdsm_record_id": "p2",
        },
    ]

    batch = multi_protein_collate(items)

    assert batch["coords"].shape == (2, 1, 3, 3)
    assert batch["wdsm_record_id"] == ["p1", "p2"]

"""Unit tests for GPU-native PoseBusters geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from genai_tps.evaluation.posebusters import (
    GPUPoseBustersEvaluator,
    POSEBUSTERS_GPU_CV_PREFIX,
    POSEBUSTERS_GPU_PASS_FRACTION,
    cv_name_for_gpu_column,
    expand_bias_cv_posebusters_gpu_all,
    expand_posebusters_gpu_all_to_cv_names,
    gpu_check_columns,
    make_posebusters_gpu_cached_column_scalar_fns,
    make_posebusters_gpu_pass_fraction_traj_fn,
    pass_fraction_from_gpu_row,
    validate_posebusters_gpu_bias_cv_names,
    vector_from_gpu_row,
)


@dataclass
class _FakeStructure:
    chains: list[dict]
    atoms: list[dict]


def _fake_structure() -> _FakeStructure:
    atoms = [
        {"name": "CA"},
        {"name": "CA"},
        {"name": "CA"},
        {"name": "O"},
        {"name": "C1"},
        {"name": "N1"},
        {"name": "O1"},
    ]
    chains = [
        {"atom_idx": 0, "atom_num": 4, "mol_type": 0, "name": "A"},
        {"atom_idx": 4, "atom_num": 3, "mol_type": 3, "name": "LIG"},
    ]
    return _FakeStructure(chains=chains, atoms=atoms)


def _reference_coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.1, 0.2, 0.1],
            [2.3, 0.2, 0.1],
            [2.2, 0.4, 0.1],
        ],
        dtype=np.float32,
    )


def _snapshot(coords: np.ndarray):
    class _Snap:
        pass

    snap = _Snap()
    snap.tensor_coords = torch.as_tensor(coords[None, ...], dtype=torch.float32)
    return snap


def test_expand_posebusters_gpu_all() -> None:
    new, raw, cvs = expand_bias_cv_posebusters_gpu_all("posebusters_gpu_all")
    assert raw == gpu_check_columns()
    assert cvs == [cv_name_for_gpu_column(c) for c in raw]
    assert new == ",".join(cvs)


def test_expand_posebusters_gpu_all_requires_sole_token() -> None:
    with pytest.raises(ValueError, match="only token"):
        expand_bias_cv_posebusters_gpu_all("posebusters_gpu_all,rg")


def test_validate_posebusters_gpu_bias_cv_names() -> None:
    validate_posebusters_gpu_bias_cv_names([POSEBUSTERS_GPU_PASS_FRACTION])
    validate_posebusters_gpu_bias_cv_names([cv_name_for_gpu_column(gpu_check_columns()[0])])
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_posebusters_gpu_bias_cv_names([POSEBUSTERS_GPU_PASS_FRACTION, "rg"])
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_posebusters_gpu_bias_cv_names([cv_name_for_gpu_column(gpu_check_columns()[0]), "rg"])


def test_pass_fraction_and_vector_from_gpu_row() -> None:
    row = {gpu_check_columns()[0]: 1.0, gpu_check_columns()[1]: 0.0}
    assert pass_fraction_from_gpu_row(row) == pytest.approx(0.5)
    vec = vector_from_gpu_row(row, [gpu_check_columns()[0], "missing"])
    assert vec[0] == 1.0 and np.isnan(vec[1])


def test_gpu_evaluator_reference_pose_passes() -> None:
    ev = GPUPoseBustersEvaluator(_fake_structure(), 7, _reference_coords())
    row = ev.evaluate_snapshot(_snapshot(_reference_coords()))
    assert set(row) == set(gpu_check_columns())
    assert pass_fraction_from_gpu_row(row) > 0.8


def test_gpu_evaluator_far_pose_fails_multiple_checks() -> None:
    coords = _reference_coords().copy()
    coords[4:] += np.array([20.0, 0.0, 0.0], dtype=np.float32)
    ev = GPUPoseBustersEvaluator(_fake_structure(), 7, _reference_coords())
    row = ev.evaluate_snapshot(_snapshot(coords))
    assert row["ligand_pocket_dist_le_6a"] == 0.0
    assert row["ligand_contacts_ge_5"] == 0.0
    assert row["ligand_hbonds_ge_1"] == 0.0
    assert pass_fraction_from_gpu_row(row) < 0.5


def test_gpu_pass_fraction_traj_fn() -> None:
    ev = GPUPoseBustersEvaluator(_fake_structure(), 7, _reference_coords())
    fn = make_posebusters_gpu_pass_fraction_traj_fn(ev)
    out = fn([_snapshot(_reference_coords())])
    assert 0.0 <= out <= 1.0


def test_gpu_cached_column_scalar_fns_single_eval_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = GPUPoseBustersEvaluator(_fake_structure(), 7, _reference_coords())
    calls = {"n": 0}
    original = ev.evaluate_snapshot

    def _wrapped(snapshot):
        calls["n"] += 1
        return original(snapshot)

    monkeypatch.setattr(ev, "evaluate_snapshot", _wrapped)
    cols = gpu_check_columns()[:2]
    fns = make_posebusters_gpu_cached_column_scalar_fns(ev, cols)
    traj = [_snapshot(_reference_coords())]
    assert fns[0](traj) in (0.0, 1.0)
    assert fns[1](traj) in (0.0, 1.0)
    assert calls["n"] == 1


def test_gpu_hybrid_cpu_fallback_runs_on_cadence() -> None:
    class _CpuEval:
        def __init__(self) -> None:
            self.calls = 0

        def bust_row(self, coords):
            self.calls += 1
            return {"cpu": float(coords.shape[0])}

    cpu = _CpuEval()
    ev = GPUPoseBustersEvaluator(
        _fake_structure(),
        7,
        _reference_coords(),
        backend_mode="hybrid",
        cpu_evaluator=cpu,
        cpu_fallback_every=2,
    )
    ev.evaluate_snapshot(_snapshot(_reference_coords()))
    assert cpu.calls == 0
    ev.evaluate_snapshot(_snapshot(_reference_coords()))
    assert cpu.calls == 1


def test_expand_posebusters_gpu_names_round_trip() -> None:
    names, raw = expand_posebusters_gpu_all_to_cv_names()
    assert names == [f"{POSEBUSTERS_GPU_CV_PREFIX}{name}" for name in raw]

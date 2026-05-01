"""Tests for the GPU-native distributed OneOPES REPEX scaffolding."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from genai_tps.simulation import repex_distributed as rd
from genai_tps.simulation.oneopes_repex import ExchangeAttempt


class _StepSim:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def step(self, n_steps: int) -> None:
        self.steps.append(int(n_steps))


def _attempt(accepted: bool = True) -> ExchangeAttempt:
    return ExchangeAttempt(
        md_step=10,
        u00=0.0,
        u01=0.0,
        u10=0.0,
        u11=0.0,
        log_accept=0.0,
        accepted=accepted,
        rng_uniform=0.5,
    )


def test_assign_replica_groups_allows_multiple_replicas_per_gpu() -> None:
    devices, replica_to_rank, rank_to_replicas = rd.assign_replica_groups([0, 0, 1, 1, 3])

    assert devices == [0, 1, 3]
    assert replica_to_rank == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    assert rank_to_replicas == {0: [0, 1], 1: [2, 3], 2: [4]}


def test_gpu_replica_group_steps_local_replicas_in_parallel() -> None:
    group = rd.GPUReplicaGroup(
        rank=0,
        gpu_id=None,
        replica_indices=[0, 1],
        rng=random.Random(0),
    )
    sim0 = _StepSim()
    sim1 = _StepSim()
    group.sims = {0: sim0, 1: sim1}

    group.step_all(7)

    assert sim0.steps == [7]
    assert sim1.steps == [7]


def test_local_exchange_swaps_positions_and_velocities(monkeypatch: pytest.MonkeyPatch) -> None:
    group = rd.GPUReplicaGroup(
        rank=0,
        gpu_id=None,
        replica_indices=[0, 1],
        rng=random.Random(0),
    )
    state: dict[tuple[str, int], str] = {
        ("pos", 0): "pos0",
        ("pos", 1): "pos1",
        ("vel", 0): "vel0",
        ("vel", 1): "vel1",
    }
    group.get_positions = lambda replica: state[("pos", int(replica))]  # type: ignore[method-assign]
    group.get_velocities = lambda replica: state[("vel", int(replica))]  # type: ignore[method-assign]
    group.set_positions = lambda replica, value: state.__setitem__(("pos", int(replica)), value)  # type: ignore[method-assign]
    group.set_velocities = lambda replica, value: state.__setitem__(("vel", int(replica)), value)  # type: ignore[method-assign]
    group.get_bias_energy = lambda replica: 0.0  # type: ignore[method-assign]
    group.evaluate_on_positions = lambda replica, positions: 0.0  # type: ignore[method-assign]
    monkeypatch.setattr(
        rd,
        "metropolis_accept_two_replica_hrex",
        lambda *args, **kwargs: (True, 0.0, 0.25),
    )

    attempt = group.local_exchange(0, 1, md_step=10)

    assert attempt.accepted is True
    assert state[("pos", 0)] == "pos1"
    assert state[("pos", 1)] == "pos0"
    assert state[("vel", 0)] == "vel1"
    assert state[("vel", 1)] == "vel0"


def test_distributed_exchange_step_uses_local_path_for_same_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = rd.GPUReplicaGroup(rank=0, gpu_id=None, replica_indices=[0, 1])
    calls: list[tuple[int, int]] = []

    def _local_exchange(i: int, j: int, *, md_step: int = 0) -> ExchangeAttempt:
        calls.append((i, j))
        return _attempt()

    group.local_exchange = _local_exchange  # type: ignore[method-assign]

    attempts = rd.distributed_exchange_step(
        group,
        replica_to_rank={0: 0, 1: 0, 2: 1},
        phase=0,
        n_replicas=3,
        md_step=10,
    )

    assert calls == [(0, 1)]
    assert [(i, j) for i, j, _ in attempts] == [(0, 1)]


def test_setup_per_replica_logging_creates_rank_and_replica_logs(tmp_path: Path) -> None:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        rd.setup_per_replica_logging(0, tmp_path, [0, 2])
        print("hello from rank zero")
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    assert (tmp_path / "rank000.log").is_file()
    assert (tmp_path / "rank000.err").is_file()
    assert (tmp_path / "rep000").is_dir()
    assert (tmp_path / "rep002").is_dir()
    assert "hello from rank zero" in (tmp_path / "rank000.log").read_text(encoding="utf-8")


def test_cpu_reference_position_fallback_returns_cpu_tensor() -> None:
    torch = pytest.importorskip("torch")
    mm = pytest.importorskip("openmm")
    unit = pytest.importorskip("openmm.unit")

    system = mm.System()
    system.addParticle(12.0)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)
    context.setPositions([mm.Vec3(1.0, 2.0, 3.0)] * unit.nanometer)
    try:
        tensor = rd._torch_tensor_from_positions(context)
    finally:
        del context
        del integrator

    assert isinstance(tensor, torch.Tensor)
    assert not tensor.is_cuda
    assert tuple(tensor.shape) == (1, 3)

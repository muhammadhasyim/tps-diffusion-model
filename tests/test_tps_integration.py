"""Integration: OPS PathSampling + BoltzDiffusionEngine + mock diffusion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.bridge import snapshot_from_gpu
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from genai_tps.backends.boltz.tps_sampling import (
    assert_trajectory_in_ensemble,
    build_fixed_length_tps_network,
    run_tps_path_sampling,
    sigma_tps_state_volumes,
    tps_ensemble,
)

from tests.mock_boltz_diffusion import MockDiffusion


@pytest.fixture
def core_engine_and_traj():
    diff = MockDiffusion()
    b, m = 1, 4
    atom_mask = torch.ones(b, m)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    engine = BoltzDiffusionEngine(
        core,
        boltz_snapshot_descriptor(n_atoms=m),
        options={"n_frames_max": 20},
    )
    x = core.sample_initial_noise()
    snaps = [
        snapshot_from_gpu(
            x,
            0,
            None,
            None,
            None,
            float(core.schedule[0].sigma_tm),
            None,
        )
    ]
    for step in range(3):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            snapshot_from_gpu(
                x,
                step + 1,
                eps,
                rr,
                tr,
                float(meta["sigma_t"]),
                meta.get("center_mean"),
            )
        )
    traj = Trajectory(snaps)
    return core, engine, traj


def test_sigma_volumes_contain_endpoints(core_engine_and_traj):
    core, _, traj = core_engine_and_traj
    sa, sb = sigma_tps_state_volumes(core)
    net = build_fixed_length_tps_network(sa, sb, len(traj))
    ens = tps_ensemble(net)
    assert_trajectory_in_ensemble(traj, ens, label="mock forward path")
    assert ens(traj, candidate=True)


def test_run_tps_path_sampling_accepts_moves(core_engine_and_traj):
    _, engine, traj = core_engine_and_traj
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        final_traj, log_entries = run_tps_path_sampling(engine, traj, n_rounds=5, log_path=log)
    assert len(final_traj) == len(traj)
    assert len(log_entries) == 5
    for e in log_entries:
        assert "accepted" in e
        assert e["step"] >= 1
        p = e.get("metropolis_acceptance")
        if p is not None:
            assert 0.0 <= float(p) <= 1.0
        m1 = e.get("min_1_r")
        if m1 is not None:
            assert 0.0 <= float(m1) <= 1.0
    assert any(e["accepted"] for e in log_entries)


def test_tps_long_run_backward_shoot_not_always_rejected(core_engine_and_traj):
    """With corrected backward kernel + bias, some MC steps should accept (not 100% reject)."""
    torch.manual_seed(42)
    _, engine, traj = core_engine_and_traj
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        _, log_entries = run_tps_path_sampling(engine, traj, n_rounds=120, log_path=log)
    assert any(e["accepted"] for e in log_entries)


def test_shooting_log_written(core_engine_and_traj):
    _, engine, traj = core_engine_and_traj
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        run_tps_path_sampling(engine, traj, n_rounds=2, log_path=log)
        text = log.read_text()
    assert "step" in text
    assert "accepted" in text


def test_periodic_callback_every(core_engine_and_traj):
    _, engine, traj = core_engine_and_traj
    seen: list[tuple[int, int]] = []

    def cb(step: int, t) -> None:
        seen.append((step, len(t)))

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        run_tps_path_sampling(
            engine,
            traj,
            n_rounds=5,
            log_path=log,
            periodic_callback=cb,
            periodic_every=2,
        )
    assert seen == [(2, len(traj)), (4, len(traj))]


def test_periodic_callbacks_second_interval(core_engine_and_traj):
    """Additive periodic_callbacks run on their own schedule alongside periodic_callback."""
    _, engine, traj = core_engine_and_traj
    seen_main: list[int] = []
    seen_extra: list[int] = []

    def cb_main(step: int, t) -> None:
        seen_main.append(step)

    def cb_extra(step: int, t) -> None:
        seen_extra.append(step)

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        run_tps_path_sampling(
            engine,
            traj,
            n_rounds=6,
            log_path=log,
            periodic_callback=cb_main,
            periodic_every=2,
            periodic_callbacks=[(cb_extra, 3)],
        )
    assert seen_main == [2, 4, 6]
    assert seen_extra == [3, 6]


def test_periodic_step_callbacks_receive_step_entry(core_engine_and_traj):
    """periodic_step_callbacks get (mc_step, entry) matching step_log rows."""
    _, engine, traj = core_engine_and_traj
    captured: list[tuple[int, dict]] = []

    def cb_step(mc_step: int, entry: dict) -> None:
        captured.append((mc_step, dict(entry)))

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "s.log"
        _, log_entries = run_tps_path_sampling(
            engine,
            traj,
            n_rounds=5,
            log_path=log,
            periodic_step_callbacks=[(cb_step, 2)],
        )
    assert len(captured) == 2
    assert captured[0][0] == 2 and captured[1][0] == 4
    for mc_step, entry in captured:
        assert entry == log_entries[mc_step - 1]
        assert entry["step"] == mc_step

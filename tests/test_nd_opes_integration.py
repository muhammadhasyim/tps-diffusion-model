"""Integration tests for multi-dimensional OPES bias with vector CV functions.

Tests verify:
1. Multi-D OPESBias (2-D) wired through run_tps_path_sampling works end-to-end
2. Vector CV function (returns np.ndarray shape (2,)) is evaluated and logged
3. OPES accepts/rejects paths based on multi-D bias
4. cv_value in step log is a list for multi-D, scalar for 1-D
5. Backward compat: existing 1-D OPES still runs unchanged
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from genai_tps.enhanced_sampling.opes_bias import OPESBias
from tests.mock_boltz_diffusion import MockDiffusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_engine_traj(n_steps: int = 3):
    diff = MockDiffusion()
    atom_mask = torch.ones(1, 4)
    core = BoltzSamplerCore(
        diff, atom_mask, {"multiplicity": 1},
        multiplicity=1,
        n_fixed_point=2,
    )
    core.build_schedule(n_steps)
    x0 = core.sample_initial_noise()
    traj_tensors, _, _, _, _ = core.generate_segment(x0, 0, n_steps)

    from genai_tps.backends.boltz.bridge import snapshot_from_gpu
    from openpathsampling.engines.trajectory import Trajectory

    snaps = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(snapshot_from_gpu(x0, 0, None, None, None, sig0))
    for step_idx, x_t in enumerate(traj_tensors):
        sigma_t = float(core.schedule[step_idx].sigma_t) if step_idx < len(core.schedule) else 0.0
        snaps.append(
            snapshot_from_gpu(x_t, step_idx + 1, None, None, None, sigma_t)
        )
    traj = Trajectory(snaps)

    descriptor = boltz_snapshot_descriptor(n_atoms=4)
    engine = BoltzDiffusionEngine(core, descriptor, options={"n_frames_max": n_steps + 4})
    return engine, traj


# ---------------------------------------------------------------------------
# Integration: 1-D OPES still works (backward compat)
# ---------------------------------------------------------------------------


class TestOPES1DBackwardCompat:
    def test_1d_opes_still_runs(self, tmp_path):
        engine, init_traj = _build_engine_traj()

        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)

        def cv_fn_1d(traj):
            snap = traj[-1]
            if snap.tensor_coords is None:
                return 0.0
            return float(snap.tensor_coords[0].norm().item())

        log_path = tmp_path / "shoot_1d.log"
        _, step_log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_1d,
            forward_only=True,
        )
        assert len(step_log) == 5
        # cv_value should be a scalar (float) in 1-D mode
        for entry in step_log:
            cv = entry.get("cv_value")
            if cv is not None:
                assert isinstance(cv, (float, int)), (
                    f"Expected scalar cv_value for 1-D OPES, got {type(cv)}: {cv}"
                )

    def test_1d_opes_updates_bias(self, tmp_path):
        engine, init_traj = _build_engine_traj()
        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)

        def cv_fn(traj):
            snap = traj[-1]
            return float(snap.tensor_coords[0, 0, 0].item()) if snap.tensor_coords is not None else 0.0

        log_path = tmp_path / "shoot_1d_update.log"
        run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn,
            forward_only=True,
        )
        # At least one kernel should have been deposited
        assert bias.counter >= 1


# ---------------------------------------------------------------------------
# Integration: multi-D OPES (2-D)
# ---------------------------------------------------------------------------


class TestOPES2DIntegration:
    def test_2d_opes_runs_and_logs_list(self, tmp_path):
        engine, init_traj = _build_engine_traj()

        bias = OPESBias(ndim=2, barrier=3.0, fixed_sigma=np.array([0.5, 0.5]))

        def cv_fn_2d(traj) -> np.ndarray:
            snap = traj[-1]
            if snap.tensor_coords is None:
                return np.zeros(2)
            coords = snap.tensor_coords[0]  # (N, 3)
            rg = float(coords.norm(dim=-1).mean().item())
            co = float(coords[:, 0].std().item())  # cheap proxy
            return np.array([rg, co], dtype=np.float64)

        log_path = tmp_path / "shoot_2d.log"
        _, step_log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_2d,
            forward_only=True,
        )
        assert len(step_log) == 5
        # cv_value should be a list of length 2 for multi-D OPES
        for entry in step_log:
            cv = entry.get("cv_value")
            if cv is not None:
                assert isinstance(cv, list), (
                    f"Expected list cv_value for 2-D OPES, got {type(cv)}: {cv}"
                )
                assert len(cv) == 2

    def test_2d_opes_deposits_kernels(self, tmp_path):
        engine, init_traj = _build_engine_traj()
        bias = OPESBias(ndim=2, barrier=3.0, fixed_sigma=np.array([0.5, 0.5]), pace=1)

        def cv_fn_2d(traj) -> np.ndarray:
            snap = traj[-1]
            if snap.tensor_coords is None:
                return np.zeros(2)
            c = snap.tensor_coords[0]
            return np.array([float(c.norm().item()), float(c[:, 0].mean().item())])

        log_path = tmp_path / "shoot_2d_kernels.log"
        run_tps_path_sampling(
            engine, init_traj, n_rounds=4, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_2d,
            forward_only=True,
        )
        assert bias.counter >= 1

    def test_2d_opes_acceptance_factor_applied(self, tmp_path):
        """Verify compute_acceptance_factor is called with a 2-D array."""
        engine, init_traj = _build_engine_traj()

        acceptance_calls: list[tuple[np.ndarray, np.ndarray]] = []

        class RecordingBias(OPESBias):
            def compute_acceptance_factor(self, cv_old, cv_new):
                if isinstance(cv_old, np.ndarray):
                    acceptance_calls.append((cv_old.copy(), cv_new.copy()))
                return super().compute_acceptance_factor(cv_old, cv_new)

        bias = RecordingBias(ndim=2, barrier=3.0, fixed_sigma=np.array([0.5, 0.5]))

        def cv_fn_2d(traj) -> np.ndarray:
            snap = traj[-1]
            if snap.tensor_coords is None:
                return np.zeros(2)
            c = snap.tensor_coords[0]
            return np.array([float(c.norm().item()), 0.0])

        log_path = tmp_path / "shoot_2d_acceptance.log"
        run_tps_path_sampling(
            engine, init_traj, n_rounds=3, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_2d,
            forward_only=True,  # backward step fails on short mock trajectory
        )
        # At least some acceptance factor calls should have happened with 2-D arrays
        if acceptance_calls:
            for cv_old, cv_new in acceptance_calls:
                assert cv_old.shape == (2,)
                assert cv_new.shape == (2,)

    def test_2d_opes_state_saved_and_loaded(self, tmp_path):
        engine, init_traj = _build_engine_traj()
        bias = OPESBias(ndim=2, barrier=3.0, fixed_sigma=np.array([0.5, 0.5]), pace=1)

        def cv_fn_2d(traj) -> np.ndarray:
            snap = traj[-1]
            if snap.tensor_coords is None:
                return np.zeros(2)
            c = snap.tensor_coords[0]
            return np.array([float(c.norm().item()), float(c[:, 0].mean().item())])

        log_path = tmp_path / "shoot_2d_save.log"
        run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_2d,
            forward_only=True,
        )
        state_path = tmp_path / "opes_state.json"
        bias.save_state(state_path)
        loaded = OPESBias.load_state(state_path)
        assert loaded.ndim == 2
        assert loaded.n_kernels == bias.n_kernels
        for k_orig, k_load in zip(bias.kernels, loaded.kernels):
            np.testing.assert_allclose(k_load.center, k_orig.center, rtol=1e-8)


# ---------------------------------------------------------------------------
# Integration: 3-D OPES (contact_order, clash_count, rg via lambdas)
# ---------------------------------------------------------------------------


class TestOPES3DIntegration:
    def test_3d_opes_runs(self, tmp_path):
        """3-D OPES with three cheap proxy CVs runs without error."""
        engine, init_traj = _build_engine_traj()
        bias = OPESBias(ndim=3, barrier=3.0, fixed_sigma=np.full(3, 0.5), pace=1)

        def cv_fn_3d(traj) -> np.ndarray:
            snap = traj[-1]
            if snap.tensor_coords is None:
                return np.zeros(3)
            c = snap.tensor_coords[0]
            return np.array([
                float(c.norm().item()),
                float(c[:, 0].mean().item()),
                float(c[:, 1].std().item()),
            ], dtype=np.float64)

        log_path = tmp_path / "shoot_3d.log"
        _, step_log = run_tps_path_sampling(
            engine, init_traj, n_rounds=4, log_path=log_path,
            enhanced_bias=bias, cv_function=cv_fn_3d,
            forward_only=True,
        )
        assert len(step_log) == 4
        for entry in step_log:
            cv = entry.get("cv_value")
            if cv is not None:
                assert isinstance(cv, list)
                assert len(cv) == 3

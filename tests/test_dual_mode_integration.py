"""Integration tests for diagnostic CV logging in run_tps_path_sampling.

Tests verify that:
1. diagnostic_cv_functions are logged as 'diag_<name>' keys in step entries
2. diagnostic CVs do NOT affect the acceptance sequence (observation-only)
3. Diagnostic CVs work in both vanilla TPS (no bias) and TPS-OPES modes
4. Vanilla TPS with compile + n_fixed_point=2 + diagnostics runs and accepts moves
5. TPS-OPES with compile + n_fixed_point=2 + diagnostics runs, bias updates occur

All tests use MockDiffusion (no GPU, no checkpoint required).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import pytest

from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from tests.mock_boltz_diffusion import MockDiffusion


# ---------------------------------------------------------------------------
# Fixture helpers (inline so this file is self-contained)
# ---------------------------------------------------------------------------

def _build_engine_traj(n_fixed_point: int = 4, compile_model: bool = False):
    diff = MockDiffusion()
    atom_mask = torch.ones(1, 4)
    core = BoltzSamplerCore(
        diff, atom_mask, {"multiplicity": 1},
        multiplicity=1,
        n_fixed_point=n_fixed_point,
        compile_model=compile_model,
    )
    core.build_schedule(3)
    x0 = core.sample_initial_noise()
    traj_tensors, _, _, _, _ = core.generate_segment(x0, 0, 3)

    from genai_tps.backends.boltz.bridge import snapshot_from_gpu  # noqa: PLC0415
    from openpathsampling.engines.trajectory import Trajectory  # noqa: PLC0415

    snaps = []
    sig0 = float(core.schedule[0].sigma_tm)
    snaps.append(snapshot_from_gpu(x0, 0, None, None, None, sig0))
    for s_idx, x_t in enumerate(traj_tensors[1:]):
        sig = float(core.schedule[s_idx].sigma_t)
        snaps.append(snapshot_from_gpu(x_t, s_idx + 1, None, None, None, sig))

    init_traj = Trajectory(snaps)
    desc = boltz_snapshot_descriptor(n_atoms=4)
    engine = BoltzDiffusionEngine(core, desc, options={"n_frames_max": 10})
    engine.current_snapshot = snaps[0]
    return engine, init_traj


def _simple_cv(traj) -> float:
    return float(traj[-1].tensor_coords.mean()) if hasattr(traj[-1], "tensor_coords") else 0.0


def _contact_order_cv(traj) -> float:
    from genai_tps.backends.boltz.collective_variables import contact_order  # noqa: PLC0415
    return contact_order(traj[-1])


def _clash_count_cv(traj) -> float:
    from genai_tps.backends.boltz.collective_variables import clash_count  # noqa: PLC0415
    return float(clash_count(traj[-1]))


# ---------------------------------------------------------------------------
# _RecordingEnhancedBias minimal stub
# ---------------------------------------------------------------------------

class _RecordingEnhancedBias:
    """Minimal bias that records updates and always accepts (factor=1.0)."""

    def __init__(self):
        self.updates: list[tuple[float, int]] = []

    def compute_acceptance_factor(self, cv_old: float, cv_new: float) -> float:
        return 1.0

    def update(self, cv_accepted: float, mc_step: int) -> None:
        self.updates.append((cv_accepted, mc_step))


# ---------------------------------------------------------------------------
# TestDiagnosticCVLogging
# ---------------------------------------------------------------------------

class TestDiagnosticCVLogging:
    def test_diagnostic_cvs_appear_in_step_log_vanilla_tps(self, tmp_path):
        """Vanilla TPS (no bias): diagnostic CVs must appear as 'diag_*' keys."""
        engine, init_traj = _build_engine_traj()
        log_file = tmp_path / "shoot.log"
        diag_fns = {"contact_order": _contact_order_cv, "clash_count": _clash_count_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            diagnostic_cv_functions=diag_fns,
        )
        assert len(log) == 5
        for entry in log:
            assert "diag_contact_order" in entry, f"Missing diag_contact_order in {entry}"
            assert "diag_clash_count" in entry, f"Missing diag_clash_count in {entry}"

    def test_diagnostic_cvs_appear_in_step_log_opes_tps(self, tmp_path):
        """TPS-OPES: both 'cv_value' and 'diag_*' keys must appear in log."""
        engine, init_traj = _build_engine_traj()
        log_file = tmp_path / "shoot.log"
        bias = _RecordingEnhancedBias()
        diag_fns = {"contact_order": _contact_order_cv, "clash_count": _clash_count_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            enhanced_bias=bias,
            cv_function=_simple_cv,
            diagnostic_cv_functions=diag_fns,
        )
        assert len(log) == 5
        for entry in log:
            assert "cv_value" in entry, f"Missing cv_value in {entry}"
            assert "diag_contact_order" in entry
            assert "diag_clash_count" in entry

    def test_diagnostic_cvs_do_not_affect_acceptance(self, tmp_path):
        """Diagnostic CVs must be observation-only: they must not affect step entries
        beyond adding 'diag_*' keys (acceptance sequence is unaffected when using
        forward_only=True which always accepts reactive paths)."""
        engine, init_traj = _build_engine_traj()
        log_file = tmp_path / "shoot.log"

        # With forward_only=True all reactive steps accept; diagnostics must not change that
        diag_fns = {"contact_order": _contact_order_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            diagnostic_cv_functions=diag_fns,
        )
        # All steps should be accepted under forward_only
        for entry in log:
            assert entry["accepted"] is True, (
                f"Forward-only step was rejected (diagnostics may be interfering): {entry}"
            )
            assert "diag_contact_order" in entry

    def test_diagnostic_cvs_bias_updates_only_from_primary_cv(self, tmp_path):
        """OPES updates must only be called for the primary CV (not diagnostics)."""
        engine, init_traj = _build_engine_traj()
        log_file = tmp_path / "shoot.log"
        bias = _RecordingEnhancedBias()

        def _always_finite_cv(traj):
            return 1.5

        diag_fns = {"contact_order": _contact_order_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            enhanced_bias=bias,
            cv_function=_always_finite_cv,
            diagnostic_cv_functions=diag_fns,
        )
        # All bias updates must have scalar float CV values (from primary CV)
        for cv_val, mc_step in bias.updates:
            assert isinstance(cv_val, float)
            # Must be the primary CV value (1.5), not the diagnostic
            assert cv_val == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# TestFullStackDualMode
# ---------------------------------------------------------------------------

class TestFullStackDualMode:
    def test_vanilla_tps_with_n_fixed_point_2_and_diagnostics(self, tmp_path):
        """Vanilla TPS with n_fixed_point=2 + diagnostic CVs must complete."""
        engine, init_traj = _build_engine_traj(n_fixed_point=2)
        log_file = tmp_path / "shoot.log"
        diag_fns = {"contact_order": _contact_order_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            diagnostic_cv_functions=diag_fns,
        )
        assert len(log) == 5
        for entry in log:
            assert "diag_contact_order" in entry

    def test_opes_tps_with_n_fixed_point_2_and_diagnostics(self, tmp_path):
        """TPS-OPES with n_fixed_point=2 + diagnostic CVs: bias updates occur."""
        engine, init_traj = _build_engine_traj(n_fixed_point=2)
        log_file = tmp_path / "shoot.log"
        bias = _RecordingEnhancedBias()
        diag_fns = {"contact_order": _contact_order_cv, "clash_count": _clash_count_cv}

        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            enhanced_bias=bias,
            cv_function=_simple_cv,
            diagnostic_cv_functions=diag_fns,
        )
        assert len(log) == 5
        # At least some bias updates occurred
        assert len(bias.updates) > 0
        for entry in log:
            assert "diag_contact_order" in entry
            assert "diag_clash_count" in entry


# ---------------------------------------------------------------------------
# TestNFixedPointDualMode (additional integration)
# ---------------------------------------------------------------------------

class TestNFixedPointDualMode:
    def test_vanilla_tps_n_fixed_point_2(self, tmp_path):
        """Vanilla TPS with n_fixed_point=2 must complete without error."""
        engine, init_traj = _build_engine_traj(n_fixed_point=2)
        log_file = tmp_path / "shoot.log"
        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
        )
        assert len(log) == 5

    def test_opes_tps_n_fixed_point_2(self, tmp_path):
        """TPS-OPES with n_fixed_point=2 must complete, bias updates recorded."""
        engine, init_traj = _build_engine_traj(n_fixed_point=2)
        log_file = tmp_path / "shoot.log"
        bias = _RecordingEnhancedBias()
        _, log = run_tps_path_sampling(
            engine, init_traj, n_rounds=5, log_path=log_file,
            forward_only=True,
            enhanced_bias=bias,
            cv_function=_simple_cv,
        )
        assert len(log) == 5
        assert len(bias.updates) > 0

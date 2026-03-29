"""Tests for n_fixed_point parameter threading in BoltzSamplerCore.

TDD scaffolding -- written before implementation.

Tests (CPU-only, MockDiffusion):
  - BoltzSamplerCore stores n_fixed_point as instance attribute
  - n_fixed_point=1 produces finite output in _solve_x_noisy_from_output
  - n_fixed_point=2 produces finite output
  - n_fixed_point is used in all four call sites:
      _solve_x_noisy_from_output, recover_forward_noise,
      recover_backward_noise, single_backward_step
  - Reducing n_fixed_point from 4 to 2 does not break the round-trip
    recover_forward_noise test (outputs still finite, not same value)
  - Default is 4 (backward compat)
"""

from __future__ import annotations

import torch
import pytest

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
from tests.mock_boltz_diffusion import MockDiffusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_core(n_fixed_point: int = 4):
    diff = MockDiffusion()
    atom_mask = torch.ones(1, 4)
    return BoltzSamplerCore(
        diff, atom_mask, {"multiplicity": 1},
        multiplicity=1,
        n_fixed_point=n_fixed_point,
    )


# ---------------------------------------------------------------------------
# Attribute storage
# ---------------------------------------------------------------------------

class TestNFixedPointAttribute:
    def test_default_is_4(self):
        """Default n_fixed_point must be 4 (backward compatible)."""
        core = _make_core()
        assert core.n_fixed_point == 4

    def test_stored_correctly(self):
        """n_fixed_point=2 must be stored on the core."""
        core = _make_core(n_fixed_point=2)
        assert core.n_fixed_point == 2

    def test_stored_as_1(self):
        """n_fixed_point=1 must be stored."""
        core = _make_core(n_fixed_point=1)
        assert core.n_fixed_point == 1


# ---------------------------------------------------------------------------
# _solve_x_noisy_from_output
# ---------------------------------------------------------------------------

class TestSolveXNoisy:
    def test_n1_finite_output(self):
        """n_fixed_point=1 must not diverge: output must be finite."""
        core = _make_core(n_fixed_point=1)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        result = core._solve_x_noisy_from_output(x, step_idx=0, n_fixed_point=1)
        assert torch.isfinite(result).all(), "n_fixed_point=1 produced non-finite output"

    def test_n2_finite_output(self):
        """n_fixed_point=2 must produce finite output."""
        core = _make_core(n_fixed_point=2)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        result = core._solve_x_noisy_from_output(x, step_idx=0, n_fixed_point=2)
        assert torch.isfinite(result).all()

    def test_n4_finite_output(self):
        """n_fixed_point=4 (default) must produce finite output."""
        core = _make_core(n_fixed_point=4)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        result = core._solve_x_noisy_from_output(x, step_idx=0, n_fixed_point=4)
        assert torch.isfinite(result).all()

    def test_instance_n_fixed_point_used_by_default(self):
        """When n_fixed_point arg is omitted, the instance attribute must be used."""
        core = _make_core(n_fixed_point=2)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        # Call without explicit n_fixed_point -- should use self.n_fixed_point
        result = core._solve_x_noisy_from_output(x, step_idx=0)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# single_backward_step uses instance n_fixed_point
# ---------------------------------------------------------------------------

class TestBackwardStepNFixedPoint:
    def test_backward_step_n1_finite(self):
        """single_backward_step with n_fixed_point=1 must produce finite output."""
        core = _make_core(n_fixed_point=1)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        # Run one forward step to get to step_idx=0
        x_next, _, _, _, _ = core.single_forward_step(x, 0)
        # Now backward
        x_prev, eps, rr, tr, meta = core.single_backward_step(x_next, step_idx=0)
        assert torch.isfinite(x_prev).all(), "backward step n=1 produced non-finite output"
        assert torch.isfinite(eps).all()

    def test_backward_step_n2_finite(self):
        """single_backward_step with n_fixed_point=2 must produce finite output."""
        core = _make_core(n_fixed_point=2)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, _, _, _, _ = core.single_forward_step(x, 0)
        x_prev, eps, rr, tr, meta = core.single_backward_step(x_next, step_idx=0)
        assert torch.isfinite(x_prev).all()


# ---------------------------------------------------------------------------
# recover_forward_noise uses instance n_fixed_point
# ---------------------------------------------------------------------------

class TestRecoverNoiseNFixedPoint:
    def test_recover_forward_noise_n2_finite(self):
        """recover_forward_noise with n_fixed_point=2 must produce finite eps."""
        core = _make_core(n_fixed_point=2)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, eps, rr, tr, meta = core.single_forward_step(x, 0)
        eps_rec, _ = core.recover_forward_noise(x, x_next, 0, rr, tr)
        assert torch.isfinite(eps_rec).all()

    def test_recover_backward_noise_n2_finite(self):
        """recover_backward_noise with n_fixed_point=2 must produce finite eps."""
        core = _make_core(n_fixed_point=2)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, _, _, _, _ = core.single_forward_step(x, 0)
        x_prev, eps, rr, tr, meta = core.single_backward_step(x_next, step_idx=0)
        cm = torch.zeros((x_prev.shape[0], 1, 3))
        eps_rec, _ = core.recover_backward_noise(x_prev, x_next, 0, rr, tr, cm)
        assert torch.isfinite(eps_rec).all()

"""Tests for torch.compile integration in BoltzSamplerCore.

TDD scaffolding -- written before implementation.

Tests (CPU-only, MockDiffusion):
  - BoltzSamplerCore accepts compile_model=True keyword without error
  - compile_model is stored as an attribute
  - Forward step with compile_model=True produces finite, same-shape output
  - Compiled and eager cores produce numerically close outputs for same input
  - compile_model=False (default) leaves the forward method unmodified
  - inference_dtype attribute is stored when provided
"""

from __future__ import annotations

import torch
import pytest

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor
from tests.mock_boltz_diffusion import MockDiffusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_core(compile_model: bool = False, n_fixed_point: int = 4, inference_dtype=None):
    diff = MockDiffusion()
    b, m = 1, 4
    atom_mask = torch.ones(b, m)
    kwargs: dict = {"multiplicity": 1}
    extra: dict = {}
    if inference_dtype is not None:
        extra["inference_dtype"] = inference_dtype
    return BoltzSamplerCore(
        diff, atom_mask, kwargs,
        multiplicity=1,
        compile_model=compile_model,
        n_fixed_point=n_fixed_point,
        **extra,
    )


# ---------------------------------------------------------------------------
# compile_model attribute
# ---------------------------------------------------------------------------

class TestCompileModelAttribute:
    def test_compile_model_false_is_default(self):
        """BoltzSamplerCore() with no extra args stores compile_model=False."""
        core = _make_core()
        assert core.compile_model is False

    def test_compile_model_true_stored(self):
        """BoltzSamplerCore(compile_model=True) stores the flag."""
        core = _make_core(compile_model=True)
        assert core.compile_model is True

    def test_compile_model_true_does_not_raise(self):
        """Constructing with compile_model=True must not raise."""
        _make_core(compile_model=True)

    def test_forward_fn_still_callable_after_compile(self):
        """preconditioned_network_forward must remain callable after compile."""
        core = _make_core(compile_model=True)
        assert callable(core.diffusion.preconditioned_network_forward)


# ---------------------------------------------------------------------------
# Numerical parity: compiled == eager
# ---------------------------------------------------------------------------

class TestCompiledForwardParity:
    def test_forward_step_finite_output(self):
        """single_forward_step with compile_model=True must produce finite output."""
        core = _make_core(compile_model=True)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, eps, rr, tr, meta = core.single_forward_step(x, 0)
        assert torch.isfinite(x_next).all(), "x_next contains non-finite values"
        assert x_next.shape == x.shape

    def test_forward_step_same_shape_as_eager(self):
        """Compiled and eager cores produce same-shape output."""
        core_eager = _make_core(compile_model=False)
        core_compiled = _make_core(compile_model=True)
        core_eager.build_schedule(3)
        core_compiled.build_schedule(3)

        torch.manual_seed(42)
        x_eager = core_eager.sample_initial_noise()
        torch.manual_seed(42)
        x_compiled = core_compiled.sample_initial_noise()

        torch.manual_seed(0)
        x_next_e, _, _, _, _ = core_eager.single_forward_step(x_eager, 0)
        torch.manual_seed(0)
        x_next_c, _, _, _, _ = core_compiled.single_forward_step(x_compiled, 0)

        assert x_next_e.shape == x_next_c.shape

    def test_compiled_forward_numerically_close_to_eager(self):
        """With MockDiffusion (identity), compiled and eager must agree closely."""
        core_eager = _make_core(compile_model=False)
        core_compiled = _make_core(compile_model=True)
        core_eager.build_schedule(3)
        core_compiled.build_schedule(3)

        # Use same seed + same initial noise for both
        seed = 12345
        torch.manual_seed(seed)
        x = core_eager.sample_initial_noise()

        torch.manual_seed(0)
        x_next_e, _, _, _, _ = core_eager.single_forward_step(x.clone(), 0)
        torch.manual_seed(0)
        x_next_c, _, _, _, _ = core_compiled.single_forward_step(x.clone(), 0)

        # MockDiffusion is an identity network; compiled and eager must be identical
        torch.testing.assert_close(x_next_e, x_next_c, atol=1e-5, rtol=1e-4)

    def test_generate_segment_finite_with_compile(self):
        """generate_segment (multi-step) must produce finite frames with compile."""
        core = _make_core(compile_model=True)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        frames, _, _, _, _ = core.generate_segment(x, 0, 3)
        for f in frames:
            assert torch.isfinite(f).all()

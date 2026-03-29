"""Tests for bfloat16/inference_dtype in BoltzSamplerCore.

TDD scaffolding -- written before implementation.

Tests (CPU-only):
  - BoltzSamplerCore accepts inference_dtype=torch.float32 without error
  - With MockDiffusion on CPU, inference_dtype is stored as an attribute
  - single_forward_step still produces finite float32 output when inference_dtype=torch.float32
  - inference_dtype=None (default) leaves behavior unchanged
  - flag threads from gpu_core to instance attribute correctly
  - (GPU-only, skipped on CPU) autocast with bfloat16 produces same-shape, finite output
"""

from __future__ import annotations

import torch
import pytest

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from tests.mock_boltz_diffusion import MockDiffusion


def _make_core(inference_dtype=None):
    diff = MockDiffusion()
    atom_mask = torch.ones(1, 4)
    kwargs: dict = {}
    if inference_dtype is not None:
        kwargs["inference_dtype"] = inference_dtype
    return BoltzSamplerCore(
        diff, atom_mask, {"multiplicity": 1},
        multiplicity=1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Attribute storage
# ---------------------------------------------------------------------------

class TestInferenceDtypeAttribute:
    def test_default_is_none(self):
        """Default inference_dtype must be None (no autocast)."""
        core = _make_core()
        assert core.inference_dtype is None

    def test_float32_stored(self):
        """inference_dtype=torch.float32 must be stored."""
        core = _make_core(inference_dtype=torch.float32)
        assert core.inference_dtype == torch.float32

    def test_constructor_does_not_raise_for_float32(self):
        """Constructing with inference_dtype=torch.float32 must not raise."""
        _make_core(inference_dtype=torch.float32)


# ---------------------------------------------------------------------------
# Forward step still finite with float32 dtype
# ---------------------------------------------------------------------------

class TestFloat32InferenceDtype:
    def test_forward_step_finite_with_float32_dtype(self):
        """inference_dtype=torch.float32 must not break forward step (CPU)."""
        core = _make_core(inference_dtype=torch.float32)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, eps, rr, tr, meta = core.single_forward_step(x, 0)
        assert torch.isfinite(x_next).all()
        assert x_next.shape == x.shape

    def test_backward_step_finite_with_float32_dtype(self):
        """inference_dtype=torch.float32 must not break backward step (CPU)."""
        core = _make_core(inference_dtype=torch.float32)
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, _, _, _, _ = core.single_forward_step(x, 0)
        x_prev, eps, rr, tr, meta = core.single_backward_step(x_next, step_idx=0)
        assert torch.isfinite(x_prev).all()

    def test_forward_parity_no_dtype_vs_float32(self):
        """inference_dtype=float32 must produce same results as no dtype (CPU/float32)."""
        core_default = _make_core()
        core_f32 = _make_core(inference_dtype=torch.float32)
        core_default.build_schedule(3)
        core_f32.build_schedule(3)

        torch.manual_seed(7)
        x = core_default.sample_initial_noise()
        torch.manual_seed(0)
        x_next_d, _, _, _, _ = core_default.single_forward_step(x.clone(), 0)
        torch.manual_seed(0)
        x_next_f, _, _, _, _ = core_f32.single_forward_step(x.clone(), 0)
        torch.testing.assert_close(x_next_d, x_next_f, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# GPU-only bfloat16 tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")
class TestBfloat16InferenceDtype:
    def test_bfloat16_stored(self):
        """inference_dtype=torch.bfloat16 must be stored on instance."""
        core = _make_core(inference_dtype=torch.bfloat16)
        assert core.inference_dtype == torch.bfloat16

    def test_forward_step_finite_bfloat16_cuda(self):
        """inference_dtype=bfloat16 on GPU must produce finite float32 output."""
        diff = MockDiffusion(device="cuda")
        core = BoltzSamplerCore(
            diff, torch.ones(1, 4),
            {"multiplicity": 1},
            multiplicity=1,
            inference_dtype=torch.bfloat16,
        )
        core.build_schedule(3)
        x = core.sample_initial_noise()
        x_next, eps, rr, tr, meta = core.single_forward_step(x, 0)
        assert torch.isfinite(x_next).all(), "bfloat16 forward step produced non-finite"
        assert x_next.dtype in (torch.float32, torch.bfloat16)

    def test_bfloat16_output_parity_within_tolerance(self):
        """bfloat16 and float32 forward passes must agree within bf16 precision."""
        diff_f32 = MockDiffusion(device="cuda")
        diff_bf16 = MockDiffusion(device="cuda")
        core_f32 = BoltzSamplerCore(
            diff_f32, torch.ones(1, 4), {"multiplicity": 1}, multiplicity=1,
        )
        core_bf16 = BoltzSamplerCore(
            diff_bf16, torch.ones(1, 4), {"multiplicity": 1}, multiplicity=1,
            inference_dtype=torch.bfloat16,
        )
        core_f32.build_schedule(3)
        core_bf16.build_schedule(3)

        torch.manual_seed(0)
        x = core_f32.sample_initial_noise()

        torch.manual_seed(1)
        x_next_f32, _, _, _, _ = core_f32.single_forward_step(x.clone(), 0)
        torch.manual_seed(1)
        x_next_bf16, _, _, _, _ = core_bf16.single_forward_step(x.clone(), 0)

        # bf16 precision allows larger tolerance
        x_next_bf16_f32 = x_next_bf16.float()
        torch.testing.assert_close(x_next_f32, x_next_bf16_f32, atol=1e-1, rtol=1e-1)

"""BoltzSamplerCore keeps tensors on the diffusion module device."""

from __future__ import annotations

import pytest
import torch

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.gpu_core import _kwargs_for_preconditioned_forward

from tests.mock_boltz_diffusion import MockDiffusion


@pytest.fixture
def core_on_device(device: torch.device) -> BoltzSamplerCore:
    diff = MockDiffusion(device)
    b, m = 1, 4
    atom_mask = torch.ones(b, m, device=device)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    return core


def test_initial_noise_on_device(core_on_device: BoltzSamplerCore, device: torch.device) -> None:
    x0 = core_on_device.sample_initial_noise()
    assert x0.device == device
    assert x0.shape == (1, 4, 3)


def test_forward_outputs_on_device(core_on_device: BoltzSamplerCore, device: torch.device) -> None:
    x = core_on_device.sample_initial_noise()
    xn, eps, rr, tr, _meta = core_on_device.single_forward_step(x, 0)
    assert xn.device == device
    assert eps.device == device
    assert rr.device == device
    assert tr.device == device


def test_backward_outputs_on_device(core_on_device: BoltzSamplerCore, device: torch.device) -> None:
    x = core_on_device.sample_initial_noise()
    x_prev, eps, rr, tr, _meta = core_on_device.single_backward_step(x, 0)
    assert x_prev.device == device
    assert eps.device == device
    assert rr.device == device
    assert tr.device == device


@pytest.mark.cuda
def test_forward_rejects_cpu_tensor_when_model_on_cuda(requires_cuda: torch.device) -> None:
    diff = MockDiffusion(requires_cuda)
    atom_mask = torch.ones(1, 4, device=requires_cuda)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    x_cpu = torch.randn(1, 4, 3, device="cpu")
    with pytest.raises(ValueError, match="diffusion device"):
        core.single_forward_step(x_cpu, 0)


@pytest.mark.cuda
def test_backward_rejects_cpu_tensor_when_model_on_cuda(requires_cuda: torch.device) -> None:
    diff = MockDiffusion(requires_cuda)
    atom_mask = torch.ones(1, 4, device=requires_cuda)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    x_cpu = torch.randn(1, 4, 3, device="cpu")
    with pytest.raises(ValueError, match="diffusion device"):
        core.single_backward_step(x_cpu, 0)


def test_preconditioned_kwargs_strip_steering_args() -> None:
    """``steering_args`` is for ``AtomDiffusion.sample``, not the score model ``forward``."""
    d = {"s_trunk": 1, "steering_args": {"fk_steering": False}}
    out = _kwargs_for_preconditioned_forward(d)
    assert "steering_args" not in out
    assert out["s_trunk"] == 1


def test_explicit_build_schedule_survives_sample_initial_noise() -> None:
    """Regression: checkpoint ``num_sampling_steps`` must not overwrite explicit ``build_schedule(n)``."""
    diff = MockDiffusion()
    diff.num_sampling_steps = 5
    atom_mask = torch.ones(1, 4)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(10)
    assert len(core.schedule) == 10
    x = core.sample_initial_noise()
    assert len(core.schedule) == 10
    assert core.num_sampling_steps == 10
    for step in range(10):
        x, *_ = core.single_forward_step(x, step)
    assert x.shape == (1, 4, 3)

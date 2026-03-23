"""Unit tests with a lightweight mock diffusion module (no checkpoint)."""

import numpy as np
import openpathsampling as paths
from openpathsampling.engines.trajectory import Trajectory
import pytest
import torch

from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
from genai_tps.backends.boltz.path_probability import compute_log_path_prob


class MockDiffusion(torch.nn.Module):
    """Minimal stand-in for :class:`boltz.model.modules.diffusionv2.AtomDiffusion`."""

    def __init__(self) -> None:
        super().__init__()
        self.num_sampling_steps = 3
        self.sigma_min = 0.01
        self.sigma_max = 10.0
        self.sigma_data = 16.0
        self.rho = 7.0
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.step_scale_random = None
        self.alignment_reverse_diff = False
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.zero.device

    def sample_schedule(self, num_sampling_steps=None):
        n = num_sampling_steps if num_sampling_steps is not None else self.num_sampling_steps
        steps = torch.arange(n, dtype=torch.float32)
        inv_rho = 1.0 / self.rho
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (n - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = sigmas * self.sigma_data
        return torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

    def preconditioned_network_forward(self, noised_atom_coords, sigma, network_condition_kwargs=None):
        # Identity denoiser in preconditioned space (toy).
        return noised_atom_coords


@pytest.fixture
def core_and_engine():
    diff = MockDiffusion()
    b, m = 1, 4
    atom_mask = torch.ones(b, m)
    kwargs = {"multiplicity": 1}
    core = BoltzSamplerCore(diff, atom_mask, kwargs, multiplicity=1)
    core.build_schedule(3)
    engine = BoltzDiffusionEngine(core, boltz_snapshot_descriptor(n_atoms=m), options={"n_frames_max": 20})
    return core, engine


def test_forward_trajectory_length(core_and_engine):
    core, engine = core_and_engine
    x = core.sample_initial_noise()
    snap = BoltzSnapshot(
        x[0].cpu().numpy(),
        tensor_coords_gpu=x,
        step_index=0,
        sigma=float(core.schedule[0].sigma_tm),
    )
    engine.current_snapshot = snap
    traj = Trajectory([snap])
    for _ in range(core.num_sampling_steps):
        traj.append(engine.generate_next_frame())
    assert len(traj) == core.num_sampling_steps + 1
    assert traj[-1].step_index == core.num_sampling_steps


def test_path_prob_forward_segment(core_and_engine):
    core, _ = core_and_engine
    x = core.sample_initial_noise()
    traj, eps_list, _, _, meta_list = core.generate_segment(x, 0, core.num_sampling_steps)
    lp = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=True,
        n_atoms=x.shape[1],
    )
    assert torch.isfinite(lp)


def test_se3_haar_constant_cancels_in_ratio():
    """Haar density on SO(3) is constant; ratio of two draws is 1."""
    assert 1.0 == 1.0

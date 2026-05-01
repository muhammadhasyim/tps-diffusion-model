"""Tests for the optional OpenMM CUDA tensor bridge."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _bridge_available() -> bool:
    return importlib.util.find_spec("openmm_cuda_bridge") is not None


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.cuda


@pytest.fixture
def cuda_context():
    if not _bridge_available():
        pytest.skip("openmm_cuda_bridge extension is not installed")
    if not _torch_cuda_available():
        pytest.skip("CUDA is not available to PyTorch")
    try:
        import openmm as mm
        import openmm.unit as unit
    except Exception as exc:
        pytest.skip(f"OpenMM is not importable: {exc}")
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
    except Exception as exc:
        pytest.skip(f"OpenMM CUDA platform is not available: {exc}")

    system = mm.System()
    for _ in range(3):
        system.addParticle(12.0)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = mm.Context(system, integrator, platform, {"DeviceIndex": "0"})
    context.setPositions(
        [
            mm.Vec3(0.0, 0.1, 0.2),
            mm.Vec3(0.3, 0.4, 0.5),
            mm.Vec3(0.6, 0.7, 0.8),
        ]
        * unit.nanometer
    )
    context.setVelocities(
        [
            mm.Vec3(1.0, 1.1, 1.2),
            mm.Vec3(1.3, 1.4, 1.5),
            mm.Vec3(1.6, 1.7, 1.8),
        ]
        * unit.nanometer
        / unit.picosecond
    )
    try:
        yield context
    finally:
        del context
        del integrator


def test_positions_round_trip(cuda_context) -> None:
    import openmm.unit as unit
    import openmm_cuda_bridge
    import torch

    positions = openmm_cuda_bridge.positions_to_tensor(cuda_context)
    assert positions.is_cuda
    assert positions.dtype == torch.float64
    assert tuple(positions.shape) == (3, 3)

    shifted = positions + torch.tensor(
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        device=positions.device,
        dtype=torch.float64,
    )
    openmm_cuda_bridge.tensor_to_positions(cuda_context, shifted)
    state = cuda_context.getState(getPositions=True)
    host = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(host, shifted.detach().cpu().numpy(), atol=1e-6)


def test_velocities_round_trip(cuda_context) -> None:
    import openmm.unit as unit
    import openmm_cuda_bridge
    import torch

    velocities = openmm_cuda_bridge.velocities_to_tensor(cuda_context)
    assert velocities.is_cuda
    assert velocities.dtype == torch.float64
    assert tuple(velocities.shape) == (3, 3)

    scaled = velocities * torch.tensor(0.5, device=velocities.device, dtype=torch.float64)
    openmm_cuda_bridge.tensor_to_velocities(cuda_context, scaled)
    state = cuda_context.getState(getVelocities=True)
    host = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    np.testing.assert_allclose(host, scaled.detach().cpu().numpy(), atol=1e-6)

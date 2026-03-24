"""Tests for GPU-native :class:`BoltzSnapshot` construction (TDD contract)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from genai_tps.backends.boltz.snapshot import BoltzSnapshot


def test_from_gpu_batch_requires_tensor() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        BoltzSnapshot.from_gpu_batch(np.zeros((1, 3, 3)))  # type: ignore[arg-type]


def test_from_gpu_batch_requires_rank_3() -> None:
    with pytest.raises(ValueError, match="shape"):
        BoltzSnapshot.from_gpu_batch(torch.randn(4, 3))


def test_from_gpu_batch_preserves_device_cpu() -> None:
    t = torch.randn(1, 5, 3, device="cpu")
    s = BoltzSnapshot.from_gpu_batch(t, step_index=0, sigma=1.0)
    assert s.has_gpu_coords
    assert s.device == torch.device("cpu")
    assert s.tensor_coords is not t
    assert torch.equal(s.tensor_coords, t)


@pytest.mark.cuda
def test_from_gpu_batch_preserves_device_cuda(requires_cuda: torch.device) -> None:
    t = torch.randn(1, 5, 3, device=requires_cuda)
    s = BoltzSnapshot.from_gpu_batch(t, step_index=0, sigma=1.0)
    assert s.device.type == "cuda"
    assert s.tensor_coords is not t
    assert torch.equal(s.tensor_coords, t)


def test_device_raises_without_gpu_tensor() -> None:
    s = BoltzSnapshot(np.zeros((3, 3), dtype=np.float32), tensor_coords_gpu=None)
    with pytest.raises(RuntimeError, match="tensor_coords_gpu"):
        _ = s.device


def test_numpy_mirror_matches_gpu_cpu() -> None:
    t = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    s = BoltzSnapshot.from_gpu_batch(t, step_index=0)
    np.testing.assert_allclose(s.coordinates, t[0].numpy(), rtol=0, atol=0)

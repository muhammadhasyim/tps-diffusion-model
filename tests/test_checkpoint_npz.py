"""Regression tests for trajectory NPZ coordinate materialization (Boltz TPS)."""

import numpy as np
import pytest
import torch

from genai_tps.backends.boltz.snapshot import BoltzSnapshot, snapshot_frame_numpy_copy


def test_snapshot_frame_numpy_copy_cpu_round_trip():
    t = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.5, 0.0, 2.25]]],
        dtype=torch.float32,
    )
    snap = BoltzSnapshot.from_gpu_batch(t, step_index=0, defer_numpy_coords=True)
    out = snapshot_frame_numpy_copy(snap)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, t.numpy()[0], rtol=0, atol=0)
    assert np.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_snapshot_frame_numpy_copy_cuda_round_trip():
    t = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.25, -1.0, 8.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    snap = BoltzSnapshot.from_gpu_batch(t, step_index=1, defer_numpy_coords=True)
    out = snapshot_frame_numpy_copy(snap)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, t.detach().cpu().numpy()[0], rtol=0, atol=0)
    assert np.isfinite(out).all()

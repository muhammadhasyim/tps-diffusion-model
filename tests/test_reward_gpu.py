"""Tests for :mod:`genai_tps.rl.reward_gpu`."""

from __future__ import annotations

import pytest
import torch

from genai_tps.rl.reward_gpu import gpu_pass_fraction_reward_from_coords, snapshot_from_coords_for_reward


def test_snapshot_from_coords_for_reward_shape():
    coords = torch.randn(2, 5, 3)
    snap = snapshot_from_coords_for_reward(coords, step_index=3, sigma=1.0)
    assert snap.step_index == 3
    assert snap.sigma == 1.0
    assert snap.tensor_coords is not None
    assert snap.tensor_coords.shape == (2, 5, 3)


def test_gpu_pass_fraction_reward_from_coords_mock_evaluator():
    class _Ev:
        def evaluate_snapshot(self, snap):
            del snap
            return {"a": 1.0, "b": 0.0}

    r = gpu_pass_fraction_reward_from_coords(
        _Ev(),
        torch.zeros(1, 4, 3),
        step_index=0,
        sigma=None,
    )
    assert r == pytest.approx(0.5)

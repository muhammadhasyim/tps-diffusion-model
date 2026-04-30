"""Tests for :mod:`genai_tps.backends.boltz.boltz2_trunk`."""

from __future__ import annotations

import pytest
import torch

from genai_tps.backends.boltz.boltz2_trunk import transfer_inference_batch_to_device


def test_transfer_inference_batch_to_device_moves_tensors():
    dev = torch.device("cpu")
    batch = {
        "x": torch.tensor([1.0]),
        "record": object(),
        "all_coords": torch.tensor([2.0]),
    }
    out = transfer_inference_batch_to_device(batch, dev)
    assert out["record"] is batch["record"]
    assert torch.equal(out["x"], batch["x"])
    assert torch.equal(out["all_coords"], batch["all_coords"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor move test")
def test_transfer_inference_batch_to_device_cuda():
    dev = torch.device("cuda", torch.cuda.current_device())
    batch = {"x": torch.tensor([1.0], device="cpu")}
    out = transfer_inference_batch_to_device(batch, dev)
    assert out["x"].device.type == "cuda"

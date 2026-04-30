"""Tests for :mod:`genai_tps.utils.compute_device`."""

from __future__ import annotations

import pytest
import torch

from genai_tps.utils.compute_device import (
    cuda_device_index_for_openmm,
    openmm_device_index_properties,
    parse_torch_device,
)


def test_parse_torch_device_cpu_cuda_variants() -> None:
    assert parse_torch_device("cpu").type == "cpu"
    assert parse_torch_device("cuda").type == "cuda"
    d = parse_torch_device("cuda:3")
    assert d.type == "cuda" and d.index == 3


def test_parse_torch_device_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_torch_device("  ")


def test_cuda_device_index_for_openmm() -> None:
    assert cuda_device_index_for_openmm(torch.device("cpu")) is None
    assert cuda_device_index_for_openmm(torch.device("cuda")) == 0
    assert cuda_device_index_for_openmm(torch.device("cuda", 2)) == 2


def test_openmm_device_index_properties() -> None:
    assert openmm_device_index_properties("CUDA", None) == {}
    assert openmm_device_index_properties("CPU", 1) == {}
    assert openmm_device_index_properties("CUDA", 2) == {"DeviceIndex": "2"}
    assert openmm_device_index_properties("OpenCL", 0) == {"DeviceIndex": "0"}

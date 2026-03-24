"""Shared pytest fixtures (Boltz import path, devices, optional CUDA)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parents[1]
_boltz_src = _root / "boltz" / "src"
if _boltz_src.is_dir():
    sys.path.insert(0, str(_boltz_src))

_scripts = _root / "scripts"
if _scripts.is_dir() and str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))


@pytest.fixture
def device() -> torch.device:
    """Default compute device for tensor tests (CUDA if available, else CPU).

    When CUDA is used, the device includes the device index (e.g. ``cuda:0``) so
    equality with :attr:`torch.Tensor.device` matches PyTorch's canonical form.
    ``torch.device("cuda")`` omits the index and does *not* compare equal to
    ``cuda:0``, which caused false failures on GPU runners.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@pytest.fixture
def requires_cuda(device: torch.device) -> torch.device:
    """Use in tests that must exercise the CUDA stack."""
    if device.type != "cuda":
        pytest.skip("CUDA not available")
    return device

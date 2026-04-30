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


@pytest.fixture
def mock_boltz_core(device: torch.device):
    """Small :class:`BoltzSamplerCore` over :class:`tests.mock_boltz_diffusion.MockDiffusion`."""
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore

    from tests.mock_boltz_diffusion import MockDiffusion

    n_atom = 4
    mock = MockDiffusion(device=device)
    atom_mask = torch.ones(1, n_atom, dtype=torch.float32, device=device)
    core = BoltzSamplerCore(mock, atom_mask, {}, multiplicity=1)
    core.build_schedule(3)
    return core


@pytest.fixture
def mock_boltz_engine(mock_boltz_core):
    """:class:`BoltzDiffusionEngine` wrapping ``mock_boltz_core`` (4 heavy atoms, 3-step schedule)."""
    from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
    from genai_tps.backends.boltz.snapshot import boltz_snapshot_descriptor

    n_atom = int(mock_boltz_core.atom_mask.shape[1])
    desc = boltz_snapshot_descriptor(n_atoms=n_atom)
    return BoltzDiffusionEngine(
        mock_boltz_core,
        desc,
        options={"n_frames_max": mock_boltz_core.num_sampling_steps + 4},
    )


@pytest.fixture
def ala_ala_pdb_path(tmp_path: Path) -> Path:
    """Write shared ALA-ALA peptide PDB under ``tmp_path``."""
    from tests.pdb_fixtures import ALA_ALA_PDB

    p = tmp_path / "ala_ala.pdb"
    p.write_text(ALA_ALA_PDB, encoding="utf-8")
    return p

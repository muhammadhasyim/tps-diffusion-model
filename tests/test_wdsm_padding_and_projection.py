"""Tests for WDSM padding to Boltz width, centering vs Boltz utils, and P_x."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parents[1]
_src = _root / "src" / "python"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
_boltz_src = _root / "boltz" / "src"
if _boltz_src.is_dir():
    sys.path.insert(0, str(_boltz_src))

from genai_tps.training.dataset import ReweightedStructureDataset, pad_reweighted_dataset_to_boltz_atom_mask
from genai_tps.training.loss import _boltz_style_center_and_augment, _center
from genai_tps.training.quotient_projection import horizontal_projection, mean_curvature_vector


def test_pad_reweighted_dataset_expands_and_matches_boltz_mask() -> None:
    m_phys, m_model = 5, 12
    coords = np.random.randn(3, m_phys, 3).astype(np.float32)
    logw = np.zeros(3, dtype=np.float64)
    atom_mask = np.ones((3, m_phys), dtype=np.float32)
    ds = ReweightedStructureDataset(coords, logw, atom_mask)
    ref = torch.zeros(m_model, dtype=torch.float32)
    ref[:m_phys] = 1.0

    out = pad_reweighted_dataset_to_boltz_atom_mask(ds, ref)
    assert out.coords.shape == (3, m_model, 3)
    assert out.atom_mask.shape == (3, m_model)
    assert np.allclose(out.coords[:, m_phys:, :], 0.0)
    assert np.allclose(out.atom_mask[:, m_phys:], 0.0)
    np.testing.assert_allclose(out.atom_mask[0], ref.numpy(), atol=1e-6)


def test_pad_reweighted_dataset_raises_when_phys_exceeds_model() -> None:
    ds = ReweightedStructureDataset(
        np.zeros((1, 20, 3), dtype=np.float32),
        np.zeros(1, dtype=np.float64),
        np.ones((1, 20), dtype=np.float32),
    )
    ref = torch.ones(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="cannot truncate"):
        pad_reweighted_dataset_to_boltz_atom_mask(ds, ref)


def test_pad_reweighted_dataset_raises_on_mask_mismatch() -> None:
    m_phys, m_model = 4, 8
    coords = np.random.randn(1, m_phys, 3).astype(np.float32)
    logw = np.zeros(1, dtype=np.float64)
    # Wrong physical mask vs Boltz (expects ones on first 4)
    atom_mask = np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    ds = ReweightedStructureDataset(coords, logw, atom_mask)
    ref = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="atom_mask"):
        pad_reweighted_dataset_to_boltz_atom_mask(ds, ref)


def test_center_matches_boltz_center_random_augmentation_no_aug() -> None:
    from boltz.model.modules.utils import center_random_augmentation

    torch.manual_seed(0)
    b, n = 2, 7
    atom_coords = torch.randn(b, n, 3, dtype=torch.float64)
    atom_mask = torch.ones(b, n, dtype=torch.float64)
    boltz_out = center_random_augmentation(
        atom_coords.clone(),
        atom_mask,
        augmentation=False,
        centering=True,
    )
    ours, _ = _center(atom_coords, atom_mask)
    torch.testing.assert_close(boltz_out, ours, rtol=0, atol=1e-12)


def test_boltz_style_center_eval_matches_center_only() -> None:
    class _M:
        training = False

    b, n = 2, 6
    x0 = torch.randn(b, n, 3, dtype=torch.float32)
    mask = torch.ones(b, n, dtype=torch.float32)
    out = _boltz_style_center_and_augment(x0, mask, _M())
    ref, _ = _center(x0, mask)
    torch.testing.assert_close(out, ref)


def test_horizontal_projection_kills_pure_rotation_velocity() -> None:
    """Pure omega x x is vertical; P_x should remove it (see test_quotient_projection)."""
    torch.manual_seed(1)
    b, n = 1, 9
    x = torch.randn(b, n, 3, dtype=torch.float64)
    x = x - (x * torch.ones(b, n, 1)).sum(dim=1, keepdim=True) / n
    mask = torch.ones(b, n, dtype=torch.float64)
    omega = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    v = torch.linalg.cross(omega.unsqueeze(1).expand(b, n, 3), x, dim=-1)
    proj = horizontal_projection(x, v, mask)
    assert proj.abs().max().item() < 1e-5


def test_mean_curvature_finite_on_simple_cloud() -> None:
    torch.manual_seed(2)
    b, n = 1, 6
    x = torch.randn(b, n, 3, dtype=torch.float64)
    x = x - (x * torch.ones(b, n, 1)).sum(dim=1, keepdim=True) / n
    mask = torch.ones(b, n, dtype=torch.float64)
    h = mean_curvature_vector(x, mask)
    assert torch.isfinite(h).all()
    assert (h * (1.0 - mask).unsqueeze(-1)).abs().max().item() == 0.0

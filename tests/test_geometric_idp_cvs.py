"""Tests for end-to-end distance, long-range contact count, and gyration-shape CVs."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _snap(coords: torch.Tensor) -> SimpleNamespace:
    """Snapshot stub: ``tensor_coords`` with shape ``(1, N, 3)``."""
    c = coords.unsqueeze(0) if coords.dim() == 2 else coords
    return SimpleNamespace(tensor_coords=c)


def test_end_to_end_two_points() -> None:
    from genai_tps.backends.boltz.collective_variables import end_to_end_distance

    x = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert end_to_end_distance(_snap(x)) == pytest.approx(5.0)
    assert end_to_end_distance(_snap(x[:1])) == 0.0


def test_end_to_end_line_matches_span() -> None:
    from genai_tps.backends.boltz.collective_variables import end_to_end_distance

    x = torch.zeros(5, 3)
    x[:, 0] = torch.arange(5, dtype=torch.float32) * 10.0
    assert end_to_end_distance(_snap(x)) == pytest.approx(40.0)


def test_ca_contact_count_single_pair() -> None:
    from genai_tps.backends.boltz.collective_variables import ca_contact_count

    x = torch.zeros(7, 3)
    x[6, 0] = 5.0
    for i in range(1, 6):
        x[i, 1] = 50.0 + float(i)
    assert ca_contact_count(_snap(x), seq_sep=6, dist_threshold=8.0) == pytest.approx(1.0)


def test_shape_kappa2_sphere_vs_rod() -> None:
    from genai_tps.backends.boltz.collective_variables import shape_kappa2

    torch.manual_seed(0)
    dirs = torch.randn(80, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sphere_pts = dirs * 10.0
    assert shape_kappa2(_snap(sphere_pts)) < 0.15

    rod = torch.stack([torch.tensor([float(i), 0.0, 0.0]) for i in range(20)])
    assert shape_kappa2(_snap(rod)) == pytest.approx(1.0, abs=1e-5)


def test_shape_acylindricity_symmetric_rod() -> None:
    from genai_tps.backends.boltz.collective_variables import shape_acylindricity

    i = torch.arange(15, dtype=torch.float32)
    z = torch.zeros(15)
    h = torch.full((15,), 0.3)
    x = torch.cat(
        [
            torch.stack([i, z, z], dim=1),
            torch.stack([i, h, z], dim=1),
            torch.stack([i, z, h], dim=1),
        ],
        dim=0,
    )
    assert abs(shape_acylindricity(_snap(x))) < 0.05


def test_new_geometric_names_in_run_opes_single_cv_list() -> None:
    """Bias CV strings in run_opes_tps must include the gyration / contact CV names."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_opes_tps.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    names: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_SINGLE_CV_NAMES":
                    if isinstance(node.value, ast.List):
                        names = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    assert names is not None
    for key in (
        "end_to_end",
        "ca_contact_count",
        "shape_kappa2",
        "shape_acylindricity",
    ):
        assert key in names

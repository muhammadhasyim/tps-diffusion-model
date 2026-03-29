"""Unit tests for IDP-oriented geometric collective variables."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


def _snap(coords: torch.Tensor) -> object:
    """Minimal snapshot with ``tensor_coords`` shape (1, N, 3)."""
    class _S:
        pass

    s = _S()
    c = coords.unsqueeze(0) if coords.dim() == 2 else coords
    s.tensor_coords = c
    return s


def test_end_to_end_two_points() -> None:
    from genai_tps.backends.boltz.collective_variables import end_to_end_distance

    x = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert end_to_end_distance(_snap(x)) == pytest.approx(5.0)
    assert end_to_end_distance(_snap(x[:1])) == 0.0


def test_end_to_end_line_matches_span() -> None:
    from genai_tps.backends.boltz.collective_variables import end_to_end_distance

    # Five points along x; first–last distance = 40 Å
    x = torch.stack([torch.tensor([float(i * 10), 0.0, 0.0]) for i in range(5)])
    assert end_to_end_distance(_snap(x)) == pytest.approx(40.0)


def test_ca_contact_count_single_pair() -> None:
    from genai_tps.backends.boltz.collective_variables import ca_contact_count

    # 7 beads: only pair (0, 6) has |i-j| >= 6; place them 5 Å apart
    x = torch.zeros(7, 3)
    x[6, 0] = 5.0
    # Spread others so no extra long-range contacts < 8 Å
    for i in range(1, 6):
        x[i, 1] = 50.0 + float(i)
    assert ca_contact_count(_snap(x), seq_sep=6, dist_threshold=8.0) == pytest.approx(1.0)


def test_shape_kappa2_sphere_vs_rod() -> None:
    from genai_tps.backends.boltz.collective_variables import shape_kappa2

    torch.manual_seed(0)
    # Many points near a sphere surface -> nearly isotropic cloud
    dirs = torch.randn(80, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sphere_pts = dirs * 10.0
    k_s = shape_kappa2(_snap(sphere_pts))
    assert k_s < 0.15

    # Colinear rod along x
    rod = torch.stack([torch.tensor([float(i), 0.0, 0.0]) for i in range(20)])
    k_r = shape_kappa2(_snap(rod))
    assert k_r == pytest.approx(1.0, abs=1e-5)


def test_shape_acylindricity_symmetric_rod() -> None:
    from genai_tps.backends.boltz.collective_variables import shape_acylindricity

    # Axisymmetric distribution about x: λ2 ≈ λ3 -> acylindricity ~ 0
    pts = []
    for i in range(15):
        pts.append([float(i), 0.0, 0.0])
        pts.append([float(i), 0.3, 0.0])
        pts.append([float(i), 0.0, 0.3])
    x = torch.tensor(pts, dtype=torch.float32)
    ac = shape_acylindricity(_snap(x))
    assert abs(ac) < 0.05


def test_new_geometric_names_in_run_opes_single_cv_list() -> None:
    """``_SINGLE_CV_NAMES`` must include the new IDP-oriented CVs (AST parse; no heavy imports)."""
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

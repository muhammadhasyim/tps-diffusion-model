"""Tests for ``scripts/render_tps_checkpoint_cartoons.py`` topology discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_render_script():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "render_tps_checkpoint_cartoons.py"
    spec = importlib.util.spec_from_file_location("_render_ckpt_cartoons", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDiscoverTopoNpz:
    def test_explicit_topo(self, tmp_path: Path) -> None:
        mod = _load_render_script()
        topo = tmp_path / "s.npz"
        topo.write_bytes(b"")
        ck = tmp_path / "trajectory_checkpoints"
        ck.mkdir()
        assert mod.discover_topo_npz(ck, topo) == topo.resolve()

    def test_explicit_missing_raises(self, tmp_path: Path) -> None:
        mod = _load_render_script()
        ck = tmp_path / "trajectory_checkpoints"
        ck.mkdir()
        with pytest.raises(FileNotFoundError):
            mod.discover_topo_npz(ck, tmp_path / "missing.npz")

    def test_auto_discover_single(self, tmp_path: Path) -> None:
        mod = _load_render_script()
        topo = tmp_path / "boltz_results_x/processed/structures/case.npz"
        topo.parent.mkdir(parents=True)
        topo.write_bytes(b"")
        ck = tmp_path / "trajectory_checkpoints"
        ck.mkdir()
        assert mod.discover_topo_npz(ck, None) == topo.resolve()

    def test_auto_discover_none_raises(self, tmp_path: Path) -> None:
        mod = _load_render_script()
        ck = tmp_path / "trajectory_checkpoints"
        ck.mkdir()
        with pytest.raises(FileNotFoundError):
            mod.discover_topo_npz(ck, None)

    def test_auto_discover_ambiguous_raises(self, tmp_path: Path) -> None:
        mod = _load_render_script()
        for name in ("a", "b"):
            p = tmp_path / f"boltz_results_{name}/processed/structures/case.npz"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"")
        ck = tmp_path / "trajectory_checkpoints"
        ck.mkdir()
        with pytest.raises(ValueError, match="Multiple"):
            mod.discover_topo_npz(ck, None)

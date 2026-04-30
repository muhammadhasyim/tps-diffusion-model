"""Tests for PLUMED kernel / config.txt inspection (OPES module gate)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def test_plumed_module_enabled_parses_on_off(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_kernel import plumed_module_enabled

    cfg = tmp_path / "config.txt"
    cfg.write_text(
        "module bias on (default-on)\n"
        "module opes off (default-off)\n"
        "module colvar on (default-on)\n",
        encoding="utf-8",
    )
    assert plumed_module_enabled("opes", config_path=cfg) is False
    assert plumed_module_enabled("bias", config_path=cfg) is True
    assert plumed_module_enabled("nope", config_path=cfg) is None


def test_assert_plumed_opes_metad_available_raises_when_opes_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from genai_tps.simulation import plumed_kernel as pk

    fake_kernel = tmp_path / "libplumedKernel.so"
    fake_kernel.write_bytes(b"")
    cfg = tmp_path / "plumed" / "src" / "config"
    cfg.mkdir(parents=True)
    (cfg / "config.txt").write_text(
        "module opes off (default-off)\n", encoding="utf-8"
    )

    monkeypatch.setattr(pk, "plumed_kernel_path", lambda: fake_kernel.resolve())

    from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

    with pytest.raises(RuntimeError, match="module opes off"):
        assert_plumed_opes_metad_available()


def test_plumed_kernel_path_prefers_executable_env_root_over_sys_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mimics ``conda run`` where ``sys.prefix`` may not match PATH's ``plumed``."""
    from genai_tps.simulation.plumed_kernel import plumed_kernel_path

    fake_env = tmp_path / "conda_like_env"
    (fake_env / "bin").mkdir(parents=True)
    plumed = fake_env / "bin" / "plumed"
    plumed.write_text("#!/bin/sh\nexec true\n", encoding="utf-8")
    plumed.chmod(0o755)

    kern = fake_env / "lib" / "libplumedKernel.so"
    kern.parent.mkdir(parents=True)
    kern.touch()

    monkeypatch.delenv("PLUMED_KERNEL", raising=False)
    fake_prefix = tmp_path / "wrong_prefix"
    fake_prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(fake_prefix))
    path_head = str(fake_env / "bin") + os.pathsep + os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", path_head)

    got = plumed_kernel_path()
    assert got == kern.resolve()


def test_assert_plumed_opes_metad_available_passes_when_opes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from genai_tps.simulation import plumed_kernel as pk

    fake_kernel = tmp_path / "libplumedKernel.so"
    fake_kernel.write_bytes(b"")
    cfg = tmp_path / "plumed" / "src" / "config"
    cfg.mkdir(parents=True)
    (cfg / "config.txt").write_text(
        "module opes on (default-off)\n", encoding="utf-8"
    )

    monkeypatch.setattr(pk, "plumed_kernel_path", lambda: fake_kernel.resolve())

    from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

    assert_plumed_opes_metad_available()


def test_assert_plumed_opes_hint_when_conda_prefix_mismatches_sys_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HPC setups often leave site Anaconda first on PATH while CONDA_PREFIX is the job env."""
    from genai_tps.simulation import plumed_kernel as pk

    env_root = tmp_path / "miniforge_env"
    env_root.mkdir()
    monkeypatch.setenv("CONDA_PREFIX", str(env_root))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "other_python_prefix"), raising=False)
    monkeypatch.delenv("PLUMED_KERNEL", raising=False)
    monkeypatch.setattr(pk, "plumed_kernel_path", lambda: None)

    from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

    with pytest.raises(RuntimeError, match="Likely cause:.*python.*is not the interpreter"):
        assert_plumed_opes_metad_available()

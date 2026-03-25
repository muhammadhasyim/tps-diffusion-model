"""Regression tests for vendored OpenPathSampling version helpers."""

from __future__ import annotations


def test_get_setup_cfg_handles_missing_setup_cfg_when_depth_negative() -> None:
    """When no setup.cfg exists in parent search, rel_path is None; must not raise.

    genai-tps uses pyproject.toml only; ``_installed_version`` defaults
    ``_version_setup_depth`` to -1, which triggers parent-dir search that
    returns None.  Previously ``os.path.join`` crashed with TypeError.
    """
    from openpathsampling.version import get_setup_cfg

    assert get_setup_cfg(-1, "setup.cfg") is None


def test_openpathsampling_imports_without_setup_cfg() -> None:
    """Package import must succeed in editable installs without setup.cfg."""
    import openpathsampling  # noqa: F401

    assert hasattr(openpathsampling, "version")

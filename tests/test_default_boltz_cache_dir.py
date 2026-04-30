"""Tests for :func:`~genai_tps.backends.boltz.cache_paths.default_boltz_cache_dir`."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_tps.backends.boltz.cache_paths import default_boltz_cache_dir


def test_default_boltz_cache_dir_prefers_boltz_cache_env(monkeypatch, tmp_path: Path) -> None:
    want = tmp_path / "custom_boltz"
    monkeypatch.setenv("BOLTZ_CACHE", str(want))
    monkeypatch.setenv("SCRATCH", "/should_not_win")
    assert default_boltz_cache_dir() == want.resolve()


def test_default_boltz_cache_dir_uses_scratch_without_boltz_cache(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert default_boltz_cache_dir() == (tmp_path / ".boltz").resolve()


def test_default_boltz_cache_dir_raises_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.setenv("HOME", "/tmp/genai_tps_fakehome_should_not_be_used")
    with pytest.raises(RuntimeError, match="BOLTZ_CACHE"):
        default_boltz_cache_dir()

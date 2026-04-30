"""Tests for :func:`~genai_tps.backends.boltz.cache_paths.default_boltz_cache_dir`."""

from __future__ import annotations

from pathlib import Path

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


def test_default_boltz_cache_dir_falls_back_to_home(monkeypatch) -> None:
    monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    monkeypatch.delenv("SCRATCH", raising=False)
    fake_home = Path("/tmp/genai_tps_fakehome_boltz_cache_test")
    monkeypatch.setenv("HOME", str(fake_home))
    assert default_boltz_cache_dir() == fake_home / ".boltz"

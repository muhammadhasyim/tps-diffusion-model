"""Unit tests for optional NVTX profiling helpers."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext

import pytest


@pytest.fixture(autouse=True)
def _reset_nvtx_util(monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from each other and from a real ``nvtx`` install."""
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    sys.modules.pop("nvtx", None)
    import genai_tps.simulation.nvtx_util as nvtx_util

    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)
    yield
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    sys.modules.pop("nvtx", None)
    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)


def test_nvtx_range_noop_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    from genai_tps.simulation.nvtx_util import nvtx_range

    with nvtx_range("should_not_import"):
        pass
    assert "nvtx" not in sys.modules


def test_nvtx_range_uses_annotate_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class _Annot:
        def __call__(self, message: str):
            calls.append(("annotate", message))
            return nullcontext()

    fake = types.SimpleNamespace(annotate=_Annot())
    monkeypatch.setenv("GENAI_TPS_NVTX", "1")
    sys.modules["nvtx"] = fake

    import genai_tps.simulation.nvtx_util as nvtx_util

    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)

    with nvtx_util.nvtx_range("hello"):
        pass
    assert calls == [("annotate", "hello")]


def test_nvtx_range_push_pop_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Fake:
        def push_range(self, message: str) -> None:
            calls.append(f"push:{message}")

        def pop_range(self) -> None:
            calls.append("pop")

    monkeypatch.setenv("GENAI_TPS_NVTX", "true")
    sys.modules["nvtx"] = _Fake()

    import genai_tps.simulation.nvtx_util as nvtx_util

    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)

    with nvtx_util.nvtx_range("legacy"):
        calls.append("body")
    assert calls == ["push:legacy", "body", "pop"]

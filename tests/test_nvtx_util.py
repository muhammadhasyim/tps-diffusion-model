"""Unit tests for optional NVTX profiling helpers."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext

import pytest


@pytest.fixture(autouse=True)
def _reset_nvtx_util(monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from each other and from a real ``nvtx`` / ``torch`` stub."""
    saved_torch = sys.modules.get("torch")
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    sys.modules.pop("nvtx", None)
    import genai_tps.utils.nvtx_util as nvtx_util

    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)
    yield
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    sys.modules.pop("nvtx", None)
    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)
    if saved_torch is not None:
        sys.modules["torch"] = saved_torch
    else:
        sys.modules.pop("torch", None)


def test_nvtx_range_noop_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENAI_TPS_NVTX", raising=False)
    from genai_tps.utils.nvtx_util import nvtx_range

    with nvtx_range("should_not_import"):
        pass
    assert "nvtx" not in sys.modules


def test_nvtx_range_prefers_torch_cuda_nvtx_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _CudaNvtx:
        @staticmethod
        def range(name: str):
            calls.append(f"range:{name}")
            return nullcontext()

    class _Cuda:
        nvtx = _CudaNvtx()

        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("torch")
            self.cuda = _Cuda()

    fake_torch = _Torch()
    monkeypatch.setenv("GENAI_TPS_NVTX", "1")
    sys.modules["torch"] = fake_torch

    import genai_tps.utils.nvtx_util as nvtx_util

    importlib.reload(nvtx_util)
    from genai_tps.utils.nvtx_util import nvtx_range

    with nvtx_range("train_step"):
        pass
    assert calls == ["range:train_step"]
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
    # Torch without usable CUDA NVTX → use ``nvtx`` package
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch

    import genai_tps.utils.nvtx_util as nvtx_util

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
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch

    import genai_tps.utils.nvtx_util as nvtx_util

    nvtx_util._nvtx_mod = False  # type: ignore[attr-defined]
    importlib.reload(nvtx_util)

    with nvtx_util.nvtx_range("legacy"):
        calls.append("body")
    assert calls == ["push:legacy", "body", "pop"]

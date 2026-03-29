"""CUDA header discovery for Triton / torch.compile (CPATH fix)."""

from __future__ import annotations

import os

import pytest

from genai_tps.backends.boltz.gpu_core import (
    _cuda_h_include_search_dirs,
    _first_dir_with_cuda_h,
    _prepend_cpath_for_cuda_toolkit,
)


@pytest.mark.skipif(
    not os.path.isfile("/usr/local/cuda/include/cuda.h"),
    reason="system CUDA toolkit headers not at /usr/local/cuda",
)
def test_first_dir_with_cuda_h_finds_standard_install() -> None:
    found = _first_dir_with_cuda_h()
    assert found is not None
    assert os.path.isfile(os.path.join(found, "cuda.h"))


def test_search_dirs_are_unique_paths() -> None:
    dirs = _cuda_h_include_search_dirs()
    assert len(dirs) == len(set(dirs))


def test_prepend_cpath_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.delenv("C_INCLUDE_PATH", raising=False)
    _prepend_cpath_for_cuda_toolkit("/tmp/fake_cuda_include")
    assert os.environ["CPATH"] == "/tmp/fake_cuda_include"
    assert os.environ["C_INCLUDE_PATH"] == "/tmp/fake_cuda_include"
    _prepend_cpath_for_cuda_toolkit("/other")
    assert os.environ["CPATH"].startswith("/other" + os.pathsep)

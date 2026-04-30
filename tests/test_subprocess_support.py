"""Tests for :mod:`genai_tps.subprocess_support`."""

from __future__ import annotations

import os

from genai_tps.subprocess_support import child_env_with_repo_src_python, repository_root


def test_repository_root_contains_src_python() -> None:
    root = repository_root()
    assert (root / "src" / "python" / "genai_tps").is_dir()


def test_child_env_prepends_repo_src_python() -> None:
    base = {"FOO": "1", "PYTHONPATH": "/tmp/existing"}
    env = child_env_with_repo_src_python(base)
    assert env["FOO"] == "1"
    src = str(repository_root() / "src" / "python")
    assert env["PYTHONPATH"].startswith(src + os.pathsep)
    assert "/tmp/existing" in env["PYTHONPATH"]

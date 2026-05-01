"""Smoke tests for GPU profiling helper scripts (no GPU required)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MONITOR = _REPO_ROOT / "scripts" / "profile" / "run_gpu_monitor.sh"
_NSYS_WRAP = _REPO_ROOT / "scripts" / "profile" / "nsys_wrap_train.sh"
_REPEX_SCALEUP = _REPO_ROOT / "scripts" / "smoke" / "run_oneopes_repex_scaleup.sh"


def _run_bash(script: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    cmd = ["/bin/bash", str(script), *args]
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        **kwargs,
    )


def test_run_gpu_monitor_help():
    r = _run_bash(_MONITOR, ["--help"])
    assert r.returncode == 0
    assert "run_gpu_monitor" in r.stdout or "high-rate" in r.stdout


def test_run_gpu_monitor_bash_syntax():
    r = subprocess.run(
        ["/bin/bash", "-n", str(_MONITOR)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_run_oneopes_repex_scaleup_bash_syntax():
    r = subprocess.run(
        ["/bin/bash", "-n", str(_REPEX_SCALEUP)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_run_gpu_monitor_with_cmd_true():
    """Background monitor + trivial command exits quickly."""
    r = _run_bash(_MONITOR, ["--gpus", "0", "--interval-ms", "200", "--mode", "nvidia", "--", "true"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_nsys_wrap_train_help():
    r = _run_bash(_NSYS_WRAP, ["--help"])
    assert r.returncode == 0
    assert "nsys_wrap_train" in r.stdout or "Nsight" in r.stdout


def test_nsys_wrap_train_bash_syntax():
    r = subprocess.run(
        ["/bin/bash", "-n", str(_NSYS_WRAP)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(
    os.environ.get("RUN_NSYS_WRAP_SMOKE") != "1",
    reason="set RUN_NSYS_WRAP_SMOKE=1 to run a short nsys profile smoke test locally",
)
@pytest.mark.skipif(not shutil.which("nsys"), reason="nsys CLI not installed")
def test_nsys_wrap_train_short_python():
    """Short nsys capture (trace=none) around Python; opt-in to avoid CI hangs."""
    r = _run_bash(
        _NSYS_WRAP,
        [
            "--nsys-duration",
            "1",
            "--nsys-trace",
            "none",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        timeout=90,
    )
    assert r.returncode == 0, r.stdout + r.stderr

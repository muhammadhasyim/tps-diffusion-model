"""Tests for Zenodo download / extract helpers (no network by default)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tarfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOWNLOAD_SCRIPT = _REPO_ROOT / "scripts" / "download_runs_n_poses_zenodo.py"
_EXTRACT_SCRIPT = _REPO_ROOT / "scripts" / "extract_zenodo_runs_n_poses.py"


def _load_download_module():
    spec = importlib.util.spec_from_file_location("download_runs_n_poses_zenodo", _DOWNLOAD_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_files_preset_and_default() -> None:
    mod = _load_download_module()
    assert mod._resolve_files_arg(files=None, preset=None) == mod.DEFAULT_FILES_STR
    assert "ground_truth.tar.gz" in mod._resolve_files_arg(files=None, preset="medium")
    assert "predictions.tar.gz" in mod._resolve_files_arg(files=None, preset="heavy")
    assert mod._resolve_files_arg(files="a.csv,b.csv", preset="heavy") == "a.csv,b.csv"


def test_extract_zenodo_runs_n_poses_idempotent(tmp_path: Path) -> None:
    inner = tmp_path / "payload.txt"
    inner.write_text("hello", encoding="utf-8")
    arc = tmp_path / "sample.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(inner, arcname="payload.txt")

    out = tmp_path / "out"
    cmd_base = [
        sys.executable,
        str(_EXTRACT_SCRIPT),
        "--archive",
        str(arc),
        "--out",
        str(out),
    ]
    r1 = subprocess.run(cmd_base, cwd=str(_REPO_ROOT), check=False, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert (out / "payload.txt").read_text(encoding="utf-8") == "hello"
    assert (out / ".extracted_ok").is_file()

    r2 = subprocess.run(cmd_base, cwd=str(_REPO_ROOT), check=False, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "skip" in r2.stdout.lower()

    r3 = subprocess.run(
        [*cmd_base, "--force"],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert r3.returncode == 0, r3.stderr


def test_extract_missing_archive_fails(tmp_path: Path) -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(_EXTRACT_SCRIPT),
            "--archive",
            str(tmp_path / "nope.tar.gz"),
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1

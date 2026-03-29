"""Tests for scripts/export_opes_basin_frames.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_export_opes_basin_frames_2d(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "export_opes_basin_frames.py"
    run = tmp_path / "run"
    ck = run / "trajectory_checkpoints"
    ck.mkdir(parents=True)
    np.savez_compressed(ck / "tps_mc_step_00000001.npz", coords=np.ones((2, 4, 3)))
    np.savez_compressed(ck / "tps_mc_step_00000002.npz", coords=np.ones((2, 4, 3)) * 2.0)

    jl = run / "tps_steps.jsonl"
    jl.write_text(
        '{"step": 1, "cv_value": [0.1, 5.5], "accepted": true}\n'
        '{"step": 2, "cv_value": [9.0, 9.0], "accepted": true}\n',
        encoding="utf-8",
    )

    out = tmp_path / "basin_out"
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run),
        "--cv-json",
        str(jl),
        "--out-dir",
        str(out),
        "--x-min",
        "0",
        "--x-max",
        "0.5",
        "--y-min",
        "5",
        "--y-max",
        "6",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    copied = out / "tps_mc_step_00000001.npz"
    assert copied.is_file()
    man = json.loads((out / "basin_manifest.json").read_text(encoding="utf-8"))
    assert man["n_copied"] == 1
    assert man["n_missing_checkpoint"] == 0
    assert len(man["entries"]) == 1

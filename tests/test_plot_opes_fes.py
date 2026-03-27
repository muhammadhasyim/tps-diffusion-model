"""Tests for plot_opes_fes CV loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_load_cv_samples_jsonl(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "tps_steps.jsonl"
    p.write_text(
        json.dumps({"step": 1, "cv_value": 0.5}) + "\n"
        + json.dumps({"step": 2, "cv_value": None}) + "\n"
        + json.dumps({"step": 3, "cv_value": 1.25}) + "\n",
        encoding="utf-8",
    )
    arr = _load_cv_samples_jsonl(p)
    np.testing.assert_array_almost_equal(arr, np.array([0.5, 1.25]))


def test_load_cv_samples_jsonl_empty_raises(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_opes_fes import _load_cv_samples_jsonl  # type: ignore[import]

    p = tmp_path / "empty.jsonl"
    p.write_text('{"step": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no finite"):
        _load_cv_samples_jsonl(p)

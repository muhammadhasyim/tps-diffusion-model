"""Unit tests for :mod:`genai_tps.simulation.gpu_util_csv_logger`."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from genai_tps.simulation.gpu_util_csv_logger import GpuUtilCsvLogger


def test_logger_with_custom_query_writes_header_and_rows(tmp_path: Path) -> None:
    def fake_query(_gpu: int) -> str:
        return "45, 2000, 16384, 120.5"

    csv_path = tmp_path / "util.csv"
    logger = GpuUtilCsvLogger(csv_path, 0.05, gpu_index=2, query_fn=fake_query)
    assert logger.start() is True
    time.sleep(0.18)
    logger.stop()

    text = csv_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    assert lines[0].startswith("unix_time_s,gpu_index,util_gpu_pct")
    data = [ln for ln in lines[1:] if not ln.endswith(",,,,")]
    assert len(data) >= 1
    parts = data[0].split(",")
    assert parts[1] == "2"
    assert parts[2] == "45"
    assert parts[3] == "2000"
    assert parts[4] == "16384"
    assert parts[5] == "120.5"


def test_start_returns_false_when_nvidia_smi_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "genai_tps.simulation.gpu_util_csv_logger.shutil.which",
        lambda _name: None,
    )
    logger = GpuUtilCsvLogger(tmp_path / "n.csv", 0.1, 0)
    assert logger.start() is False

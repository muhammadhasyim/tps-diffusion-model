"""Tests for aligning Boltz sampler quotient_space_sampling with WDSM training."""

from __future__ import annotations

import json
from pathlib import Path

from genai_tps.backends.boltz.inference import quotient_space_sampling_for_checkpoint


def test_quotient_flag_none_checkpoint(tmp_path: Path) -> None:
    assert quotient_space_sampling_for_checkpoint(None) is False


def test_quotient_flag_from_training_summary_true_quotient(tmp_path: Path) -> None:
    ckpt = tmp_path / "boltz2_wdsm_final.pt"
    ckpt.write_text("stub")
    summary = tmp_path / "training_summary.json"
    summary.write_text(
        json.dumps({"config": {"loss_type": "true-quotient", "quotient_space_sampling": True}})
    )
    assert quotient_space_sampling_for_checkpoint(ckpt) is True


def test_quotient_flag_from_training_summary_cartesian(tmp_path: Path) -> None:
    ckpt = tmp_path / "boltz2_wdsm_final.pt"
    ckpt.write_text("stub")
    summary = tmp_path / "training_summary.json"
    summary.write_text(json.dumps({"config": {"loss_type": "cartesian"}}))
    assert quotient_space_sampling_for_checkpoint(ckpt) is False


def test_quotient_flag_heuristic_directory_name(tmp_path: Path) -> None:
    sub = tmp_path / "sft_true-quotient"
    sub.mkdir()
    ckpt = sub / "boltz2_wdsm_final.pt"
    ckpt.write_text("stub")
    assert quotient_space_sampling_for_checkpoint(ckpt) is True

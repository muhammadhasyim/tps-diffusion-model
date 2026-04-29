"""Tests for ATLAS fine-tuning CLI command construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "atlas" / "02_finetune_boltz2_atlas.py"
_SPEC = importlib.util.spec_from_file_location("atlas_finetune_cli", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_train_command = _MODULE.build_train_command


def test_build_train_command_includes_val_and_loss_type(tmp_path):
    cmd = build_train_command(
        train_script=Path("scripts/train_weighted_dsm.py"),
        yaml_path=Path("input.yaml"),
        data_npz=Path("train.npz"),
        val_npz=Path("val.npz"),
        out_dir=tmp_path / "out",
        loss_type="true-quotient",
        epochs=3,
        batch_size=2,
        learning_rate=1e-6,
        beta=0.02,
        max_grad_norm=0.5,
        checkpoint_every=1,
        save_every_batches=0,
        lr_schedule="constant",
        lr_warmup_epochs=0,
        lr_min=1e-7,
        early_stopping_patience=4,
        diffusion_steps=8,
        recycling_steps=1,
        device="cpu",
        cache=Path(".boltz"),
        resume_from=None,
    )

    assert "--loss-type" in cmd
    assert "true-quotient" in cmd
    assert "--val-data" in cmd
    assert "val.npz" in cmd


def test_build_train_command_omits_missing_val_data(tmp_path):
    cmd = build_train_command(
        train_script=Path("scripts/train_weighted_dsm.py"),
        yaml_path=Path("input.yaml"),
        data_npz=Path("train.npz"),
        val_npz=None,
        out_dir=tmp_path / "out",
        loss_type="cartesian",
        epochs=3,
        batch_size=2,
        learning_rate=1e-6,
        beta=0.02,
        max_grad_norm=0.5,
        checkpoint_every=1,
        save_every_batches=0,
        lr_schedule="constant",
        lr_warmup_epochs=0,
        lr_min=1e-7,
        early_stopping_patience=4,
        diffusion_steps=8,
        recycling_steps=1,
        device="cpu",
        cache=Path(".boltz"),
        resume_from=None,
    )

    assert "--val-data" not in cmd

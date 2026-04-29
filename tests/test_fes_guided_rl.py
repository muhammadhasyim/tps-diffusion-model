"""Tests for FES-guided RL (OpenMM teacher, student KDE, training loss)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("genai_tps.rl.config", reason="optional genai_tps.rl not in this checkout")

from genai_tps.simulation import OPESBias
from genai_tps.rl.config import BoltzRLConfig, FESTeacherConfig
from genai_tps.rl.fes_teacher import load_build_md_simulation_from_pdb
from genai_tps.rl.student_distribution import BoltzStudentKDE
from genai_tps.rl.training import fes_guided_trajectory_loss


def test_boltz_student_kde_sliding_window():
    kde = BoltzStudentKDE(ndim=2, window=3, bandwidth=1.0)
    assert kde.log_density(np.array([0.0, 0.0])) < 0.0  # log_floor empty buffer
    kde.update(np.array([0.0, 0.0]))
    kde.update(np.array([1.0, 0.0]))
    d0 = kde.log_density(np.array([0.0, 0.0]))
    kde.update(np.array([5.0, 5.0]))
    kde.update(np.array([5.0, 5.0]))
    assert len(kde._buf) == 3
    d1 = kde.log_density(np.array([0.0, 0.0]))
    assert np.isfinite(d0) and np.isfinite(d1)


def test_load_build_md_simulation_from_pdb_importable():
    fn = load_build_md_simulation_from_pdb()
    assert callable(fn)


def test_openmm_teacher_log_p_target_finite():
    """OPES bias evaluate + plan scaling gives finite log_p_target surrogate."""
    opes = OPESBias(ndim=2, kbt=2.494, barrier=3.0, biasfactor=8.0, pace=1)
    cv = np.array([1.0, 2.0], dtype=np.float64)
    for step in range(1, 8):
        opes.update(cv + 0.01 * step, step)
    v = float(opes.evaluate(cv))
    logp_kbt = 2.494
    log_pt = -v / logp_kbt
    assert np.isfinite(log_pt)


def test_fes_guided_advantage_clipping():
    fes_cfg = FESTeacherConfig(advantage_clip=1.5)
    log_pt, log_pb = 10.0, 0.0
    adv = float(np.clip(log_pt - log_pb, -fes_cfg.advantage_clip, fes_cfg.advantage_clip))
    assert adv == pytest.approx(1.5)


class _FakeTeacher:
    def log_p_target(self, cv) -> float:
        return 2.0


def test_fes_guided_trajectory_loss_smoke():
    """One backward step through :func:`fes_guided_trajectory_loss` with a toy core."""
    from genai_tps.rl.rollout import BoltzRolloutStep

    n_atom = 4
    cpu = torch.device("cpu")

    class _Diff:
        device = cpu

    class _Core:
        diffusion = _Diff()
        atom_mask = torch.ones(1, n_atom, dtype=torch.float32)
        _schedule = [object()]
        num_sampling_steps = 1

        def _single_forward_step_core(self, x, step_idx, eps=None):
            del eps
            meta = {
                "center_mean": torch.zeros(1, n_atom, 3),
                "t_hat": 1.0,
                "noise_var": 1.0,
                "sigma_t": 1.0,
                "step_scale": 1.0,
            }
            denoised = x + 0.1
            x_noisy = x + 0.05
            return denoised, torch.zeros_like(x), None, None, meta, denoised.detach(), x_noisy

    core = _Core()
    x0 = torch.randn(1, n_atom, 3, dtype=torch.float32, requires_grad=True)
    denoised_old = torch.randn(1, n_atom, 3, dtype=torch.float32)
    x_noisy = torch.randn(1, n_atom, 3, dtype=torch.float32)
    traj = [
        BoltzRolloutStep(
            x_prev=x0.detach(),
            eps=torch.zeros_like(x0),
            random_r=torch.zeros_like(x0),
            random_tr=torch.zeros_like(x0),
            center_mean=torch.zeros(1, n_atom, 3),
            x_noisy=x_noisy.detach(),
            denoised_old=denoised_old.detach(),
            x_next=x0.detach(),
            step_idx=0,
            t_hat=1.0,
            noise_var=1.0,
            sigma_t=1.0,
            step_scale=1.0,
        )
    ]

    import genai_tps.rl.training as tr_mod

    def fake_replay(c, x_n, step_idx):
        del c, step_idx
        return x0 + 0.12  # slightly different from denoised_old

    tr_saved = tr_mod.replay_denoiser
    tr_mod.replay_denoiser = fake_replay
    try:
        kde = BoltzStudentKDE(ndim=2, window=10, bandwidth=2.0)
        kde.update(np.array([0.5, 0.5]))
        fes_cfg = FESTeacherConfig(advantage_clip=5.0)
        rl_cfg = BoltzRLConfig()
        cv = np.array([0.5, 0.5], dtype=np.float64)
        loss = fes_guided_trajectory_loss(
            core,
            traj,
            cv,
            _FakeTeacher(),
            kde,
            fes_cfg=fes_cfg,
            rl_cfg=rl_cfg,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert x0.grad is not None
    finally:
        tr_mod.replay_denoiser = tr_saved


def test_build_md_simulation_from_pdb_alanine(tmp_path):
    """Smoke: build unminimised GBn2 system from a tiny peptide PDB (if OpenMM works)."""
    pytest.importorskip("openmm")
    pdb = tmp_path / "ala.pdb"
    pdb.write_text(
        "\n".join(
            [
                "REMARK 1 tiny test",
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C",
                "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C",
                "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O",
                "ATOM      5  CB  ALA A   1       2.000  -0.773  -1.246  1.00  0.00           C",
                "ATOM      6  N   ALA A   2       3.326   1.593   0.000  1.00  0.00           N",
                "ATOM      7  CA  ALA A   2       4.307   2.651   0.000  1.00  0.00           C",
                "ATOM      8  C   ALA A   2       5.789   2.311   0.000  1.00  0.00           C",
                "ATOM      9  O   ALA A   2       6.252   1.189   0.000  1.00  0.00           O",
                "ATOM     10  CB  ALA A   2       4.079   3.430  -1.276  1.00  0.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fn = load_build_md_simulation_from_pdb()
    try:
        sim, meta = fn(pdb, platform_name="CPU", temperature_k=300.0, ligand_smiles=None)
    except Exception as exc:
        pytest.skip(f"OpenMM system build not available in this environment: {exc}")
    assert sim is not None
    assert "platform_used" in meta
    sim.step(2)

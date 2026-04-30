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


def test_build_md_simulation_from_pdb_alanine(ala_ala_pdb_path):
    """Smoke: build unminimised GBn2 system from a tiny peptide PDB (if OpenMM works)."""
    pytest.importorskip("openmm")
    pdb = ala_ala_pdb_path
    fn = load_build_md_simulation_from_pdb()
    try:
        sim, meta = fn(pdb, platform_name="CPU", temperature_k=300.0, ligand_smiles=None)
    except Exception as exc:
        pytest.skip(f"OpenMM system build not available in this environment: {exc}")
    assert sim is not None
    assert "platform_used" in meta
    sim.step(2)


# ---------------------------------------------------------------------------
# Bidirectional FES loop tests
# ---------------------------------------------------------------------------


class TestOPESHeightScale:
    """OPESBias.update with height_scale < 1 produces proportionally smaller kernels."""

    def test_height_scale_reduces_kernel_weight(self):
        opes_full = OPESBias(ndim=1, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, fixed_sigma=1.0)
        opes_half = OPESBias(ndim=1, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, fixed_sigma=1.0)

        opes_full.update(3.0, mc_step=1, height_scale=1.0)
        opes_half.update(3.0, mc_step=1, height_scale=0.5)

        assert opes_full.n_kernels == 1
        assert opes_half.n_kernels == 1
        assert opes_half.kernels[0].height == pytest.approx(
            opes_full.kernels[0].height * 0.5, rel=1e-10
        )

    def test_height_scale_zero_deposits_zero_weight_kernel(self):
        opes = OPESBias(ndim=1, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, fixed_sigma=1.0)
        opes.update(1.0, mc_step=1, height_scale=0.0)
        assert opes.n_kernels == 1
        assert opes.kernels[0].height == pytest.approx(0.0, abs=1e-15)

    def test_default_height_scale_unchanged(self):
        """Calling without height_scale is equivalent to height_scale=1.0."""
        opes_a = OPESBias(ndim=1, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, fixed_sigma=1.0)
        opes_b = OPESBias(ndim=1, kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1, fixed_sigma=1.0)

        opes_a.update(2.0, mc_step=1)
        opes_b.update(2.0, mc_step=1, height_scale=1.0)

        assert opes_a.kernels[0].height == pytest.approx(opes_b.kernels[0].height, rel=1e-12)
        assert opes_a.evaluate(2.0) == pytest.approx(opes_b.evaluate(2.0), rel=1e-12)


class TestSetPositionsFromBoltz:
    """OpenMMTeacher.set_positions_from_boltz roundtrip."""

    @pytest.fixture
    def teacher_and_coords(self, tmp_path):
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
            pytest.skip(f"OpenMM system build not available: {exc}")

        from genai_tps.rl.fes_teacher import build_openmm_indices_for_boltz_atoms

        class _FakeStructure:
            """Minimal structure with atoms array matching the PDB heavy atoms."""

            def __init__(self, n):
                import numpy as _np
                self.atoms = _np.zeros(
                    n,
                    dtype=[("name", "U4"), ("coords", "f4", (3,))],
                )
                names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB"]
                for i, nm in enumerate(names):
                    self.atoms[i]["name"] = nm
                self.residues = _np.zeros(
                    2,
                    dtype=[("atom_idx", "i4"), ("atom_num", "i4"), ("res_idx", "i4")],
                )
                self.residues[0] = (0, 5, 0)
                self.residues[1] = (5, 5, 1)
                self.chains = _np.zeros(
                    1,
                    dtype=[("name", "U4"), ("atom_idx", "i4"), ("atom_num", "i4"),
                           ("res_idx", "i4"), ("res_num", "i4")],
                )
                self.chains[0] = ("A", 0, 10, 0, 2)

        n_heavy = 10
        structure = _FakeStructure(n_heavy)
        omm_idx = build_openmm_indices_for_boltz_atoms(structure, sim.topology)

        class _MinTeacher:
            pass

        teacher = _MinTeacher()
        teacher.sim = sim
        teacher._omm_idx = omm_idx
        teacher._n_boltz = n_heavy
        # Bind the real method
        from genai_tps.rl.fes_teacher import OpenMMTeacher as _OT
        import types
        teacher.set_positions_from_boltz = types.MethodType(_OT.set_positions_from_boltz, teacher)

        return teacher, omm_idx

    def test_roundtrip_positions(self, teacher_and_coords):
        """set -> read back: heavy-atom positions should approximately match."""
        import openmm.unit as unit

        teacher, omm_idx = teacher_and_coords
        new_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 1.4, 0.0],
                [1.3, 2.4, 0.0],
                [2.0, -0.8, -1.2],
                [3.3, 1.6, 0.0],
                [4.3, 2.6, 0.0],
                [5.8, 2.3, 0.0],
                [6.3, 1.2, 0.0],
                [4.1, 3.4, -1.3],
            ],
            dtype=np.float64,
        )
        teacher.set_positions_from_boltz(new_coords, minimize_steps=0)

        state = teacher.sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        for i in range(teacher._n_boltz):
            j = int(omm_idx[i])
            read_back_angstrom = pos_nm[j] * 10.0
            np.testing.assert_allclose(read_back_angstrom, new_coords[i], atol=1e-4)


class TestBidirectionalLoopIntegration:
    """Two-iteration smoke test of the bidirectional loop with mock teacher."""

    def test_two_iterations_mock(self):
        """Wiring test: OPES deposits from both physics and generative, disagreement selection."""
        opes = OPESBias(ndim=2, kbt=2.494, barrier=5.0, biasfactor=10.0, pace=1)

        class _MockTeacher:
            def __init__(self, opes_bias):
                self.opes = opes_bias
                self._md_counter = 0

            def run_md_burst(self, n_steps, deposit_pace):
                out = []
                for _ in range(n_steps):
                    self._md_counter += 1
                    if self._md_counter % deposit_pace == 0:
                        cv = np.array([1.0 + 0.01 * self._md_counter, 2.0])
                        self.opes.update(cv, self._md_counter)
                        out.append((cv.copy(), 1.0))
                return out

            def log_p_target(self, cv):
                v = float(self.opes.evaluate(cv))
                return -v / 2.494

        teacher = _MockTeacher(opes)
        student_kde = BoltzStudentKDE(2, window=50, bandwidth=1.0)
        fes_cfg = FESTeacherConfig(
            md_steps_per_burst=20,
            md_deposit_pace=5,
            boltz_rollouts_per_iter=3,
            n_iters=2,
            generative_deposit_weight=0.2,
            disagreement_warmstart=False,
        )

        gen_counter = 0
        for it in range(1, fes_cfg.n_iters + 1):
            teacher.run_md_burst(fes_cfg.md_steps_per_burst, fes_cfg.md_deposit_pace)

            rng = np.random.default_rng(seed=it)
            cvs = [rng.normal([1.0, 2.0], 0.5) for _ in range(fes_cfg.boltz_rollouts_per_iter)]

            for cv in cvs:
                gen_counter += 1
                teacher.opes.update(
                    cv, gen_counter, height_scale=fes_cfg.generative_deposit_weight,
                )

            for cv in cvs:
                student_kde.update(cv)

            disagreements = [
                abs(teacher.log_p_target(cv) - student_kde.log_density(cv))
                for cv in cvs
            ]
            best_k = int(np.argmax(disagreements))
            assert 0 <= best_k < len(cvs)

        # Verify both physics and generative kernels accumulated
        assert opes.n_kernels > 0
        assert opes.counter > fes_cfg.n_iters * (fes_cfg.md_steps_per_burst // fes_cfg.md_deposit_pace)
        assert len(student_kde._buf) == fes_cfg.n_iters * fes_cfg.boltz_rollouts_per_iter

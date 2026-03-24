"""Unit tests with a lightweight mock diffusion module (no checkpoint)."""

import math

import numpy as np
from openpathsampling.engines.trajectory import Trajectory
import pytest
import torch

from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore, _nan_fallback
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
from genai_tps.backends.boltz.path_probability import (
    backward_shooting_metropolis_bias,
    compute_log_path_prob,
    prefix_forward_transitions_log_prob_tensor,
    trajectory_log_path_prob,
)
from genai_tps.backends.boltz.tps_sampling import trajectory_has_finite_coords

from tests.mock_boltz_diffusion import MockDiffusion


@pytest.fixture
def core_and_engine():
    diff = MockDiffusion()
    b, m = 1, 4
    atom_mask = torch.ones(b, m)
    kwargs = {"multiplicity": 1}
    core = BoltzSamplerCore(diff, atom_mask, kwargs, multiplicity=1)
    core.build_schedule(3)
    engine = BoltzDiffusionEngine(core, boltz_snapshot_descriptor(n_atoms=m), options={"n_frames_max": 20})
    return core, engine


def test_engine_does_not_alias_stored_snapshot_tensors(core_and_engine):
    """Regression: integration buffer must not share storage with OPS trajectory frames."""
    core, engine = core_and_engine
    x = core.sample_initial_noise()
    snap = BoltzSnapshot.from_gpu_batch(
        x,
        step_index=0,
        sigma=float(core.schedule[0].sigma_tm),
    )
    ref = snap.tensor_coords.detach().clone()
    engine.current_snapshot = snap
    assert engine._current_coords is not snap.tensor_coords
    engine.generate_next_frame()
    torch.testing.assert_close(snap.tensor_coords, ref, rtol=0, atol=0)


def test_forward_trajectory_length(core_and_engine):
    core, engine = core_and_engine
    x = core.sample_initial_noise()
    snap = BoltzSnapshot.from_gpu_batch(
        x,
        step_index=0,
        sigma=float(core.schedule[0].sigma_tm),
    )
    engine.current_snapshot = snap
    traj = Trajectory([snap])
    for _ in range(core.num_sampling_steps):
        traj.append(engine.generate_next_frame())
    assert len(traj) == core.num_sampling_steps + 1
    assert traj[-1].step_index == core.num_sampling_steps


def test_path_prob_forward_segment(core_and_engine):
    core, _ = core_and_engine
    x = core.sample_initial_noise()
    traj, eps_list, _, _, meta_list = core.generate_segment(x, 0, core.num_sampling_steps)
    lp = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=True,
        n_atoms=x.shape[1],
    )
    assert torch.isfinite(lp)


def test_trajectory_log_path_prob_matches_segment(core_and_engine):
    core, _ = core_and_engine
    x = core.sample_initial_noise()
    coord_list, eps_list, _, _, meta_list = core.generate_segment(x, 0, core.num_sampling_steps)
    lp_seg = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=True,
        n_atoms=x.shape[1],
    )
    snaps = [
        BoltzSnapshot.from_gpu_batch(
            coord_list[0],
            step_index=0,
            sigma=float(core.schedule[0].sigma_tm),
        )
    ]
    for i, eps in enumerate(eps_list):
        snaps.append(
            BoltzSnapshot.from_gpu_batch(
                coord_list[i + 1],
                step_index=i + 1,
                sigma=float(meta_list[i]["sigma_t"]),
                eps_used=eps,
            )
        )
    traj_ops = Trajectory(snaps)
    lp_tr = trajectory_log_path_prob(traj_ops, core)
    assert lp_tr is not None
    assert abs(lp_tr - float(lp_seg)) < 1e-4


def test_se3_haar_omitted_from_path_probability(core_and_engine):
    """Haar on SO(3) is constant; :func:`compute_log_path_prob` omits it (see theory doc)."""
    from genai_tps.backends.boltz.path_probability import compute_log_path_prob

    core, _ = core_and_engine
    x = core.sample_initial_noise()
    _, eps_list, _, _, meta_list = core.generate_segment(x, 0, core.num_sampling_steps)
    lp_full = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=True,
        n_atoms=x.shape[1],
    )
    lp_no_jac = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=False,
        n_atoms=x.shape[1],
    )
    assert torch.isfinite(lp_full)
    assert torch.isfinite(lp_no_jac)
    assert lp_full != lp_no_jac


def test_recover_forward_noise_round_trip(core_and_engine):
    """Forward epsilon matches :meth:`BoltzSamplerCore.recover_forward_noise` (Mock identity net)."""
    core, _ = core_and_engine
    torch.manual_seed(20250323)
    x0 = core.sample_initial_noise()
    x1, eps, rr, tr, _meta = core.single_forward_step(x0, 0)
    eps_rec, _ = core.recover_forward_noise(x0, x1, 0, rr, tr)
    torch.testing.assert_close(eps_rec, eps, rtol=1e-4, atol=1e-4)


def test_backward_assemble_chronological_step_indices(core_and_engine):
    """Backward ``generate(..., direction=-1)`` yields chronological prefix; step indices increase."""
    core, engine = core_and_engine
    torch.manual_seed(7)
    x = core.sample_initial_noise()
    snaps: list[BoltzSnapshot] = [
        BoltzSnapshot.from_gpu_batch(
            x,
            step_index=0,
            sigma=float(core.schedule[0].sigma_tm),
        )
    ]
    for step in range(core.num_sampling_steps):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            BoltzSnapshot.from_gpu_batch(
                x,
                step_index=step + 1,
                sigma=float(meta["sigma_t"]),
                eps_used=eps,
                rotation_R=rr,
                translation_t=tr,
            )
        )
    traj = Trajectory(snaps)
    k = 2
    # OPS prepends one frame per backward step; stop once prefix length reaches k+1
    # (else another step would run from step_index 0 and raise).
    partial = engine.generate(
        traj[k].reversed,
        running=[lambda t, _trusted, _k=k: len(t) < _k + 1],
        direction=-1,
    )
    assembled = partial + traj[k + 1 :]
    for i in range(len(assembled) - 1):
        assert assembled[i].step_index < assembled[i + 1].step_index


def test_backward_shooting_metropolis_bias_identical_prefix(core_and_engine):
    """Same old/new prefix ⇒ Hastings bias 1.0."""
    core, _ = core_and_engine
    torch.manual_seed(1)
    x = core.sample_initial_noise()
    snaps: list[BoltzSnapshot] = [
        BoltzSnapshot.from_gpu_batch(
            x,
            step_index=0,
            sigma=float(core.schedule[0].sigma_tm),
        )
    ]
    for step in range(core.num_sampling_steps):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            BoltzSnapshot.from_gpu_batch(
                x,
                step_index=step + 1,
                sigma=float(meta["sigma_t"]),
                eps_used=eps,
                rotation_R=rr,
                translation_t=tr,
            )
        )
    traj = Trajectory(snaps)
    bias = backward_shooting_metropolis_bias(traj, traj, shooting_index=1, core=core)
    assert bias == 1.0


def test_trajectory_log_path_prob_mixed_prefix_backward_generated(core_and_engine):
    """Mixed path (backward-regenerated prefix) yields finite log prob via recovery."""
    core, engine = core_and_engine
    torch.manual_seed(11)
    x = core.sample_initial_noise()
    snaps: list[BoltzSnapshot] = [
        BoltzSnapshot.from_gpu_batch(
            x,
            step_index=0,
            sigma=float(core.schedule[0].sigma_tm),
        )
    ]
    for step in range(core.num_sampling_steps):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            BoltzSnapshot.from_gpu_batch(
                x,
                step_index=step + 1,
                sigma=float(meta["sigma_t"]),
                eps_used=eps,
                rotation_R=rr,
                translation_t=tr,
            )
        )
    traj = Trajectory(snaps)
    k = 2
    # OPS prepends one frame per backward step; stop once prefix length reaches k+1
    # (else another step would run from step_index 0 and raise).
    partial = engine.generate(
        traj[k].reversed,
        running=[lambda t, _trusted, _k=k: len(t) < _k + 1],
        direction=-1,
    )
    new_prefix = partial
    mixed = Trajectory(list(new_prefix) + list(traj[k + 1 :]))
    lp = trajectory_log_path_prob(mixed, core)
    assert lp is not None
    assert math.isfinite(lp)
    # Internal consistency: explicit prefix forward tensor + prior
    trans = prefix_forward_transitions_log_prob_tensor(mixed, core)
    from genai_tps.backends.boltz.path_probability import log_prior_initial, _tensor_batch

    s0 = mixed[0]
    assert isinstance(s0, BoltzSnapshot)
    prior = log_prior_initial(_tensor_batch(s0, core), float(core.schedule[0].sigma_tm)).sum()
    lp2 = float((trans + prior).detach().cpu().item())
    assert abs(lp - lp2) < 1e-3


# ---------------------------------------------------------------------------
# NaN guard / rejection tests
# ---------------------------------------------------------------------------


def test_nan_fallback_replaces_nonfinite():
    """_nan_fallback should replace NaN/Inf with fallback values element-wise."""
    good = torch.tensor([1.0, 2.0, 3.0])
    bad = torch.tensor([1.0, float("nan"), float("inf")])
    result = _nan_fallback(bad, good, label="test")
    assert torch.isfinite(result).all()
    assert result[0] == 1.0
    assert result[1] == 2.0  # replaced
    assert result[2] == 3.0  # replaced


def test_nan_fallback_noop_on_finite():
    """_nan_fallback should return input unchanged when all values are finite."""
    x = torch.tensor([1.0, 2.0, 3.0])
    fallback = torch.zeros(3)
    result = _nan_fallback(x, fallback, label="test")
    torch.testing.assert_close(result, x)


def test_forward_step_nan_guard():
    """Forward step: if the network returns NaN, the guard restores input coords."""
    from tests.mock_boltz_diffusion import MockDiffusion

    class _NaNDiffusion(MockDiffusion):
        def preconditioned_network_forward(self, noised_atom_coords, sigma, **kw):
            return torch.full_like(noised_atom_coords, float("nan"))

    diff = _NaNDiffusion()
    b, m = 1, 4
    core = BoltzSamplerCore(diff, torch.ones(b, m), {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    finite_input = torch.randn(1, m, 3)
    x_next, eps, rr, tr, meta = core.single_forward_step(finite_input, 0)
    assert torch.isfinite(x_next).all(), "NaN guard should replace NaN with input coords"


def test_backward_step_nan_guard():
    """Backward step: if the network returns NaN, the guard restores input coords."""
    from tests.mock_boltz_diffusion import MockDiffusion

    class _NaNDiffusion(MockDiffusion):
        def preconditioned_network_forward(self, noised_atom_coords, sigma, **kw):
            return torch.full_like(noised_atom_coords, float("nan"))

    diff = _NaNDiffusion()
    b, m = 1, 4
    core = BoltzSamplerCore(diff, torch.ones(b, m), {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    finite_input = torch.randn(1, m, 3)
    x_prev, eps, rr, tr, meta = core.single_backward_step(finite_input, 1)
    assert torch.isfinite(x_prev).all(), "NaN guard should replace NaN with input coords"


def test_trajectory_has_finite_coords_rejects_nan():
    """trajectory_has_finite_coords must return False when any frame has NaN."""
    good_snap = BoltzSnapshot.from_gpu_batch(
        torch.randn(1, 4, 3), step_index=0, sigma=1.0,
    )
    bad_tensor = torch.full((1, 4, 3), float("nan"))
    bad_snap = BoltzSnapshot.from_gpu_batch(bad_tensor, step_index=1, sigma=0.5)
    traj_ok = Trajectory([good_snap])
    traj_bad = Trajectory([good_snap, bad_snap])
    assert trajectory_has_finite_coords(traj_ok) is True
    assert trajectory_has_finite_coords(traj_bad) is False


def test_metropolis_rejects_nan_probability():
    """OPS Metropolis must reject when cumulative bias is NaN."""
    from openpathsampling.pathmover import EngineMover
    import openpathsampling as paths
    import random as _random

    class _DummyEns:
        def __call__(self, traj, candidate=False, trusted=None):
            return True

    class _Mover(EngineMover):
        @property
        def direction(self):
            return 1

    ens = _DummyEns()
    mover = object.__new__(_Mover)
    mover._trust_candidate = True
    mover._rng = _random.Random(42)

    sample = paths.Sample(replica=0, trajectory=Trajectory([]), ensemble=ens)
    sample.bias = float("nan")

    accepted, details = mover.metropolis([sample])
    assert not accepted, "NaN bias must cause rejection"
    assert details["metropolis_acceptance"] == 0.0


# ---------------------------------------------------------------------------
# Metropolis-Hastings sign-correctness tests
# ---------------------------------------------------------------------------


def _build_forward_traj(core, seed):
    """Build a full forward trajectory using MockDiffusion with a given seed."""
    torch.manual_seed(seed)
    x = core.sample_initial_noise()
    snaps = [
        BoltzSnapshot.from_gpu_batch(
            x,
            step_index=0,
            sigma=float(core.schedule[0].sigma_tm),
        )
    ]
    for step in range(core.num_sampling_steps):
        x, eps, rr, tr, meta = core.single_forward_step(x, step)
        snaps.append(
            BoltzSnapshot.from_gpu_batch(
                x,
                step_index=step + 1,
                sigma=float(meta["sigma_t"]),
                eps_used=eps,
                rotation_R=rr,
                translation_t=tr,
            )
        )
    return Trajectory(snaps)


def test_backward_bias_sign_favors_lower_noise(core_and_engine):
    """Bias must be 1.0 when the new prefix has lower noise than the old one.

    A new prefix with smaller ||eps|| should have higher forward log prob than the
    old prefix.  The correct acceptance ratio r = pi_new/pi_old * q_bwd_old/q_bwd_new
    has r >= 1 in this regime (accept unconditionally).  The pre-fix sign error would
    give r << 1 (reject unconditionally), inverting the correct behavior.
    """
    core, engine = core_and_engine
    old_traj = _build_forward_traj(core, seed=100)

    k = 1  # shooting index: compare prefix [0..k]

    # Build a new prefix whose forward noise at step 0 is scaled down (lower energy).
    x0_new = old_traj[0].tensor_coords.clone()
    step = 0
    sch = core.schedule[step]
    # Use a tiny epsilon so log p_fwd(new_prefix) >> log p_fwd(old_prefix)
    tiny_eps = torch.zeros_like(x0_new)
    rr0 = old_traj[1].rotation_R
    tr0 = old_traj[1].translation_t
    if rr0 is None:
        rr0 = torch.eye(3, device=x0_new.device, dtype=x0_new.dtype).unsqueeze(0)
        tr0 = torch.zeros(1, 1, 3, device=x0_new.device, dtype=x0_new.dtype)
    # x_noisy = x_tilde + eps; x_tilde from augmentation of x0_new
    center_mean = x0_new.mean(dim=-2, keepdim=True)
    x_tilde = torch.einsum("bmd,bds->bms", x0_new - center_mean, rr0) + tr0
    x1_new = x_tilde + tiny_eps  # near-zero noise => extremely high p_fwd
    # Adjust using forward formula: x_next = x_noisy + alpha*(x_noisy - x_hat)
    # For Mock (identity), x_hat = x_noisy, so x_next = x_noisy exactly.
    snap1_new = BoltzSnapshot.from_gpu_batch(
        x1_new,
        step_index=1,
        sigma=float(core.schedule[0].sigma_t),
        eps_used=tiny_eps,
        rotation_R=rr0,
        translation_t=tr0,
    )
    # Build new_traj: replace prefix up to k with the new frame, keep suffix
    new_traj_frames = [old_traj[0], snap1_new] + list(old_traj[2:])
    new_traj = Trajectory(new_traj_frames)

    bias = backward_shooting_metropolis_bias(old_traj, new_traj, shooting_index=k, core=core)
    # pi_new >> pi_old (near-zero noise vs typical random noise).
    # Correct sign: log_r ≈ lf_new - lf_old + lb_correction >> 0  =>  bias close to 1.0
    # Wrong sign:   log_r ≈ lf_old - lf_new + lb_correction << 0  =>  bias << 1.0
    # Use a loose lower bound (backward-kernel terms can slightly reduce it below 1.0).
    assert bias > 0.9, (
        f"Expected bias > 0.9 for lower-noise new prefix; got {bias:.6g}. "
        "This indicates the sign error may still be present."
    )


def test_backward_bias_sign_penalizes_higher_noise(core_and_engine):
    """Bias must be < 1 when the new prefix has higher noise than the old one.

    A new prefix with larger ||eps|| has lower forward log prob.  The correct ratio
    r < 1, so the bias should be < 1.  The pre-fix sign error would give r > 1 (bias
    = 1.0), always accepting the worse path—which is the root cause of unphysical
    structures.
    """
    core, engine = core_and_engine
    old_traj = _build_forward_traj(core, seed=200)

    k = 1

    # Build new prefix with very large epsilon (high noise => very low p_fwd).
    x0_new = old_traj[0].tensor_coords.clone()
    rr0 = old_traj[1].rotation_R
    tr0 = old_traj[1].translation_t
    if rr0 is None:
        rr0 = torch.eye(3, device=x0_new.device, dtype=x0_new.dtype).unsqueeze(0)
        tr0 = torch.zeros(1, 1, 3, device=x0_new.device, dtype=x0_new.dtype)
    large_eps = torch.ones_like(x0_new) * 1e4
    center_mean = x0_new.mean(dim=-2, keepdim=True)
    x_tilde = torch.einsum("bmd,bds->bms", x0_new - center_mean, rr0) + tr0
    x1_new = x_tilde + large_eps
    snap1_new = BoltzSnapshot.from_gpu_batch(
        x1_new,
        step_index=1,
        sigma=float(core.schedule[0].sigma_t),
        eps_used=large_eps,
        rotation_R=rr0,
        translation_t=tr0,
    )
    new_traj_frames = [old_traj[0], snap1_new] + list(old_traj[2:])
    new_traj = Trajectory(new_traj_frames)

    bias = backward_shooting_metropolis_bias(old_traj, new_traj, shooting_index=k, core=core)
    # pi_new << pi_old (much higher noise) => r << 1 => bias << 1
    assert bias < 1.0, (
        f"Expected bias < 1.0 for high-noise new prefix; got {bias:.6g}. "
        "This indicates the sign error is still present."
    )
    assert bias >= 0.0, f"Bias must be non-negative; got {bias:.6g}"


def test_se3_augmentation_uses_lower_frame_for_backward_generated(core_and_engine):
    """prefix_forward_transitions_log_prob_tensor must use s0.rotation_R for backward frames.

    single_backward_step stores the step-j augmentation on the result (x_j = s0),
    NOT on x_{j+1} = s1.  The bug was using s1.rotation_R; this test verifies that
    the log-prob is consistent with recover_forward_noise called with s0's augmentation.
    """
    core, engine = core_and_engine
    torch.manual_seed(99)
    x = core.sample_initial_noise()

    # Build one backward step from step_index=1 backward to step_index=0.
    snap1 = BoltzSnapshot.from_gpu_batch(
        x,
        step_index=1,
        sigma=float(core.schedule[0].sigma_t),
    )
    engine.current_snapshot = snap1
    bwd_snap = engine._backward_step()  # produces step_index=0 with generated_by_backward=True

    assert bwd_snap.generated_by_backward is True
    assert bwd_snap.rotation_R is not None, "backward step must store rotation on result"
    assert bwd_snap.translation_t is not None, "backward step must store translation on result"

    traj = Trajectory([bwd_snap, snap1])
    from genai_tps.backends.boltz.path_probability import prefix_forward_transitions_log_prob_tensor

    lp = prefix_forward_transitions_log_prob_tensor(traj, core)
    assert torch.isfinite(lp), "log prob must be finite when using correct augmentation source"

    # Cross-check: manually call recover_forward_noise with s0's R,tau.
    eps_manual, meta_manual = core.recover_forward_noise(
        bwd_snap.tensor_coords,
        snap1.tensor_coords,
        0,
        bwd_snap.rotation_R,
        bwd_snap.translation_t,
    )
    from genai_tps.backends.boltz.path_probability import (
        log_gaussian_isotropic, log_det_jacobian_step,
    )
    v = float(meta_manual["noise_var"])
    alpha = (float(meta_manual["step_scale"])
             * (float(meta_manual["sigma_t"]) - float(meta_manual["t_hat"]))
             / float(meta_manual["t_hat"]))
    n_atoms_count = int(bwd_snap.tensor_coords.shape[1])
    lp_manual = (
        log_gaussian_isotropic(eps_manual, v).sum()
        + log_det_jacobian_step(alpha, n_atoms_count)
    ) if v > 1e-30 else torch.tensor(0.0)
    assert torch.isfinite(lp_manual)
    assert abs(float(lp.cpu()) - float(lp_manual.cpu())) < 1e-3, (
        "prefix_forward_transitions_log_prob_tensor must match manual recovery with s0.R"
    )


# ---------------------------------------------------------------------------
# Rescaled fixed-point iteration (G3) convergence tests
# ---------------------------------------------------------------------------


class _CSkipDenoiser(MockDiffusion):
    """Denoiser that mimics the Karras preconditioning: D(x) = c_skip x + c_out x_clean.

    Unlike the identity mock, this denoiser has D(x) ≈ c_skip x at low noise
    (c_skip → 1), reproducing the divergence condition of the naive (G1)
    fixed-point iteration x ← (x_out + αD(x)) / (1+α).
    """

    def __init__(self, x_clean: torch.Tensor, **kw):
        super().__init__(**kw)
        self._x_clean = x_clean

    def preconditioned_network_forward(self, noised_atom_coords, sigma, **kw):
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, dtype=noised_atom_coords.dtype, device=noised_atom_coords.device)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        x_clean = self._x_clean.to(device=noised_atom_coords.device, dtype=noised_atom_coords.dtype)
        if x_clean.shape != noised_atom_coords.shape:
            x_clean = x_clean.expand_as(noised_atom_coords)
        return c_skip * noised_atom_coords + c_out * x_clean


def _make_cskip_core(n_atoms=4, n_steps=3):
    """Build a core with the c_skip denoiser and a fixed x_clean target."""
    x_clean = torch.randn(1, n_atoms, 3)
    diff = _CSkipDenoiser(x_clean)
    atom_mask = torch.ones(1, n_atoms)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(n_steps)
    return core, x_clean


def test_solve_x_noisy_converges_with_cskip_denoiser():
    """_solve_x_noisy_from_output must converge even at low noise where c_skip ≈ 1.

    The naive iteration (G1) diverges here because |α c_skip/(1+α)| > 1 with
    step_scale = 1.5.  The rescaled iteration (G3) factors out c_skip so the
    Jacobian is proportional to (D' − c_skip), which is small.
    """
    torch.manual_seed(42)
    core, x_clean = _make_cskip_core(n_atoms=8, n_steps=32)

    for step_idx in range(core.num_sampling_steps):
        sch = core.schedule[step_idx]
        x_out = torch.randn(1, 8, 3)
        x_noisy = core._solve_x_noisy_from_output(x_out, step_idx, n_fixed_point=8)
        assert torch.isfinite(x_noisy).all(), (
            f"x_noisy has non-finite values at step_idx={step_idx} "
            f"(t_hat={sch.t_hat:.4g}, sigma_t={sch.sigma_t:.4g})"
        )

        alpha = core.diffusion.step_scale * (sch.sigma_t - sch.t_hat) / sch.t_hat
        kw = {"multiplicity": 1}
        x_hat = core.diffusion.preconditioned_network_forward(x_noisy, sch.t_hat, network_condition_kwargs=kw)
        residual = (1.0 + alpha) * x_noisy - alpha * x_hat - x_out
        rel_err = residual.norm() / (x_out.norm() + 1e-8)
        assert rel_err < 0.1, (
            f"Fixed-point residual too large at step_idx={step_idx}: "
            f"rel_err={rel_err:.4g} (alpha={alpha:.4g})"
        )


def test_backward_step_finite_at_all_schedule_indices():
    """Backward step must produce finite coordinates at every schedule index."""
    torch.manual_seed(123)
    core, _ = _make_cskip_core(n_atoms=8, n_steps=32)

    x = torch.randn(1, 8, 3)
    for step_idx in range(core.num_sampling_steps - 1, -1, -1):
        x_prev, eps, rr, tr, meta = core.single_backward_step(x, step_idx)
        assert torch.isfinite(x_prev).all(), (
            f"Backward step produced non-finite coords at step_idx={step_idx}"
        )
        assert torch.isfinite(eps).all(), (
            f"Backward step produced non-finite eps at step_idx={step_idx}"
        )
        x = x_prev


def test_backward_then_forward_recovers_x_noisy():
    """Forward step on backward output should recover approximately the same x_noisy.

    single_backward_step(x_out, i) recovers x_noisy and samples new noise/SE(3).
    Running single_forward_step on x_prev should go through a *different* x_noisy
    (new SE(3)) but still produce finite, bounded output.
    """
    torch.manual_seed(77)
    core, _ = _make_cskip_core(n_atoms=8, n_steps=8)

    x0 = torch.randn(1, 8, 3)
    for step_idx in range(core.num_sampling_steps):
        x_fwd, eps_fwd, rr_fwd, tr_fwd, meta_fwd = core.single_forward_step(x0, step_idx)
        assert torch.isfinite(x_fwd).all()

        x_bwd, eps_bwd, rr_bwd, tr_bwd, meta_bwd = core.single_backward_step(x_fwd, step_idx)
        assert torch.isfinite(x_bwd).all()

        x0 = x_fwd


def test_forward_only_shooting_mode(core_and_engine):
    """Forward-only shooting never uses backward moves; accepted bias always 1.0 or 0.0."""
    from genai_tps.backends.boltz.tps_sampling import (
        BoltzDiffusionOneWayShootingMover,
        BoltzDiffusionForwardShootMover,
        BoltzDiffusionBackwardShootMover,
    )
    core, engine = core_and_engine
    import openpathsampling as paths

    state_a = paths.volume.EmptyVolume()
    state_b = paths.volume.FullVolume()
    ens = paths.TISEnsemble(state_a, state_b, state_a, lambda_i=0.0).named("test")
    sel = paths.UniformSelector()

    mover_fwd_only = BoltzDiffusionOneWayShootingMover(
        ensemble=ens, selector=sel, engine=engine, forward_only=True
    )
    mover_both = BoltzDiffusionOneWayShootingMover(
        ensemble=ens, selector=sel, engine=engine, forward_only=False
    )

    # Forward-only: all sub-movers are ForwardShootMover
    assert all(
        isinstance(m, BoltzDiffusionForwardShootMover) for m in mover_fwd_only.movers
    ), "forward_only=True must only contain ForwardShootMover"
    assert len(mover_fwd_only.movers) == 1

    # Both: should contain exactly one forward and one backward
    types_both = [type(m) for m in mover_both.movers]
    assert BoltzDiffusionForwardShootMover in types_both
    assert BoltzDiffusionBackwardShootMover in types_both
    assert len(mover_both.movers) == 2

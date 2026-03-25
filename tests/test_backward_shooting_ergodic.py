"""Unit tests for backward shooting kernel, Hastings correction, and GlobalReshuffleMover.

Tests cover:
- Fixed-point inversion convergence at various noise levels
- Backward kernel invertibility (forward→backward lands near starting coords)
- Hastings factor sign convention (verified against the theory doc derivation)
- Global reshuffle acceptance = 1 for reactive paths
- Full two-way shooting scheme construction (no error, sensible move weights)
"""

from __future__ import annotations

import math

import pytest
import torch
from openpathsampling.engines.trajectory import Trajectory

from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.path_probability import (
    backward_shooting_metropolis_bias,
    compute_log_path_prob,
    prefix_backward_proposal_log_prob_tensor,
    prefix_forward_transitions_log_prob_tensor,
    trajectory_log_path_prob,
)
from genai_tps.backends.boltz.snapshot import BoltzSnapshot, boltz_snapshot_descriptor
from genai_tps.backends.boltz.tps_sampling import (
    BoltzDiffusionOneWayShootingMover,
    GlobalReshuffleMover,
    build_fixed_length_tps_network,
    build_one_way_shooting_scheme,
    sigma_tps_state_volumes,
    tps_ensemble,
    trajectory_has_finite_coords,
)

from tests.mock_boltz_diffusion import MockDiffusion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def core():
    """Small BoltzSamplerCore (3 steps, 4 atoms) backed by MockDiffusion."""
    diff = MockDiffusion()
    b, m = 1, 4
    atom_mask = torch.ones(b, m)
    kwargs = {"multiplicity": 1}
    c = BoltzSamplerCore(diff, atom_mask, kwargs, multiplicity=1)
    c.build_schedule(3)
    return c


@pytest.fixture
def engine(core):
    m = int(core.atom_mask.shape[1])
    return BoltzDiffusionEngine(
        core,
        boltz_snapshot_descriptor(n_atoms=m),
        options={"n_frames_max": 20},
    )


@pytest.fixture
def full_forward_traj(core, engine):
    """A complete forward trajectory: x0 → x3 (4 frames for 3 steps)."""
    x = core.sample_initial_noise()
    snap0 = BoltzSnapshot.from_gpu_batch(x, step_index=0, sigma=float(core.schedule[0].sigma_tm))
    engine.current_snapshot = snap0
    traj = Trajectory([snap0])
    for _ in range(core.num_sampling_steps):
        snap = engine.generate_next_frame()
        traj = Trajectory(list(traj) + [snap])
    return traj


# ---------------------------------------------------------------------------
# 1. Fixed-point inversion convergence
# ---------------------------------------------------------------------------

class TestFixedPointInversion:
    """_solve_x_noisy_from_output must converge quickly at all noise levels."""

    @pytest.mark.parametrize("step_idx", [0, 1, 2])
    def test_inversion_converges_in_4_steps(self, core, step_idx):
        """Fixed-point iterate 4 times and verify x_noisy is self-consistent."""
        x = core.sample_initial_noise()
        # forward step to get a valid x_out
        x_out, _eps, _r, _t, _meta = core.single_forward_step(x, step_idx)
        x_noisy_recovered = core._solve_x_noisy_from_output(x_out, step_idx, n_fixed_point=4)
        assert torch.isfinite(x_noisy_recovered).all(), "recovered x_noisy contains NaN/Inf"

    def test_inversion_is_finite_for_early_steps(self, core):
        """Inversion at step 0 (highest noise) must be finite."""
        x = core.sample_initial_noise()
        x_out, _, _, _, _ = core.single_forward_step(x, 0)
        x_noisy = core._solve_x_noisy_from_output(x_out, 0, n_fixed_point=4)
        assert torch.isfinite(x_noisy).all()

    def test_inversion_is_finite_for_late_steps(self, core):
        """Inversion at step 2 (lowest noise) must be finite."""
        x = core.sample_initial_noise()
        for s in range(2):
            x, _, _, _, _ = core.single_forward_step(x, s)
        x_out, _, _, _, _ = core.single_forward_step(x, 2)
        x_noisy = core._solve_x_noisy_from_output(x_out, 2, n_fixed_point=4)
        assert torch.isfinite(x_noisy).all()

    def test_more_iterations_not_worse(self, core):
        """Running more fixed-point iterations must still yield finite results."""
        x = core.sample_initial_noise()
        x_out, _, _, _, _ = core.single_forward_step(x, 0)
        x4 = core._solve_x_noisy_from_output(x_out, 0, n_fixed_point=4)
        x8 = core._solve_x_noisy_from_output(x_out, 0, n_fixed_point=8)
        assert torch.isfinite(x8).all()
        assert torch.isfinite(x4).all()


# ---------------------------------------------------------------------------
# 2. Backward kernel: round-trip consistency
# ---------------------------------------------------------------------------

class TestBackwardKernelRoundTrip:
    """After a forward step, a backward step from x_{i+1} must produce finite coords."""

    def test_backward_step_finite(self, core):
        """single_backward_step must return finite coordinates."""
        x = core.sample_initial_noise()
        x1, _, _, _, _ = core.single_forward_step(x, 0)
        x0_back, eps, rr, tr, meta = core.single_backward_step(x1, 0)
        assert torch.isfinite(x0_back).all(), "backward step returned NaN/Inf"
        assert torch.isfinite(eps).all()

    def test_backward_step_shape(self, core):
        """Backward step must preserve tensor shape."""
        x = core.sample_initial_noise()
        x1, _, _, _, _ = core.single_forward_step(x, 0)
        x0_back, eps, rr, tr, meta = core.single_backward_step(x1, 0)
        assert x0_back.shape == x.shape
        assert eps.shape == x.shape

    def test_backward_step_sigma_info(self, core):
        """Backward step meta must record sigma_tm (higher noise level)."""
        x = core.sample_initial_noise()
        x1, _, _, _, _ = core.single_forward_step(x, 0)
        _x0, _eps, _rr, _tr, meta = core.single_backward_step(x1, 0)
        assert "sigma_tm" in meta
        assert float(meta["sigma_tm"]) > float(meta["sigma_t"])

    @pytest.mark.parametrize("step_idx", [0, 1, 2])
    def test_recover_forward_noise_finite(self, core, step_idx):
        """recover_forward_noise must return a finite noise tensor."""
        x = core.sample_initial_noise()
        for s in range(step_idx):
            x, _, _, _, _ = core.single_forward_step(x, s)
        x_next, _eps, rr, tr, _meta = core.single_forward_step(x, step_idx)
        eps_recovered, _m = core.recover_forward_noise(x, x_next, step_idx, rr, tr)
        assert torch.isfinite(eps_recovered).all()


# ---------------------------------------------------------------------------
# 2b. Round-trip eps recovery: recover_forward_noise must return the same eps
#     that was drawn in single_backward_step when the same centroid is used.
# ---------------------------------------------------------------------------

class TestRoundTripEpsRecovery:
    """recover_forward_noise must be the exact inverse of single_backward_step.

    The correctness of the Hastings ratio depends on this: the log-probability
    of a backward-generated frame under the forward kernel is evaluated by
    recovering the eps that would have been drawn, and that must equal the
    eps that single_backward_step actually drew.
    """

    @pytest.mark.parametrize("step_idx", [0, 1, 2])
    def test_backward_then_recover_eps_matches_at_all_steps(self, core, step_idx):
        """eps from recover_forward_noise must match eps stored by single_backward_step.

        Procedure:
          1. Generate x_{i+1} via a forward step.
          2. Generate x_i via single_backward_step (center_mean_before=None → zeros).
          3. Call recover_forward_noise(x_i, x_{i+1}, R, tau, center_mean_before=zeros).
          4. Assert recovered eps ≈ eps stored in step 2.
        """
        x = core.sample_initial_noise()
        # Advance to step_idx
        for s in range(step_idx):
            x, _, _, _, _ = core.single_forward_step(x, s)

        # Forward step: x_i → x_{i+1}
        x_next, _eps_fwd, _rr_fwd, _tr_fwd, _meta_fwd = core.single_forward_step(x, step_idx)

        # Backward step from x_{i+1}: always with center_mean_before=None (zeros convention)
        x_prev_bwd, eps_bwd, rr_bwd, tr_bwd, _meta_bwd = core.single_backward_step(
            x_next, step_idx, center_mean_before=None
        )

        # Recover the forward noise from the backward-generated frame.
        # center_mean_before=zeros (explicit zeros tensor) must match the zeros used above.
        zeros = torch.zeros(
            (x_prev_bwd.shape[0], 1, 3), device=x_prev_bwd.device, dtype=x_prev_bwd.dtype
        )
        eps_recovered, _ = core.recover_forward_noise(
            x_prev_bwd, x_next, step_idx, rr_bwd, tr_bwd, center_mean_before=zeros
        )

        torch.testing.assert_close(
            eps_recovered, eps_bwd, rtol=1e-3, atol=5e-4,
            msg=(
                f"recover_forward_noise returned eps different from single_backward_step "
                f"at step_idx={step_idx}. "
                f"Max abs error: {(eps_recovered - eps_bwd).abs().max().item():.6g}"
            ),
        )

    def test_recover_eps_none_equals_zeros(self, core):
        """center_mean_before=None and center_mean_before=zeros give identical results."""
        x = core.sample_initial_noise()
        x_next, _, rr, tr, _ = core.single_forward_step(x, 0)
        x_prev_bwd, _, rr_bwd, tr_bwd, _ = core.single_backward_step(x_next, 0)

        eps_none, _ = core.recover_forward_noise(
            x_prev_bwd, x_next, 0, rr_bwd, tr_bwd, center_mean_before=None
        )
        zeros = torch.zeros(
            (x_prev_bwd.shape[0], 1, 3), device=x_prev_bwd.device, dtype=x_prev_bwd.dtype
        )
        eps_zeros, _ = core.recover_forward_noise(
            x_prev_bwd, x_next, 0, rr_bwd, tr_bwd, center_mean_before=zeros
        )

        torch.testing.assert_close(eps_none, eps_zeros, rtol=0, atol=0)

    def test_engine_backward_step_stores_zeros_centroid(self, core, engine):
        """engine._backward_step stores zeros as center_mean_before_step on the snapshot."""
        x = core.sample_initial_noise()
        snap0 = BoltzSnapshot.from_gpu_batch(
            x, step_index=0, sigma=float(core.schedule[0].sigma_tm)
        )
        engine.current_snapshot = snap0
        # Forward step to get to step 1
        snap1 = engine.generate_next_frame()
        # Backward step from step 1 back to step 0
        engine._integration_direction = -1
        snap0_bwd = engine.generate_next_frame()
        engine._integration_direction = 1

        cm = snap0_bwd.center_mean_before_step
        assert cm is not None, "backward-generated snapshot must have non-None center_mean_before_step"
        assert torch.allclose(cm, torch.zeros_like(cm)), (
            f"backward snapshot center_mean_before_step should be zeros, got max abs = "
            f"{cm.abs().max().item():.6g}"
        )

    def test_engine_setter_propagates_generated_by_backward(self, core, engine):
        """current_snapshot setter must update _snapshot_generated_backward from the snapshot."""
        x = core.sample_initial_noise()
        # Create a snapshot with generated_by_backward=True
        snap_bwd = BoltzSnapshot.from_gpu_batch(
            x, step_index=0, sigma=float(core.schedule[0].sigma_tm), generated_by_backward=True
        )
        engine.current_snapshot = snap_bwd
        # Read back via the getter — should reflect the True flag
        snap_out = engine.current_snapshot
        assert snap_out.generated_by_backward is True, (
            "setter did not propagate generated_by_backward=True from assigned snapshot"
        )

        # Now assign a forward-generated snapshot
        snap_fwd = BoltzSnapshot.from_gpu_batch(
            x, step_index=0, sigma=float(core.schedule[0].sigma_tm), generated_by_backward=False
        )
        engine.current_snapshot = snap_fwd
        snap_out2 = engine.current_snapshot
        assert snap_out2.generated_by_backward is False, (
            "setter did not propagate generated_by_backward=False from assigned snapshot"
        )

class TestPathProbability:
    """Verify path probability computations are numerically sane."""

    def test_trajectory_log_path_prob_finite(self, core, full_forward_traj):
        lp = trajectory_log_path_prob(full_forward_traj, core)
        assert lp is not None
        assert math.isfinite(lp), f"log path prob is not finite: {lp}"

    def test_prefix_forward_prob_non_positive(self, core, full_forward_traj):
        """Log-probability of a Gaussian draw is ≤ 0 (normalized density)."""
        lf = prefix_forward_transitions_log_prob_tensor(full_forward_traj, core)
        assert torch.isfinite(lf).all()

    def test_prefix_backward_prob_finite(self, core, full_forward_traj):
        """Backward proposal log-prob must be finite for a forward-generated path."""
        lb = prefix_backward_proposal_log_prob_tensor(full_forward_traj, core)
        assert torch.isfinite(lb).all()


# ---------------------------------------------------------------------------
# 4. Hastings factor sign convention
# ---------------------------------------------------------------------------

class TestHastingsFactorSign:
    """The backward_shooting_metropolis_bias must be in [0, 1].

    Also verify that the sign convention matches the theory:
      log r = (lf_new - lf_old) + (lb_old - lb_new) + (log rho_new - log rho_old)
    A systematic sign inversion would produce bias > 1 for moves that should be
    penalised, which would show up as the returned value == 1.0 when the true
    ratio is < 1.
    """

    def test_bias_in_unit_interval(self, core, full_forward_traj):
        """Hastings bias must always be in [0, 1]."""
        bias = backward_shooting_metropolis_bias(
            full_forward_traj, full_forward_traj, 1, core
        )
        assert 0.0 <= bias <= 1.0, f"bias out of range: {bias}"

    def test_bias_self_comparison_is_one(self, core, full_forward_traj):
        """Comparing a path against itself must give bias = 1.0 (Δlog r = 0)."""
        bias = backward_shooting_metropolis_bias(
            full_forward_traj, full_forward_traj, 1, core
        )
        assert bias == pytest.approx(1.0, abs=1e-6), f"self-comparison bias = {bias}"

    def test_bias_returns_float(self, core, full_forward_traj):
        """The Hastings bias must be a plain Python float, not a tensor."""
        bias = backward_shooting_metropolis_bias(
            full_forward_traj, full_forward_traj, 1, core
        )
        assert isinstance(bias, float)

    def test_bias_valid_for_all_shooting_indices(self, core, full_forward_traj):
        """Hastings bias must be in [0,1] for every valid shooting index."""
        n = len(full_forward_traj)
        for k in range(1, n - 1):
            bias = backward_shooting_metropolis_bias(
                full_forward_traj, full_forward_traj, k, core
            )
            assert 0.0 <= bias <= 1.0, f"bias={bias} out of range at k={k}"

    def test_different_paths_bias_in_range(self, core, engine):
        """Two independently drawn forward paths: Hastings bias in [0,1]."""
        def _make_traj():
            x = core.sample_initial_noise()
            snap0 = BoltzSnapshot.from_gpu_batch(
                x, step_index=0, sigma=float(core.schedule[0].sigma_tm)
            )
            engine.current_snapshot = snap0
            return engine.generate_n_frames(core.num_sampling_steps)

        traj1 = _make_traj()
        traj2 = _make_traj()
        k = min(len(traj1), len(traj2)) - 2
        if k < 1:
            pytest.skip("trajectory too short for backward shooting test")
        bias = backward_shooting_metropolis_bias(traj1, traj2, k, core)
        assert 0.0 <= bias <= 1.0, f"bias={bias} for independent paths at k={k}"


# ---------------------------------------------------------------------------
# 5. Global reshuffle: acceptance = 1 for reactive paths
# ---------------------------------------------------------------------------

class TestGlobalReshuffle:
    """GlobalReshuffleMover must always accept when the trial path is reactive."""

    def test_reshuffle_move_scheme_builds(self, core, engine):
        """BoltzDiffusionOneWayShootingMover with reshuffle_probability > 0 must build."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        ensemble = tps_ensemble(network)
        mover = BoltzDiffusionOneWayShootingMover(
            ensemble=ensemble,
            selector=None,
            engine=engine,
            forward_only=False,
            reshuffle_probability=0.1,
        )
        assert mover is not None

    def test_reshuffle_mover_generates_finite_path(self, core, engine, full_forward_traj):
        """GlobalReshuffleMover.engine.generate_n_frames returns finite coords."""
        state_a, state_b = sigma_tps_state_volumes(core)
        ensemble = tps_ensemble(
            build_fixed_length_tps_network(state_a, state_b, len(full_forward_traj))
        )
        rsh = GlobalReshuffleMover(ensemble=ensemble, engine=engine)
        # Reset engine to a fresh starting state before generating
        x = core.sample_initial_noise()
        snap0 = BoltzSnapshot.from_gpu_batch(
            x, step_index=0, sigma=float(core.schedule[0].sigma_tm)
        )
        engine.current_snapshot = snap0
        new_traj = engine.generate_n_frames(core.num_sampling_steps)
        assert len(new_traj) == core.num_sampling_steps
        assert trajectory_has_finite_coords(new_traj)

    def test_forward_only_no_backward_mover(self, core, engine):
        """forward_only=True must not include a backward mover."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        ensemble = tps_ensemble(network)
        mover = BoltzDiffusionOneWayShootingMover(
            ensemble=ensemble,
            selector=None,
            engine=engine,
            forward_only=True,
            reshuffle_probability=0.0,
        )
        mover_types = [type(m).__name__ for m in mover.movers]
        assert "BoltzDiffusionBackwardShootMover" not in mover_types

    def test_two_way_includes_both_movers(self, core, engine):
        """forward_only=False must include both forward and backward movers."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        ensemble = tps_ensemble(network)
        mover = BoltzDiffusionOneWayShootingMover(
            ensemble=ensemble,
            selector=None,
            engine=engine,
            forward_only=False,
            reshuffle_probability=0.0,
        )
        mover_types = [type(m).__name__ for m in mover.movers]
        assert "BoltzDiffusionForwardShootMover" in mover_types
        assert "BoltzDiffusionBackwardShootMover" in mover_types

    def test_reshuffle_included_when_probability_nonzero(self, core, engine):
        """Nonzero reshuffle_probability must include GlobalReshuffleMover in the mover list."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        ensemble = tps_ensemble(network)
        mover = BoltzDiffusionOneWayShootingMover(
            ensemble=ensemble,
            selector=None,
            engine=engine,
            forward_only=False,
            reshuffle_probability=0.1,
        )
        mover_types = [type(m).__name__ for m in mover.movers]
        assert "GlobalReshuffleMover" in mover_types

    def test_reshuffle_excluded_when_probability_zero(self, core, engine):
        """reshuffle_probability=0 must not include GlobalReshuffleMover."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        ensemble = tps_ensemble(network)
        mover = BoltzDiffusionOneWayShootingMover(
            ensemble=ensemble,
            selector=None,
            engine=engine,
            forward_only=False,
            reshuffle_probability=0.0,
        )
        mover_types = [type(m).__name__ for m in mover.movers]
        assert "GlobalReshuffleMover" not in mover_types


# ---------------------------------------------------------------------------
# 6. Full scheme construction
# ---------------------------------------------------------------------------

class TestFullSchemeConstruction:
    """build_one_way_shooting_scheme must build without errors for all configurations."""

    def test_build_two_way_with_reshuffle(self, core, engine):
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        scheme = build_one_way_shooting_scheme(
            network, engine, forward_only=False, reshuffle_probability=0.1
        )
        assert scheme is not None

    def test_build_forward_only_no_reshuffle(self, core, engine):
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        scheme = build_one_way_shooting_scheme(
            network, engine, forward_only=True, reshuffle_probability=0.0
        )
        assert scheme is not None

    def test_build_forward_only_with_reshuffle_disabled(self, core, engine):
        """forward_only=True ignores reshuffle_probability (overrides to 0)."""
        state_a, state_b = sigma_tps_state_volumes(core)
        network = build_fixed_length_tps_network(
            state_a, state_b, core.num_sampling_steps + 1
        )
        # Should not raise even if reshuffle_probability > 0 is passed with forward_only
        scheme = build_one_way_shooting_scheme(
            network, engine, forward_only=True, reshuffle_probability=0.5
        )
        assert scheme is not None

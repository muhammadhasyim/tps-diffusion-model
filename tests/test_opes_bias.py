"""Tests for the OPES adaptive bias enhanced sampling technique.

Covers:
  - Kernel evaluation (truncated Gaussian)
  - Kernel deposition and compression (merging)
  - Bias potential computation
  - Acceptance factor (detailed balance)
  - Adaptive sigma (Welford)
  - State save/load roundtrip
  - Convergence: bias flattens a peaked distribution
  - Reweighting recovers unbiased distribution
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from genai_tps.enhanced_sampling.opes_bias import OPESBias, Kernel


class TestKernelEvaluation:
    """Tests for individual kernel evaluation."""

    def test_kernel_at_center(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        k = Kernel(center=3.0, sigma=1.0, height=1.0)
        val = bias._evaluate_single_kernel(k, 3.0)
        expected = 1.0 * (math.exp(0.0) - bias.cutoff_value)
        assert val == pytest.approx(expected, rel=1e-10)

    def test_kernel_decays_with_distance(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        k = Kernel(center=0.0, sigma=1.0, height=1.0)
        val_0 = bias._evaluate_single_kernel(k, 0.0)
        val_1 = bias._evaluate_single_kernel(k, 1.0)
        val_2 = bias._evaluate_single_kernel(k, 2.0)
        assert val_0 > val_1 > val_2 > 0

    def test_kernel_zero_beyond_cutoff(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        k = Kernel(center=0.0, sigma=1.0, height=1.0)
        far = bias.kernel_cutoff * 1.0 + 0.1
        assert bias._evaluate_single_kernel(k, far) == 0.0
        assert bias._evaluate_single_kernel(k, -far) == 0.0

    def test_kernel_exactly_at_cutoff(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        k = Kernel(center=0.0, sigma=1.0, height=1.0)
        at_cutoff = bias.kernel_cutoff
        assert bias._evaluate_single_kernel(k, at_cutoff) == 0.0

    def test_height_scales_linearly(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        k1 = Kernel(center=0.0, sigma=1.0, height=1.0)
        k2 = Kernel(center=0.0, sigma=1.0, height=3.0)
        val1 = bias._evaluate_single_kernel(k1, 0.5)
        val2 = bias._evaluate_single_kernel(k2, 0.5)
        assert val2 == pytest.approx(3.0 * val1, rel=1e-10)


class TestKernelDepositionAndCompression:
    """Tests for kernel addition and merging."""

    def test_first_kernel_deposited(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1)
        bias.update(cv_accepted=3.0, mc_step=1)
        assert bias.n_kernels == 1
        assert bias.kernels[0].center == pytest.approx(3.0, abs=1e-6)

    def test_distant_kernels_not_merged(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        compression_threshold=1.0, fixed_sigma=1.0)
        bias.update(cv_accepted=0.0, mc_step=1)
        bias.update(cv_accepted=5.0, mc_step=2)
        assert bias.n_kernels == 2

    def test_close_kernels_merged(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        compression_threshold=1.0, fixed_sigma=1.0)
        bias.update(cv_accepted=0.0, mc_step=1)
        bias.update(cv_accepted=0.1, mc_step=2)
        assert bias.n_kernels == 1
        assert bias.kernels[0].center != 0.0

    def test_many_same_point_compresses(self):
        """Depositing many kernels at the same CV should compress to one."""
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        compression_threshold=1.0, fixed_sigma=1.0, explore=True)
        for i in range(100):
            bias.update(cv_accepted=5.0, mc_step=i + 1)
        assert bias.n_kernels == 1
        assert bias.counter == 100
        assert bias.kernels[0].height > 1.0

    def test_pace_controls_deposition_frequency(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=5,
                        fixed_sigma=1.0, explore=True)
        for i in range(10):
            bias.update(cv_accepted=float(i) * 10.0, mc_step=i + 1)
        assert bias.n_kernels == 2
        assert bias.counter == 2


class TestBiasPotential:
    """Tests for the bias potential V(s)."""

    def test_no_bias_before_kernels(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        assert bias.evaluate(3.0) == 0.0

    def test_bias_nonzero_after_kernels(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        bias.update(cv_accepted=3.0, mc_step=1)
        bias.update(cv_accepted=3.0, mc_step=2)
        assert bias.evaluate(3.0) != 0.0

    def test_bias_increases_at_visited_locations(self):
        """The bias should be higher (more repulsive) where we've deposited more kernels."""
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(20):
            bias.update(cv_accepted=5.0, mc_step=i + 1)
        for i in range(5):
            bias.update(cv_accepted=10.0, mc_step=i + 21)

        v_visited = bias.evaluate(5.0)
        v_less_visited = bias.evaluate(10.0)
        assert v_visited > v_less_visited

    def test_epsilon_prevents_log_of_zero(self):
        """At unvisited locations, epsilon prevents -inf."""
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        bias.update(cv_accepted=0.0, mc_step=1)
        v_far = bias.evaluate(100.0)
        assert math.isfinite(v_far)


class TestAcceptanceFactor:
    """Tests for the Metropolis acceptance factor."""

    def test_no_bias_returns_one(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        assert bias.compute_acceptance_factor(1.0, 5.0) == 1.0

    def test_factor_is_finite(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(50):
            bias.update(cv_accepted=float(i) * 0.5, mc_step=i + 1)
        factor = bias.compute_acceptance_factor(3.0, 12.0)
        assert math.isfinite(factor)
        assert factor >= 0

    def test_factor_detailed_balance(self):
        """factor(a,b) * factor(b,a) should approximately equal 1 for the same bias state."""
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(30):
            bias.update(cv_accepted=float(i), mc_step=i + 1)
        cv_a, cv_b = 5.0, 15.0
        fwd = bias.compute_acceptance_factor(cv_a, cv_b)
        bwd = bias.compute_acceptance_factor(cv_b, cv_a)
        assert fwd * bwd == pytest.approx(1.0, rel=1e-8)

    def test_favors_unvisited_regions(self):
        """Moving to a less-visited region should have factor > 1."""
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(50):
            bias.update(cv_accepted=5.0, mc_step=i + 1)
        factor = bias.compute_acceptance_factor(5.0, 20.0)
        assert factor > 1.0

    def test_protocol_compliance(self):
        """OPESBias satisfies the EnhancedSamplingBias protocol."""
        from genai_tps.enhanced_sampling import EnhancedSamplingBias

        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        assert isinstance(bias, EnhancedSamplingBias)


class TestAdaptiveSigma:
    """Tests for the Welford running variance and adaptive kernel width."""

    def test_sigma_starts_at_sigma_0(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        sigma = bias._current_sigma()
        assert sigma == pytest.approx(bias.sigma_0, rel=1e-6)

    def test_sigma_adapts_with_data(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0)
        rng = np.random.default_rng(123)
        for cv in rng.normal(5.0, 0.5, size=100):
            bias._welford_update(float(cv))
        sigma = bias._current_sigma()
        assert sigma != bias.sigma_0
        assert sigma > 0

    def test_sigma_min_respected(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, sigma_min=0.5)
        for cv in [1.0, 1.0, 1.0, 1.0, 1.0]:
            bias._welford_update(cv)
        sigma = bias._current_sigma()
        assert sigma >= 0.5

    def test_fixed_sigma_overrides(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, fixed_sigma=2.0)
        for cv in [1.0, 2.0, 3.0, 4.0, 5.0]:
            bias._welford_update(cv)
        assert bias._current_sigma() == 2.0


class TestStateSaveLoad:
    """Tests for state serialization and deserialization."""

    def test_roundtrip(self, tmp_path):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        rng = np.random.default_rng(42)
        for i in range(50):
            bias.update(cv_accepted=float(rng.normal(5.0, 2.0)), mc_step=i + 1)

        save_path = tmp_path / "opes_state.json"
        bias.save_state(save_path)

        loaded = OPESBias.load_state(save_path)
        assert loaded.n_kernels == bias.n_kernels
        assert loaded.counter == bias.counter
        assert loaded.zed == pytest.approx(bias.zed, rel=1e-10)
        assert loaded.sum_weights == pytest.approx(bias.sum_weights, rel=1e-10)

        cv_test = 5.0
        assert loaded.evaluate(cv_test) == pytest.approx(bias.evaluate(cv_test), rel=1e-10)

    def test_load_preserves_config(self, tmp_path):
        bias = OPESBias(kbt=2.0, barrier=8.0, biasfactor=20.0, epsilon=0.01,
                        kernel_cutoff=4.0, compression_threshold=0.5, pace=3,
                        sigma_min=0.1, explore=True)
        save_path = tmp_path / "config_test.json"
        bias.save_state(save_path)

        loaded = OPESBias.load_state(save_path)
        assert loaded.kbt == 2.0
        assert loaded.barrier == 8.0
        assert loaded.biasfactor == 20.0
        assert loaded.epsilon == 0.01
        assert loaded.kernel_cutoff == 4.0
        assert loaded.compression_threshold == 0.5
        assert loaded.pace == 3
        assert loaded.sigma_min == 0.1
        assert loaded.explore is True


class TestBiasOnGrid:
    """Tests for the grid evaluation helper."""

    def test_grid_output_shape(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(20):
            bias.update(cv_accepted=float(i), mc_step=i + 1)
        cv_grid, bias_grid = bias.bias_on_grid(0.0, 20.0, n_points=100)
        assert cv_grid.shape == (100,)
        assert bias_grid.shape == (100,)
        assert np.all(np.isfinite(bias_grid))


class TestKdeProbability:
    """Normalized kernel mixture density used for OPES plots."""

    def test_positive_and_matches_raw_kde_norm(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(15):
            bias.update(cv_accepted=float(i), mc_step=i + 1)
        assert bias.kde_norm > 0
        mid = 7.0
        p = bias.kde_probability(mid)
        assert p > 0
        assert p == pytest.approx(bias._raw_kde(mid) / bias.kde_norm)


class TestReweighting:
    """Tests for reweight_samples."""

    def test_weights_sum_to_one(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(30):
            bias.update(cv_accepted=float(i), mc_step=i + 1)
        cv_values = np.linspace(0, 30, 50)
        weights = bias.reweight_samples(cv_values)
        assert weights.sum() == pytest.approx(1.0, rel=1e-10)
        assert np.all(weights >= 0)

    def test_weights_are_finite(self):
        bias = OPESBias(kbt=1.0, barrier=5.0, biasfactor=10.0, pace=1,
                        fixed_sigma=1.0, explore=True)
        for i in range(20):
            bias.update(cv_accepted=5.0, mc_step=i + 1)
        cv_values = np.array([5.0, 5.0, 5.0, 20.0, 20.0])
        weights = bias.reweight_samples(cv_values)
        assert np.all(np.isfinite(weights))


class TestConvergence:
    """Integration test: OPES bias should flatten a peaked distribution."""

    def test_bias_flattens_peaked_input(self):
        """Simulate OPES on a peaked Gaussian p(s) ~ N(5, 0.5).

        After many updates at samples from the biased distribution, the bias
        should develop positive values near the peak (repulsive) and negative
        values in the tails (attractive), pushing the effective distribution
        toward flatter.
        """
        bias = OPESBias(
            kbt=1.0,
            barrier=3.0,
            biasfactor=float("inf"),
            pace=1,
            fixed_sigma=0.5,
            explore=True,
        )

        rng = np.random.default_rng(7)
        for step in range(1, 501):
            cv = float(rng.normal(5.0, 0.5))
            bias.update(cv, step)

        v_peak = bias.evaluate(5.0)
        v_tail = bias.evaluate(8.0)
        assert v_peak > v_tail, (
            f"Bias at peak ({v_peak:.3f}) should exceed bias at tail ({v_tail:.3f})"
        )

    def test_explore_vs_convergence_mode(self):
        """Explore mode should not reweight kernels; convergence should."""
        rng = np.random.default_rng(99)
        cvs = [float(rng.normal(5.0, 1.0)) for _ in range(100)]

        explore = OPESBias(kbt=1.0, barrier=3.0, biasfactor=10.0, pace=1,
                           fixed_sigma=0.5, explore=True)
        converge = OPESBias(kbt=1.0, barrier=3.0, biasfactor=10.0, pace=1,
                            fixed_sigma=0.5, explore=False)

        for i, cv in enumerate(cvs):
            explore.update(cv, i + 1)
            converge.update(cv, i + 1)

        assert explore.n_kernels > 0
        assert converge.n_kernels > 0
        v_e = explore.evaluate(5.0)
        v_c = converge.evaluate(5.0)
        assert v_e != v_c

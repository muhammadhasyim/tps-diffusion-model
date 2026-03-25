"""Tests for the exponential tilting enhanced sampling technique.

Covers:
  - Bias factor computation (correctness, edge cases)
  - Detailed balance (factor symmetry)
  - Lambda annealing
  - Sample recording
  - MBAR integration (with mock data)
"""

from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genai_tps.enhanced_sampling.exponential_tilting import ExponentialTiltingBias
from genai_tps.enhanced_sampling.mbar_analysis import (
    MBARDistributionEstimator,
    MBARResult,
    TiltedSamples,
)


def _has_pymbar() -> bool:
    try:
        import pymbar  # noqa: F401

        return True
    except ImportError:
        return False


class TestExponentialTiltingBias:
    """Unit tests for ExponentialTiltingBias."""

    def test_zero_lambda_returns_one(self):
        bias = ExponentialTiltingBias(lambda_value=0.0)
        assert bias.compute_acceptance_factor(1.0, 5.0) == 1.0
        assert bias.compute_acceptance_factor(5.0, 1.0) == 1.0

    def test_positive_lambda_favors_lower_cv(self):
        bias = ExponentialTiltingBias(lambda_value=1.0)
        factor_decrease = bias.compute_acceptance_factor(5.0, 3.0)
        factor_increase = bias.compute_acceptance_factor(3.0, 5.0)
        assert factor_decrease > 1.0
        assert factor_increase < 1.0

    def test_negative_lambda_favors_higher_cv(self):
        bias = ExponentialTiltingBias(lambda_value=-1.0)
        factor_increase = bias.compute_acceptance_factor(3.0, 5.0)
        factor_decrease = bias.compute_acceptance_factor(5.0, 3.0)
        assert factor_increase > 1.0
        assert factor_decrease < 1.0

    def test_known_value(self):
        bias = ExponentialTiltingBias(lambda_value=2.0)
        factor = bias.compute_acceptance_factor(1.0, 3.0)
        expected = math.exp(-2.0 * (3.0 - 1.0))
        assert factor == pytest.approx(expected, rel=1e-10)

    def test_detailed_balance_symmetry(self):
        """factor(a, b) * factor(b, a) == 1  (detailed balance)."""
        bias = ExponentialTiltingBias(lambda_value=3.7)
        cv_a, cv_b = 2.5, 7.3
        fwd = bias.compute_acceptance_factor(cv_a, cv_b)
        bwd = bias.compute_acceptance_factor(cv_b, cv_a)
        assert fwd * bwd == pytest.approx(1.0, rel=1e-10)

    def test_overflow_protection(self):
        bias = ExponentialTiltingBias(lambda_value=1e6)
        factor = bias.compute_acceptance_factor(0.0, 1.0)
        assert math.isfinite(factor)
        assert factor >= 0.0

    def test_set_lambda(self):
        bias = ExponentialTiltingBias(lambda_value=1.0)
        assert bias.lambda_value == 1.0
        bias.set_lambda(5.0)
        assert bias.lambda_value == 5.0
        factor = bias.compute_acceptance_factor(0.0, 1.0)
        assert factor == pytest.approx(math.exp(-5.0), rel=1e-10)

    def test_update_records_sample(self):
        bias = ExponentialTiltingBias(lambda_value=1.0)
        bias.update(3.14, mc_step=1)
        bias.update(2.71, mc_step=2)
        assert len(bias.samples) == 2
        assert bias.samples[0]["cv"] == 3.14
        assert bias.samples[0]["lambda"] == 1.0
        assert bias.samples[1]["mc_step"] == 2

    def test_clear_samples(self):
        bias = ExponentialTiltingBias(lambda_value=1.0)
        for i in range(10):
            bias.update(float(i), mc_step=i + 1)
        assert len(bias.samples) == 10
        bias.clear_samples()
        assert len(bias.samples) == 0

    def test_reduced_potential(self):
        bias = ExponentialTiltingBias(lambda_value=2.5)
        assert bias.reduced_potential(4.0) == pytest.approx(10.0)

    def test_protocol_compliance(self):
        """ExponentialTiltingBias satisfies the EnhancedSamplingBias protocol."""
        from genai_tps.enhanced_sampling import EnhancedSamplingBias

        bias = ExponentialTiltingBias(lambda_value=1.0)
        assert isinstance(bias, EnhancedSamplingBias)


class TestMBARDistributionEstimator:
    """Tests for the MBAR analysis module."""

    @pytest.fixture()
    def gaussian_samples(self) -> list[TiltedSamples]:
        """Synthetic samples from a Gaussian tilted by different lambdas.

        The true distribution is N(mu=5, sigma=1).  Exponential tilting with
        lambda shifts the mean to mu - lambda * sigma^2.
        """
        rng = np.random.default_rng(42)
        mu, sigma = 5.0, 1.0
        n_per_state = 500
        lambdas = [0.0, 0.5, -0.5, 1.0, -1.0]
        samples = []
        for lam in lambdas:
            shifted_mu = mu - lam * sigma ** 2
            cvs = rng.normal(shifted_mu, sigma, size=n_per_state).tolist()
            samples.append(TiltedSamples(lambda_value=lam, cv_values=cvs))
        return samples

    def test_add_samples(self, gaussian_samples):
        est = MBARDistributionEstimator()
        for s in gaussian_samples:
            est.add_samples(s)
        assert est.n_states == 5
        assert est.n_total_samples == 2500

    def test_empty_samples_skipped(self):
        est = MBARDistributionEstimator()
        est.add_samples(TiltedSamples(lambda_value=1.0, cv_values=[]))
        assert est.n_states == 0

    def test_build_u_kn_shape(self, gaussian_samples):
        est = MBARDistributionEstimator()
        for s in gaussian_samples:
            est.add_samples(s)
        u_kn, N_k, all_cvs = est._build_u_kn()
        assert u_kn.shape == (5, 2500)
        assert N_k.shape == (5,)
        assert all_cvs.shape == (2500,)

    @pytest.mark.skipif(
        not _has_pymbar(),
        reason="pymbar not installed",
    )
    def test_too_few_states_raises(self):
        est = MBARDistributionEstimator()
        est.add_samples(TiltedSamples(lambda_value=0.0, cv_values=[1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="at least 2 states"):
            est.estimate()

    @pytest.mark.skipif(
        not _has_pymbar(),
        reason="pymbar not installed",
    )
    def test_estimate_gaussian(self, gaussian_samples):
        """MBAR should recover a distribution centered near mu=5."""
        est = MBARDistributionEstimator()
        for s in gaussian_samples:
            est.add_samples(s)
        result = est.estimate(n_bins=30, cv_range=(2.0, 8.0))
        assert isinstance(result, MBARResult)
        assert len(result.bin_centers) == 30
        assert result.bin_probabilities.sum() > 0

        peak_idx = np.argmax(result.bin_probabilities)
        peak_cv = result.bin_centers[peak_idx]
        assert abs(peak_cv - 5.0) < 1.0

    def test_save_load_roundtrip(self, gaussian_samples, tmp_path):
        est = MBARDistributionEstimator()
        for s in gaussian_samples:
            est.add_samples(s)
        save_path = tmp_path / "samples.json"
        est.save_samples(save_path)

        est2 = MBARDistributionEstimator.load_samples(save_path)
        assert est2.n_states == est.n_states
        assert est2.n_total_samples == est.n_total_samples

    def test_add_samples_from_bias(self):
        bias = ExponentialTiltingBias(lambda_value=1.0)
        for i in range(20):
            bias.update(float(i) * 0.5, mc_step=i + 1)
        bias.set_lambda(2.0)
        for i in range(20):
            bias.update(float(i) * 0.3, mc_step=i + 21)

        est = MBARDistributionEstimator()
        est.add_samples_from_bias(bias, burn_in_fraction=0.1)
        assert est.n_states == 2

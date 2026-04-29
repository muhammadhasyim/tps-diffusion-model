"""Tests for harmonic umbrella enhanced sampling (:mod:`HarmonicUmbrellaBias`)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pymbar  # noqa: F401
import pytest

from genai_tps.simulation import EnhancedSamplingBias, HarmonicUmbrellaBias
from genai_tps.simulation.mbar_analysis import MBARDistributionEstimator, MBARResult, WindowSamples


class TestHarmonicUmbrellaBias:
    """Unit tests for harmonic umbrella restraint on scalar CV."""

    def test_zero_kappa_returns_one_for_any_move(self) -> None:
        b = HarmonicUmbrellaBias(center=3.0, kappa=0.0, kbt=2.494)
        assert b.compute_acceptance_factor(1.0, 9.0) == pytest.approx(1.0)

    def test_detailed_balance_symmetry(self) -> None:
        b = HarmonicUmbrellaBias(center=2.0, kappa=2.5, kbt=2.494)
        a, z = 0.3, 6.8
        fwd = b.compute_acceptance_factor(a, z)
        bwd = b.compute_acceptance_factor(z, a)
        assert fwd * bwd == pytest.approx(1.0, rel=1e-9)

    def test_known_value(self) -> None:
        b = HarmonicUmbrellaBias(center=0.0, kappa=4.0, kbt=1.0)
        v_old = b.bias_potential(1.0)
        v_new = b.bias_potential(3.0)
        expected = math.exp(-((v_new - v_old)))
        assert b.compute_acceptance_factor(1.0, 3.0) == pytest.approx(expected, rel=1e-12)

    def test_overflow_protection_large_kappa(self) -> None:
        b = HarmonicUmbrellaBias(center=5.0, kappa=1e8, kbt=1.0)
        f = b.compute_acceptance_factor(5.1, 5.2)
        assert math.isfinite(f)

    def test_reduced_potential(self) -> None:
        b = HarmonicUmbrellaBias(center=1.0, kappa=2.0, kbt=1.5)
        assert b.reduced_potential(4.0) == pytest.approx(b.bias_potential(4.0) / 1.5)

    def test_protocol(self) -> None:
        b = HarmonicUmbrellaBias(center=2.4, kappa=1.0, kbt=2.494)
        assert isinstance(b, EnhancedSamplingBias)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "hub.json"
        b = HarmonicUmbrellaBias(center=8.8, kappa=3.1, kbt=2.494)
        b.update(4.44, mc_step=1)
        b.save_state(path)
        b2 = HarmonicUmbrellaBias.load_state(path)
        assert b2.center == pytest.approx(8.8)
        assert b2.kappa == pytest.approx(3.1)
        assert len(b2.samples) == 1


class TestMBARDistributionEstimatorWindows:
    """MBAR with WindowSamples."""

    @pytest.fixture()
    def overlapping_windows(self) -> list[WindowSamples]:
        rng = np.random.default_rng(42)
        kappa = 4.0
        kbt = 1.0
        centers = [4.5, 5.0, 5.5]
        out = []
        for c in centers:
            cvs = rng.normal(loc=c, scale=0.5, size=400).tolist()
            out.append(WindowSamples(center=c, kappa=kappa, kbt=kbt, cv_values=cvs))
        return out

    def test_add_samples_and_u_kn(self, overlapping_windows: list[WindowSamples]) -> None:
        est = MBARDistributionEstimator()
        for s in overlapping_windows:
            est.add_samples(s)
        u_kn, nk, flat = est._build_u_kn()
        assert u_kn.shape == (3, 1200)
        assert nk.shape == (3,)
        assert flat.shape == (1200,)

    def test_estimate_runs(self, overlapping_windows: list[WindowSamples]) -> None:
        est = MBARDistributionEstimator()
        for s in overlapping_windows:
            est.add_samples(s)
        result = est.estimate(n_bins=25, cv_range=(2.0, 9.0))
        assert isinstance(result, MBARResult)
        assert result.bin_probabilities.sum() > 0
        peak = result.bin_centers[np.argmax(result.bin_probabilities)]
        assert 4.0 < float(peak) < 7.0

    def test_add_samples_from_bias(self) -> None:
        b = HarmonicUmbrellaBias(center=-1.0, kappa=2.4, kbt=3.14)
        for mc in range(30):
            b.update(float(mc) * 0.2, mc_step=mc + 1)
        b.set_center(2.0)
        for mc in range(30):
            b.update(float(mc) * 0.1 + 1.0, mc_step=mc + 31)
        est = MBARDistributionEstimator()
        est.add_samples_from_bias(b, burn_in_fraction=0.0)
        assert est.n_states == 2

    def test_save_load_json(
        self, overlapping_windows: list[WindowSamples], tmp_path: Path,
    ) -> None:
        est = MBARDistributionEstimator()
        for s in overlapping_windows:
            est.add_samples(s)
        path = tmp_path / "w.json"
        est.save_samples(path)
        est2 = MBARDistributionEstimator.load_samples(path)
        assert est2.n_states == est.n_states

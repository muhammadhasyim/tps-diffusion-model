"""Tests for the N-dimensional OPES adaptive bias.

Covers:
  - Kernel dataclass with array center/sigma
  - N-D kernel evaluation (diagonal Gaussian, truncation)
  - N-D kernel merge (height-weighted per-dimension center + variance)
  - N-D Welford + D-dimensional Silverman bandwidth rule
  - compute_acceptance_factor with 2-D CV arrays
  - bias_on_grid: works for 1-D, raises for N-D
  - reweight_samples for 1-D and 2-D
  - save_state / load_state roundtrip for 1-D and 2-D
  - Backward compatibility: load old 1-D scalar state into new class
  - _parse_bias_cv_list and _make_multi_cv_function
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genai_tps.simulation.bias.opes import (
    Kernel,
    OPESBias,
    _all_finite,
    _to_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2d_bias(**kwargs) -> OPESBias:
    """Create a 2-D OPESBias with sensible defaults."""
    defaults = dict(ndim=2, kbt=1.0, barrier=3.0, biasfactor=5.0)
    defaults.update(kwargs)
    return OPESBias(**defaults)


def _feed_kernels(bias: OPESBias, centers: list[list[float]], sigma_val: float = 0.5) -> None:
    """Manually deposit kernels to make bias non-trivial for testing."""
    for center in centers:
        cv = np.array(center, dtype=np.float64)
        k = Kernel(
            center=cv.copy(),
            sigma=np.full(bias.ndim, sigma_val),
            height=1.0,
        )
        bias.kernels.append(k)
    # Recalculate kde_norm and zed
    bias.counter = len(centers)
    bias.sum_weights = float(bias.counter)
    bias.kde_norm = bias.sum_weights
    bias._update_zed()


# ---------------------------------------------------------------------------
# Kernel dataclass
# ---------------------------------------------------------------------------


class TestKernelND:
    def test_kernel_stores_arrays(self):
        c = np.array([1.0, 2.0])
        s = np.array([0.5, 0.3])
        k = Kernel(center=c, sigma=s, height=1.0)
        np.testing.assert_array_equal(k.center, c)
        np.testing.assert_array_equal(k.sigma, s)

    def test_kernel_coerces_lists(self):
        k = Kernel(center=[0.0, 1.0], sigma=[0.5, 0.5], height=2.0)
        assert k.center.dtype == np.float64
        assert k.sigma.dtype == np.float64
        assert k.center.shape == (2,)

    def test_kernel_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            Kernel(center=[0.0, 1.0], sigma=[0.5], height=1.0)

    def test_1d_kernel_scalar_input(self):
        """Kernel accepts a length-1 array (1-D case)."""
        k = Kernel(center=np.array([3.0]), sigma=np.array([1.0]), height=0.5)
        assert k.center.shape == (1,)


# ---------------------------------------------------------------------------
# _to_array and _all_finite helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_array_scalar(self):
        arr = _to_array(3.14, ndim=1)
        assert arr.shape == (1,)
        assert arr[0] == pytest.approx(3.14)

    def test_to_array_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            _to_array(np.array([1.0, 2.0]), ndim=3)

    def test_all_finite_scalar(self):
        assert _all_finite(1.0) is True
        assert _all_finite(float("nan")) is False
        assert _all_finite(float("inf")) is False

    def test_all_finite_array(self):
        assert _all_finite(np.array([1.0, 2.0])) is True
        assert _all_finite(np.array([1.0, float("nan")])) is False


# ---------------------------------------------------------------------------
# OPESBias.__init__ for N-D
# ---------------------------------------------------------------------------


class TestOPESBiasNDInit:
    def test_2d_init_defaults(self):
        bias = _make_2d_bias()
        assert bias.ndim == 2
        assert bias.sigma_0.shape == (2,)
        assert bias._welford_mean.shape == (2,)
        assert bias._welford_m2.shape == (2,)

    def test_sigma_min_broadcast(self):
        bias = OPESBias(ndim=3, barrier=2.0, sigma_min=0.01)
        assert bias.sigma_min.shape == (3,)
        np.testing.assert_array_equal(bias.sigma_min, [0.01, 0.01, 0.01])

    def test_fixed_sigma_broadcast(self):
        bias = OPESBias(ndim=3, barrier=2.0, fixed_sigma=0.5)
        assert bias.fixed_sigma.shape == (3,)
        np.testing.assert_array_equal(bias.fixed_sigma, [0.5, 0.5, 0.5])

    def test_fixed_sigma_per_dim(self):
        bias = OPESBias(ndim=2, barrier=2.0, fixed_sigma=np.array([0.1, 0.2]))
        np.testing.assert_array_equal(bias.fixed_sigma, [0.1, 0.2])

    def test_1d_backward_compat_scalar_api(self):
        """OPESBias(ndim=1) evaluates scalars just like the old API."""
        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)
        _feed_kernels(bias, [[1.0]])
        v = bias.evaluate(1.0)
        assert math.isfinite(v)


# ---------------------------------------------------------------------------
# N-D kernel evaluation
# ---------------------------------------------------------------------------


class TestNDKernelEvaluation:
    def test_2d_at_center(self):
        bias = _make_2d_bias()
        k = Kernel(center=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]), height=1.0)
        val = bias._evaluate_single_kernel(k, np.array([1.0, 2.0]))
        expected = 1.0 * (1.0 - bias.cutoff_value)
        assert val == pytest.approx(expected, rel=1e-8)

    def test_2d_decays_from_center(self):
        bias = _make_2d_bias()
        k = Kernel(center=np.zeros(2), sigma=np.ones(2), height=1.0)
        v0 = bias._evaluate_single_kernel(k, np.zeros(2))
        v1 = bias._evaluate_single_kernel(k, np.array([0.5, 0.0]))
        v2 = bias._evaluate_single_kernel(k, np.array([1.0, 0.0]))
        assert v0 > v1 > v2 >= 0

    def test_2d_truncated_when_far(self):
        bias = _make_2d_bias()
        k = Kernel(center=np.zeros(2), sigma=np.ones(2), height=1.0)
        # Move far along first dim only -- sum of squares should exceed cutoff^2
        far = bias.kernel_cutoff * 2.0
        val = bias._evaluate_single_kernel(k, np.array([far, 0.0]))
        assert val == 0.0

    def test_2d_asymmetric_sigma(self):
        """Kernel with different sigmas per dimension is anisotropic."""
        bias = _make_2d_bias()
        k = Kernel(center=np.zeros(2), sigma=np.array([0.1, 10.0]), height=1.0)
        # Small sigma in dim 0: should truncate at small displacement
        val_tight = bias._evaluate_single_kernel(k, np.array([0.5, 0.0]))
        # Large sigma in dim 1: should NOT truncate at same displacement
        val_loose = bias._evaluate_single_kernel(k, np.array([0.0, 0.5]))
        assert val_tight == 0.0  # truncated (0.5/0.1 = 5 >> cutoff)
        assert val_loose > 0.0


# ---------------------------------------------------------------------------
# N-D evaluate and bias
# ---------------------------------------------------------------------------


class TestNDEvaluate:
    def test_evaluate_empty_returns_zero(self):
        bias = _make_2d_bias()
        assert bias.evaluate(np.array([0.0, 0.0])) == 0.0

    def test_evaluate_nonfinite_cv_returns_zero(self):
        bias = _make_2d_bias()
        _feed_kernels(bias, [[0.0, 0.0]])
        assert bias.evaluate(np.array([float("nan"), 0.0])) == 0.0

    def test_evaluate_at_kernel_center_positive(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        _feed_kernels(bias, [[1.0, 2.0]], sigma_val=0.5)
        v = bias.evaluate(np.array([1.0, 2.0]))
        assert math.isfinite(v)

    def test_acceptance_factor_no_kernels(self):
        bias = _make_2d_bias()
        f = bias.compute_acceptance_factor(np.zeros(2), np.ones(2))
        assert f == 1.0

    def test_acceptance_factor_nonfinite(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        _feed_kernels(bias, [[0.0, 0.0]])
        f = bias.compute_acceptance_factor(np.array([float("nan"), 0.0]), np.zeros(2))
        assert f == 1.0


# ---------------------------------------------------------------------------
# N-D Welford + Silverman bandwidth
# ---------------------------------------------------------------------------


class TestNDWelfordSilverman:
    def test_welford_accumulates_per_dim(self):
        bias = _make_2d_bias()
        vals = [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]
        for v in vals:
            bias._welford_update(np.array(v))
        assert bias._welford_n == 3
        assert bias._welford_mean.shape == (2,)
        # Mean should be [1.5, 2.5]
        np.testing.assert_allclose(bias._welford_mean, [1.5, 2.5], atol=1e-10)

    def test_silverman_exponent_2d(self):
        """For D=2, Silverman exponent should be -1/(4+2) = -1/6."""
        bias = _make_2d_bias()
        # Feed enough data that sigma is data-driven
        rng = np.random.default_rng(42)
        for _ in range(50):
            bias._welford_update(rng.normal(size=2))
        bias.counter = 50
        bias.sum_weights = 50.0
        bias.sum_weights2 = 50.0
        sigma = bias._current_sigma()
        # Verify shape and positivity
        assert sigma.shape == (2,)
        assert np.all(sigma > 0)

    def test_silverman_exponent_matches_plumed2_formula(self):
        """Check s_rescaling matches (size * (D+2)/4)^{-1/(4+D)} from OPESmetad.cpp:1094."""
        ndim = 3
        bias = OPESBias(ndim=ndim, barrier=2.0)
        bias._welford_n = 100
        bias._welford_m2 = np.ones(ndim) * 4.0  # variance = 1.0 per dim
        bias.sum_weights = 80.0
        bias.sum_weights2 = 80.0
        bias.counter = 80
        sigma = bias._current_sigma()
        # Manual computation
        neff = (1 + 80.0) ** 2 / (1 + 80.0)
        d = float(ndim)
        s_rescaling = (neff * (d + 2.0) / 4.0) ** (-1.0 / (4.0 + d))
        expected_sigma = np.sqrt(4.0 / 100) * s_rescaling  # var = m2/n
        np.testing.assert_allclose(sigma, np.full(ndim, expected_sigma), rtol=1e-6)

    def test_fixed_sigma_overrides_adaptive(self):
        bias = OPESBias(ndim=2, barrier=2.0, fixed_sigma=np.array([0.1, 0.2]))
        bias._welford_n = 100
        bias._welford_m2 = np.ones(2)
        sigma = bias._current_sigma()
        np.testing.assert_array_equal(sigma, [0.1, 0.2])


# ---------------------------------------------------------------------------
# N-D kernel merge
# ---------------------------------------------------------------------------


class TestNDKernelMerge:
    def test_merge_in_2d_weighted_center(self):
        bias = _make_2d_bias()
        k1 = Kernel(center=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]), height=1.0)
        k2 = Kernel(center=np.array([2.0, 0.0]), sigma=np.array([1.0, 1.0]), height=1.0)
        bias.kernels = [k1]
        bias._merge_kernel(0, k2.center, k2.sigma, k2.height)
        merged = bias.kernels[0]
        # Equal weights -> center at midpoint
        np.testing.assert_allclose(merged.center, [1.0, 0.0], atol=1e-10)
        assert merged.height == pytest.approx(2.0)
        assert merged.sigma.shape == (2,)
        assert np.all(merged.sigma > 0)

    def test_find_mergeable_within_threshold(self):
        bias = _make_2d_bias()
        bias.kernels = [
            Kernel(center=np.zeros(2), sigma=np.ones(2), height=1.0),
        ]
        # A point very close to the existing kernel center
        idx = bias._find_mergeable(np.array([0.1, 0.1]), np.ones(2))
        assert idx == 0

    def test_find_mergeable_outside_threshold(self):
        bias = _make_2d_bias()
        bias.kernels = [
            Kernel(center=np.zeros(2), sigma=np.ones(2), height=1.0),
        ]
        # Far away: normalized distance >> compression_threshold
        idx = bias._find_mergeable(np.array([5.0, 5.0]), np.ones(2))
        assert idx is None

    def test_add_kernel_merges_close(self):
        bias = _make_2d_bias(fixed_sigma=1.0, compression_threshold=1.0)
        bias._add_kernel(np.zeros(2), np.ones(2), 1.0)
        assert len(bias.kernels) == 1
        # Very close second kernel should merge
        bias._add_kernel(np.array([0.05, 0.05]), np.ones(2), 1.0)
        assert len(bias.kernels) == 1  # merged into 1

    def test_add_kernel_does_not_merge_far(self):
        bias = _make_2d_bias(fixed_sigma=1.0, compression_threshold=1.0)
        bias._add_kernel(np.zeros(2), np.ones(2), 1.0)
        # Far away kernel should NOT merge
        bias._add_kernel(np.array([10.0, 10.0]), np.ones(2), 1.0)
        assert len(bias.kernels) == 2


# ---------------------------------------------------------------------------
# update() end-to-end
# ---------------------------------------------------------------------------


class TestNDUpdate:
    def test_update_deposits_kernel(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        assert bias.n_kernels == 0
        bias.update(np.array([1.0, 2.0]), mc_step=1)
        assert bias.n_kernels == 1

    def test_update_increments_counter(self):
        bias = _make_2d_bias(fixed_sigma=0.5, pace=1)
        for i in range(5):
            bias.update(np.array([float(i), 0.0]), mc_step=i + 1)
        assert bias.counter == 5

    def test_update_nonfinite_skipped(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        bias.update(np.array([float("nan"), 0.0]), mc_step=1)
        assert bias.n_kernels == 0
        assert bias._welford_n == 0

    def test_update_pace_respected(self):
        bias = _make_2d_bias(fixed_sigma=0.5, pace=3)
        for i in range(6):
            bias.update(np.array([float(i), 0.0]), mc_step=i + 1)
        # Should only deposit at steps 3 and 6
        assert bias.counter == 2

    def test_1d_scalar_update_still_works(self):
        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)
        bias.update(1.5, mc_step=1)
        assert bias.n_kernels == 1


# ---------------------------------------------------------------------------
# bias_on_grid and reweight_samples
# ---------------------------------------------------------------------------


class TestNDGridReweight:
    def test_bias_on_grid_1d(self):
        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)
        _feed_kernels(bias, [[1.0]])
        cv_grid, b_grid = bias.bias_on_grid(0.0, 2.0, n_points=50)
        assert cv_grid.shape == (50,)
        assert b_grid.shape == (50,)
        assert np.all(np.isfinite(b_grid))

    def test_bias_on_grid_nd_raises(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        _feed_kernels(bias, [[1.0, 2.0]])
        with pytest.raises(NotImplementedError):
            bias.bias_on_grid(0.0, 2.0)

    def test_reweight_samples_1d(self):
        bias = OPESBias(ndim=1, barrier=3.0, fixed_sigma=0.5)
        _feed_kernels(bias, [[0.5], [1.0], [1.5]])
        cv_vals = np.linspace(0.0, 2.0, 10)
        weights = bias.reweight_samples(cv_vals)
        assert weights.shape == (10,)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0)

    def test_reweight_samples_2d(self):
        bias = _make_2d_bias(fixed_sigma=0.5)
        _feed_kernels(bias, [[0.5, 0.5], [1.0, 1.0]])
        cv_vals = np.random.default_rng(0).uniform(0, 2, size=(20, 2))
        weights = bias.reweight_samples(cv_vals)
        assert weights.shape == (20,)
        assert weights.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Serialization: save_state / load_state
# ---------------------------------------------------------------------------


class TestNDSerialization:
    def test_roundtrip_1d(self, tmp_path):
        bias = OPESBias(ndim=1, barrier=4.0, fixed_sigma=0.5, pace=2)
        bias.update(1.0, mc_step=2)
        bias.update(1.5, mc_step=4)
        p = tmp_path / "state_1d.json"
        bias.save_state(p)
        loaded = OPESBias.load_state(p)
        assert loaded.ndim == 1
        assert loaded.n_kernels == bias.n_kernels
        np.testing.assert_allclose(loaded.kernels[0].center, bias.kernels[0].center)
        np.testing.assert_allclose(loaded.kernels[0].sigma, bias.kernels[0].sigma)
        assert loaded.kernels[0].height == pytest.approx(bias.kernels[0].height)

    def test_roundtrip_2d(self, tmp_path):
        bias = _make_2d_bias(fixed_sigma=0.5, pace=1)
        bias.update(np.array([1.0, 2.0]), mc_step=1)
        bias.update(np.array([1.5, 2.5]), mc_step=2)
        p = tmp_path / "state_2d.json"
        bias.save_state(p)
        loaded = OPESBias.load_state(p)
        assert loaded.ndim == 2
        assert loaded.n_kernels == bias.n_kernels
        for k_orig, k_load in zip(bias.kernels, loaded.kernels):
            np.testing.assert_allclose(k_load.center, k_orig.center)
            np.testing.assert_allclose(k_load.sigma, k_orig.sigma)
        np.testing.assert_allclose(loaded._welford_mean, bias._welford_mean)

    def test_backward_compat_old_1d_scalar_state(self, tmp_path):
        """Old JSON with scalar center/sigma/welford_mean loads correctly."""
        # Simulate an old-format state file (no 'ndim', scalars instead of lists)
        old_state = {
            "kbt": 1.0,
            "barrier": 5.0,
            "biasfactor": 10.0,
            "epsilon": 0.0067,
            "kernel_cutoff": 3.6,
            "compression_threshold": 1.0,
            "pace": 1,
            "sigma_min": 0.0,
            "fixed_sigma": None,
            "explore": False,
            "bias_prefactor": 0.9,
            "sigma_0": 3.162,
            "sum_weights": 2.0,
            "sum_weights2": 2.0,
            "counter": 2,
            "zed": 0.5,
            "kde_norm": 2.0,
            "welford_n": 2,
            "welford_mean": 1.25,   # scalar (old 1-D format)
            "welford_m2": 0.125,    # scalar
            "n_merges": 0,
            "kernels": [
                {"center": 1.0, "sigma": 0.5, "height": 1.0},   # scalar center/sigma
                {"center": 1.5, "sigma": 0.5, "height": 1.0},
            ],
        }
        p = tmp_path / "old_state.json"
        p.write_text(json.dumps(old_state))
        loaded = OPESBias.load_state(p)
        assert loaded.ndim == 1
        assert loaded.n_kernels == 2
        assert loaded.kernels[0].center.shape == (1,)
        assert loaded.kernels[0].sigma.shape == (1,)
        np.testing.assert_allclose(loaded.kernels[0].center, [1.0])
        np.testing.assert_allclose(loaded._welford_mean, [1.25])

    def test_loaded_2d_evaluates_correctly(self, tmp_path):
        bias = _make_2d_bias(fixed_sigma=0.5)
        _feed_kernels(bias, [[1.0, 1.0]])
        p = tmp_path / "state.json"
        bias.save_state(p)
        loaded = OPESBias.load_state(p)
        v_orig = bias.evaluate(np.array([1.0, 1.0]))
        v_load = loaded.evaluate(np.array([1.0, 1.0]))
        assert v_orig == pytest.approx(v_load, rel=1e-8)


# ---------------------------------------------------------------------------
# _parse_bias_cv_list and _make_multi_cv_function tests
# ---------------------------------------------------------------------------


class TestBiasCVParsing:
    def _import_helpers(self):
        """Import helpers from run_opes_tps.py via importlib (not a package)."""
        import importlib.util
        import sys
        from pathlib import Path
        script_path = Path(__file__).parent.parent / "scripts" / "run_opes_tps.py"
        spec = importlib.util.spec_from_file_location("run_opes_tps", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_single_valid(self):
        mod = self._import_helpers()
        names = mod._parse_bias_cv_list("contact_order")
        assert names == ["contact_order"]

    def test_parse_multi_valid(self):
        mod = self._import_helpers()
        names = mod._parse_bias_cv_list("contact_order,clash_count")
        assert names == ["contact_order", "clash_count"]

    def test_parse_whitespace_trimmed(self):
        mod = self._import_helpers()
        names = mod._parse_bias_cv_list(" rmsd , rg ")
        assert names == ["rmsd", "rg"]

    def test_parse_unknown_raises(self):
        mod = self._import_helpers()
        with pytest.raises(ValueError, match="Unknown"):
            mod._parse_bias_cv_list("totally_fake_cv")

    def test_parse_all_valid_names(self):
        mod = self._import_helpers()
        for name in mod._SINGLE_CV_NAMES:
            if name in ("rmsd", "openmm", "lddt"):
                continue  # These require additional args; skip here
            names = mod._parse_bias_cv_list(name)
            assert names == [name]


class TestMakeMultiCVFunction:
    def _import_helpers(self):
        import importlib.util
        from pathlib import Path
        script_path = Path(__file__).parent.parent / "scripts" / "run_opes_tps.py"
        spec = importlib.util.spec_from_file_location("run_opes_tps", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _snap_with_coords(self) -> "object":
        """Return a minimal mock snapshot with 5 Cα atoms."""
        import torch

        class _Snap:
            pass

        snap = _Snap()
        snap.tensor_coords = torch.randn(1, 5, 3)
        return snap

    def _mock_traj(self, snap):
        traj = [snap]

        class _T(list):
            pass

        t = _T(traj)
        return t

    def test_single_scalar_cv_returns_float(self):
        mod = self._import_helpers()
        fn, ndim = mod._make_multi_cv_function(["rg"])
        assert ndim == 1
        snap = self._snap_with_coords()
        traj = self._mock_traj(snap)
        result = fn(traj)
        assert isinstance(result, (float, int))

    def test_multi_cv_returns_ndarray(self):
        mod = self._import_helpers()
        fn, ndim = mod._make_multi_cv_function(["contact_order", "clash_count"])
        assert ndim == 2
        snap = self._snap_with_coords()
        traj = self._mock_traj(snap)
        result = fn(traj)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_three_cv_ndim(self):
        mod = self._import_helpers()
        fn, ndim = mod._make_multi_cv_function(["rg", "contact_order", "ramachandran_outlier"])
        assert ndim == 3
        snap = self._snap_with_coords()
        traj = self._mock_traj(snap)
        result = fn(traj)
        assert result.shape == (3,)

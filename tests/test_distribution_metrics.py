"""Tests for genai_tps.evaluation.distribution_metrics.

Properties verified:
1. torsion_js_distance: symmetry, range, identical returns 0, disjoint ~1.
2. wasserstein_1d: symmetry, identical returns 0, known 1-D example.
3. rmsf_spearman_rmse: perfect correlation returns rho=1 rmse=0, shape mismatch error.
4. wasserstein_2_per_atom: dict keys, shape, mean-only case = Euclidean distance.
5. ensemble_rmsf: shape, zeros for single-frame ensemble, mask produces NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from genai_tps.evaluation.distribution_metrics import (
    ensemble_rmsf,
    rmsf_spearman_rmse,
    torsion_js_distance,
    wasserstein_1d,
    wasserstein_2_per_atom,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# torsion_js_distance
# ---------------------------------------------------------------------------

def test_torsion_js_identical_returns_zero():
    """Same samples in both P and Q -> JS distance should be (near) zero."""
    angles = RNG.uniform(-180, 180, size=500)
    js = torsion_js_distance(angles, angles)
    assert js < 0.05, f"Expected near-zero for identical inputs, got {js:.4f}"


def test_torsion_js_range():
    """JS distance is always in [0, 1]."""
    p = RNG.uniform(-180, 180, size=300)
    q = RNG.uniform(-180, 180, size=300)
    js = torsion_js_distance(p, q)
    assert 0.0 <= js <= 1.0 + 1e-9, f"JS distance out of [0,1]: {js}"


def test_torsion_js_symmetry():
    """JSD is symmetric: d(P, Q) == d(Q, P)."""
    p = RNG.normal(0, 30, size=400)
    q = RNG.normal(90, 30, size=400)
    js_pq = torsion_js_distance(p, q)
    js_qp = torsion_js_distance(q, p)
    assert abs(js_pq - js_qp) < 1e-9, f"JS not symmetric: {js_pq:.6f} vs {js_qp:.6f}"


def test_torsion_js_disjoint_near_one():
    """Perfectly disjoint distributions (no bin overlap) should give JS ~ 1."""
    # All P in [-180, -90], all Q in [90, 180] -- non-overlapping hemispheres
    p = RNG.uniform(-180, -90, size=1000)
    q = RNG.uniform(90, 180, size=1000)
    js = torsion_js_distance(p, q, n_bins=36)
    # With Laplace smoothing JS won't reach exactly 1, but should be > 0.8
    assert js > 0.8, f"Expected near-1 for disjoint distributions, got {js:.4f}"


def test_torsion_js_different_bins():
    """Custom n_bins parameter is respected."""
    p = RNG.uniform(-180, 180, size=300)
    q = RNG.uniform(-180, 180, size=300)
    js_18 = torsion_js_distance(p, q, n_bins=18)
    js_72 = torsion_js_distance(p, q, n_bins=72)
    # Both should be valid (not checking equality, just validity)
    assert 0.0 <= js_18 <= 1.0
    assert 0.0 <= js_72 <= 1.0


# ---------------------------------------------------------------------------
# wasserstein_1d
# ---------------------------------------------------------------------------

def test_wasserstein_1d_identical():
    a = RNG.normal(0, 1, size=200)
    w = wasserstein_1d(a, a)
    assert w == pytest.approx(0.0, abs=1e-10)


def test_wasserstein_1d_symmetry():
    a = RNG.normal(0, 1, size=200)
    b = RNG.normal(3, 1, size=200)
    assert wasserstein_1d(a, b) == pytest.approx(wasserstein_1d(b, a), rel=1e-6)


def test_wasserstein_1d_uniform_shift():
    """W1 between U[0,1] and U[1,2] should be exactly 1."""
    a = np.linspace(0, 1, 1000)
    b = np.linspace(1, 2, 1000)
    w = wasserstein_1d(a, b)
    assert w == pytest.approx(1.0, abs=0.01)


def test_wasserstein_1d_weighted():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    w = wasserstein_1d(a, b)
    assert w == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# rmsf_spearman_rmse
# ---------------------------------------------------------------------------

def test_rmsf_perfect_correlation():
    ref = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    pred = ref * 1.0
    rho, rmse = rmsf_spearman_rmse(pred, ref)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert rmse == pytest.approx(0.0, abs=1e-9)


def test_rmsf_known_values():
    ref = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])
    rho, rmse = rmsf_spearman_rmse(pred, ref)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert rmse == pytest.approx(1.0, abs=1e-9)


def test_rmsf_negative_correlation():
    ref = np.arange(10, dtype=float)
    pred = ref[::-1].copy()
    rho, _ = rmsf_spearman_rmse(pred, ref)
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_rmsf_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmsf_spearman_rmse(np.ones(5), np.ones(7))


# ---------------------------------------------------------------------------
# wasserstein_2_per_atom
# ---------------------------------------------------------------------------

def test_w2_per_atom_keys():
    T, N = 10, 5
    p = RNG.normal(0, 1, (T, N, 3))
    q = RNG.normal(0, 1, (T, N, 3))
    result = wasserstein_2_per_atom(p, q)
    assert set(result.keys()) >= {"w2_squared", "mean_sq", "bures", "rmwd"}


def test_w2_per_atom_shape():
    T, N = 20, 6
    p = RNG.normal(0, 1, (T, N, 3))
    q = RNG.normal(0, 1, (T, N, 3))
    result = wasserstein_2_per_atom(p, q)
    assert result["rmwd"].shape == (N,)
    assert result["mean_sq"].shape == (N,)


def test_w2_per_atom_single_frame_equals_euclid():
    """Single-frame ensemble: W2 should equal Euclidean distance (no covariance)."""
    N = 8
    p = RNG.normal(0, 1, (1, N, 3))
    q = RNG.normal(0, 1, (1, N, 3))
    result = wasserstein_2_per_atom(p, q)
    expected = np.sqrt(np.sum((p[0] - q[0]) ** 2, axis=-1))
    np.testing.assert_allclose(result["rmwd"], expected, rtol=1e-5)


def test_w2_per_atom_identical_zero():
    """Identical ensembles -> W2 = 0."""
    T, N = 15, 4
    p = RNG.normal(0, 1, (T, N, 3))
    result = wasserstein_2_per_atom(p, p)
    np.testing.assert_allclose(result["w2_squared"], 0.0, atol=1e-8)


def test_w2_per_atom_non_negative():
    T, N = 12, 5
    p = RNG.normal(0, 1, (T, N, 3))
    q = RNG.normal(1, 2, (T, N, 3))
    result = wasserstein_2_per_atom(p, q)
    assert np.all(result["w2_squared"] >= -1e-9)
    assert np.all(result["rmwd"] >= -1e-9)


def test_w2_per_atom_bad_shape():
    with pytest.raises(ValueError):
        wasserstein_2_per_atom(np.ones((10, 5)), np.ones((10, 5)))  # missing last dim


# ---------------------------------------------------------------------------
# ensemble_rmsf
# ---------------------------------------------------------------------------

def test_ensemble_rmsf_shape():
    T, N = 50, 12
    coords = RNG.normal(0, 1, (T, N, 3))
    rmsf = ensemble_rmsf(coords)
    assert rmsf.shape == (N,)


def test_ensemble_rmsf_single_frame_is_zero():
    """Single frame -> RMSF = 0 for all atoms."""
    N = 10
    coords = RNG.normal(0, 1, (1, N, 3))
    rmsf = ensemble_rmsf(coords)
    np.testing.assert_allclose(rmsf, 0.0, atol=1e-10)


def test_ensemble_rmsf_non_negative():
    T, N = 30, 8
    coords = RNG.normal(0, 1, (T, N, 3))
    rmsf = ensemble_rmsf(coords)
    assert np.all(rmsf >= 0.0)


def test_ensemble_rmsf_static_atom_is_zero():
    """An atom with identical coordinates across frames -> RMSF = 0."""
    T, N = 20, 5
    coords = RNG.normal(0, 1, (T, N, 3))
    coords[:, 2, :] = coords[0, 2, :]  # atom 2 is static
    rmsf = ensemble_rmsf(coords)
    assert rmsf[2] == pytest.approx(0.0, abs=1e-9)


def test_ensemble_rmsf_mask_produces_nan():
    T, N = 20, 5
    coords = RNG.normal(0, 1, (T, N, 3))
    mask = np.ones(N)
    mask[3] = 0  # atom 3 is padding
    rmsf = ensemble_rmsf(coords, atom_mask=mask)
    assert np.isnan(rmsf[3])
    assert not np.isnan(rmsf[0])


def test_ensemble_rmsf_known_value():
    """Manually compute RMSF for a 2-frame ensemble of 1 atom."""
    # Atom at (0,0,0) and (2,0,0) -> mean (1,0,0), RMSF = sqrt(((1)^2+(1)^2)/2) = 1.0
    coords = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    rmsf = ensemble_rmsf(coords)
    assert rmsf[0] == pytest.approx(1.0, abs=1e-9)

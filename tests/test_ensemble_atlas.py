"""Tests for genai_tps.evaluation.ensemble_atlas."""

from __future__ import annotations

import numpy as np
import pytest

from genai_tps.evaluation.ensemble_atlas import (
    atlas_benchmark,
    pairwise_rmsd_pearson,
    pca_wasserstein2,
    sasa_pearson,
    transient_contact_jaccard,
)

RNG = np.random.default_rng(42)


def _coords(T: int, N: int) -> np.ndarray:
    return RNG.normal(0, 1, (T, N, 3))


# ---------------------------------------------------------------------------
# pairwise_rmsd_pearson
# ---------------------------------------------------------------------------

def test_pairwise_rmsd_identical_is_one():
    """Comparing an ensemble to itself gives Pearson r = 1."""
    c = _coords(T=10, N=6)
    r = pairwise_rmsd_pearson(c, c)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_pairwise_rmsd_range():
    p = _coords(T=8, N=5)
    q = _coords(T=8, N=5)
    r = pairwise_rmsd_pearson(p, q)
    assert -1.0 <= r <= 1.0


def test_pairwise_rmsd_ca_mask():
    """ca_mask correctly subsets atoms before RMSD computation."""
    T, N = 8, 10
    p = _coords(T, N)
    q = _coords(T, N)
    ca = np.array([True, False] * (N // 2))
    r_masked = pairwise_rmsd_pearson(p, q, ca_mask=ca)
    r_full = pairwise_rmsd_pearson(p[:, ca, :], q[:, ca, :])
    assert r_masked == pytest.approx(r_full, abs=1e-9)


def test_pairwise_rmsd_single_frame_is_nan():
    """Single-frame ensemble has no pairs -> returns NaN."""
    c = _coords(T=1, N=5)
    r = pairwise_rmsd_pearson(c, c)
    assert np.isnan(r)


# ---------------------------------------------------------------------------
# pca_wasserstein2
# ---------------------------------------------------------------------------

def test_pca_w2_keys():
    p = _coords(20, 8)
    q = _coords(20, 8)
    result = pca_wasserstein2(p, q)
    assert set(result.keys()) >= {"per_pc_w2", "mean_w2", "explained_variance"}


def test_pca_w2_identical_near_zero():
    """Comparing an ensemble to itself gives near-zero W2."""
    c = _coords(30, 6)
    result = pca_wasserstein2(c, c)
    assert result["mean_w2"] == pytest.approx(0.0, abs=1e-9)


def test_pca_w2_shape():
    n_pc = 3
    p = _coords(20, 8)
    q = _coords(20, 8)
    result = pca_wasserstein2(p, q, n_components=n_pc)
    assert len(result["per_pc_w2"]) == n_pc
    assert len(result["explained_variance"]) == n_pc


def test_pca_w2_non_negative():
    p = _coords(15, 6)
    q = _coords(15, 6)
    result = pca_wasserstein2(p, q)
    assert np.all(result["per_pc_w2"] >= 0.0)


def test_pca_explained_variance_sums_to_le_one():
    p = _coords(25, 8)
    result = pca_wasserstein2(p, p, n_components=2)
    assert result["explained_variance"].sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# transient_contact_jaccard
# ---------------------------------------------------------------------------

def test_jaccard_identical_is_one():
    """Same ensemble -> identical transient contacts -> Jaccard = 1."""
    c = _coords(30, 8)
    j = transient_contact_jaccard(c, c, cutoff_ang=0.5)
    assert j == pytest.approx(1.0, abs=1e-9)


def test_jaccard_range():
    p = _coords(30, 8)
    q = _coords(30, 8)
    j = transient_contact_jaccard(p, q)
    assert 0.0 <= j <= 1.0


def test_jaccard_empty_returns_one():
    """Both ensembles with no transient contacts -> Jaccard = 1.0."""
    p = _coords(5, 4)
    q = _coords(5, 4)
    # Very large cutoff forces all contacts to be always-present, not transient
    j = transient_contact_jaccard(p, q, cutoff_ang=1e8)
    assert j == pytest.approx(1.0)


def test_jaccard_ca_mask():
    T, N = 20, 10
    p = _coords(T, N)
    q = _coords(T, N)
    ca = np.zeros(N, dtype=bool)
    ca[:5] = True
    j_masked = transient_contact_jaccard(p, q, ca_mask=ca)
    j_full = transient_contact_jaccard(p[:, ca, :], q[:, ca, :])
    assert j_masked == pytest.approx(j_full, abs=1e-9)


# ---------------------------------------------------------------------------
# sasa_pearson
# ---------------------------------------------------------------------------

def test_sasa_pearson_identical():
    sasa = RNG.uniform(0, 100, (20, 10))
    r = sasa_pearson(sasa, sasa)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_sasa_pearson_range():
    sa = RNG.uniform(0, 100, (20, 10))
    sb = RNG.uniform(0, 100, (20, 10))
    r = sasa_pearson(sa, sb)
    assert -1.0 <= r <= 1.0


def test_sasa_pearson_1d_input():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    r = sasa_pearson(a, b)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_sasa_pearson_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sasa_pearson(np.ones(5), np.ones(7))


# ---------------------------------------------------------------------------
# atlas_benchmark (composite)
# ---------------------------------------------------------------------------

def test_atlas_benchmark_keys():
    T, N = 10, 6
    p = _coords(T, N)
    q = _coords(T, N)
    result = atlas_benchmark(p, q)
    expected = {
        "pairwise_rmsd_pearson",
        "pca_w2",
        "transient_contact_jaccard",
        "rmsf_pearson",
        "sasa_pearson",
    }
    assert expected.issubset(set(result.keys()))


def test_atlas_benchmark_sasa_nan_when_not_provided():
    p = _coords(10, 6)
    q = _coords(10, 6)
    result = atlas_benchmark(p, q, sasa_p=None, sasa_q=None)
    assert np.isnan(result["sasa_pearson"])


def test_atlas_benchmark_with_sasa():
    T, N = 10, 6
    p = _coords(T, N)
    q = _coords(T, N)
    sasa_p = RNG.uniform(0, 100, (T, N))
    sasa_q = RNG.uniform(0, 100, (T, N))
    result = atlas_benchmark(p, q, sasa_p=sasa_p, sasa_q=sasa_q)
    assert not np.isnan(result["sasa_pearson"])


def test_atlas_benchmark_identical_input():
    """Self-comparison: pairwise r=1, PCA w2~0, jaccard=1, rmsf pearson=1."""
    T, N = 15, 6
    c = _coords(T, N)
    result = atlas_benchmark(c, c)
    assert result["pairwise_rmsd_pearson"] == pytest.approx(1.0, abs=1e-6)
    assert result["pca_w2"]["mean_w2"] == pytest.approx(0.0, abs=1e-9)
    assert result["transient_contact_jaccard"] == pytest.approx(1.0)

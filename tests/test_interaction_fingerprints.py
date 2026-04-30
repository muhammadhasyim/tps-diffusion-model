"""Tests for genai_tps.evaluation.interaction_fingerprints."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("prolif", reason="ProLIF dependency for interaction fingerprints")

from genai_tps.evaluation.interaction_fingerprints import (
    compute_interaction_distances,
    interaction_ws_distances,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# interaction_ws_distances
# ---------------------------------------------------------------------------

def _make_fp(n_frames: int = 50) -> dict[str, np.ndarray]:
    return {
        "LIG:HBAcceptor": (RNG.random(n_frames) > 0.5).astype(float),
        "LIG:Hydrophobic": (RNG.random(n_frames) > 0.3).astype(float),
        "LIG:PiStacking": (RNG.random(n_frames) > 0.8).astype(float),
    }


def test_ws_output_keys():
    fp_p = _make_fp()
    fp_r = _make_fp()
    result = interaction_ws_distances(fp_p, fp_r)
    assert set(result.keys()) >= {"per_interaction_w1", "mean_w1", "n_interactions", "labels"}


def test_ws_identical_zero():
    """Comparing a fingerprint dict with itself gives zero mean W1."""
    fp = _make_fp()
    result = interaction_ws_distances(fp, fp)
    assert result["mean_w1"] == pytest.approx(0.0, abs=1e-10)


def test_ws_shared_interactions():
    """Only shared interactions are compared when only_shared=True."""
    fp_p = {"A": np.zeros(10), "B": np.ones(10)}
    fp_r = {"B": np.ones(10), "C": np.zeros(10)}
    result = interaction_ws_distances(fp_p, fp_r, only_shared=True)
    assert result["n_interactions"] == 1  # only "B" is shared
    assert result["labels"] == ["B"]


def test_ws_all_interactions_when_not_shared():
    """With only_shared=False, all labels from both dicts are included."""
    fp_p = {"A": np.zeros(10)}
    fp_r = {"B": np.zeros(10)}
    result = interaction_ws_distances(fp_p, fp_r, only_shared=False)
    assert result["n_interactions"] == 2


def test_ws_empty_intersection_returns_nan():
    """Disjoint label sets with only_shared=True give NaN mean_w1."""
    result = interaction_ws_distances({"A": np.zeros(5)}, {"B": np.zeros(5)}, only_shared=True)
    assert result["n_interactions"] == 0
    assert np.isnan(result["mean_w1"])


def test_ws_per_interaction_range():
    """Per-interaction W1 must be non-negative."""
    fp_p = _make_fp(200)
    fp_r = _make_fp(200)
    result = interaction_ws_distances(fp_p, fp_r)
    for label, w1 in result["per_interaction_w1"].items():
        assert w1 >= 0.0, f"Negative W1 for {label}: {w1}"


def test_ws_binary_occupation_diff():
    """For binary 0/1 arrays W1 = |mean(P) - mean(Q)| exactly."""
    arr_p = np.array([1., 1., 0., 0.])  # occupation = 0.5
    arr_r = np.array([1., 1., 1., 0.])  # occupation = 0.75
    fp_p = {"A": arr_p}
    fp_r = {"A": arr_r}
    result = interaction_ws_distances(fp_p, fp_r)
    expected = abs(arr_p.mean() - arr_r.mean())
    assert result["per_interaction_w1"]["A"] == pytest.approx(expected, abs=1e-9)


def test_ws_n_interactions_matches_labels():
    fp_p = _make_fp()
    fp_r = _make_fp()
    result = interaction_ws_distances(fp_p, fp_r)
    assert result["n_interactions"] == len(result["labels"])
    assert result["n_interactions"] == len(result["per_interaction_w1"])


# ---------------------------------------------------------------------------
# compute_interaction_distances
# ---------------------------------------------------------------------------

def test_compute_interaction_distances_is_callable():
    """compute_interaction_distances must be importable and callable."""
    assert callable(compute_interaction_distances)

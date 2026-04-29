"""Unit tests for PoseBusters trajectory CV helpers (no real PoseBusters I/O)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genai_tps.evaluation.posebusters import (
    POSEBUSTERS_CV_PREFIX,
    cv_name_for_column,
    expand_bias_cv_posebusters_all,
    make_posebusters_cached_column_scalar_fns,
    make_posebusters_pass_fraction_traj_fn,
    pass_fraction_from_row,
    validate_posebusters_bias_cv_names,
    vector_from_row,
)


class _PbRow:
    """Minimal row shaped like ``pd.Series`` for ``pass_fraction_from_row`` / ``vector_from_row``."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    @property
    def values(self):
        return self._data.values()

    @property
    def index(self):
        return list(self._data.keys())

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)


def test_pass_fraction_from_row_bools() -> None:
    row = _PbRow({"a": True, "b": False, "c": np.bool_(True)})
    assert pass_fraction_from_row(row) == pytest.approx(2.0 / 3.0)


def test_pass_fraction_from_row_no_bools() -> None:
    row = _PbRow({"x": 1.0, "y": "ok"})
    assert pass_fraction_from_row(row) == 0.0


def test_vector_from_row() -> None:
    row = _PbRow({"chk_a": True, "chk_b": False, "num": 0.5})
    v = vector_from_row(row, ["chk_a", "chk_b", "missing", "num"])
    assert v[0] == 1.0 and v[1] == 0.0 and np.isnan(v[2]) and v[3] == 0.5


def test_cv_name_for_column() -> None:
    assert cv_name_for_column("My Check!") == f"{POSEBUSTERS_CV_PREFIX}my_check"


def test_validate_posebusters_bias_cv_names_pass_fraction_alone_ok() -> None:
    validate_posebusters_bias_cv_names(["posebusters_pass_fraction"])


def test_validate_posebusters_bias_cv_names_pass_fraction_mixed() -> None:
    with pytest.raises(ValueError, match="posebusters_pass_fraction cannot"):
        validate_posebusters_bias_cv_names(["posebusters_pass_fraction", "rg"])


def test_validate_posebusters_bias_cv_names_pb_cols_mixed() -> None:
    with pytest.raises(ValueError, match="posebusters__"):
        validate_posebusters_bias_cv_names([f"{POSEBUSTERS_CV_PREFIX}a", "rg"])


def test_expand_bias_cv_posebusters_all(monkeypatch: pytest.MonkeyPatch) -> None:
    import genai_tps.evaluation.posebusters as mod

    def _fake_expand(*_a, **_k):
        return [f"{POSEBUSTERS_CV_PREFIX}a", f"{POSEBUSTERS_CV_PREFIX}b"], ["col_a", "col_b"]

    monkeypatch.setattr(mod, "expand_posebusters_all_to_cv_names", _fake_expand)
    new, raw, cvs = mod.expand_bias_cv_posebusters_all(
        "posebusters_all",
        None,
        0,
        Path("/tmp/pb_scratch"),
        np.zeros((1, 3), dtype=np.float32),
    )
    assert new == f"{POSEBUSTERS_CV_PREFIX}a,{POSEBUSTERS_CV_PREFIX}b"
    assert raw == ["col_a", "col_b"]
    assert cvs == [f"{POSEBUSTERS_CV_PREFIX}a", f"{POSEBUSTERS_CV_PREFIX}b"]


def test_expand_bias_cv_posebusters_all_requires_sole_token() -> None:
    with pytest.raises(ValueError, match="only token"):
        expand_bias_cv_posebusters_all(
            f"posebusters_all,{POSEBUSTERS_CV_PREFIX}x",
            None,
            0,
            Path("/tmp"),
            np.zeros((1, 3)),
        )


def _mock_traj(n_atoms: int = 5):
    class _Snap:
        pass

    snap = _Snap()
    snap.tensor_coords = torch.randn(1, n_atoms, 3)
    traj = [snap]

    class _T(list):
        pass

    return _T(traj)


def test_make_posebusters_pass_fraction_traj_fn() -> None:
    class _Ev:
        n_struct = 5

        def bust_row(self, coords, full_report: bool = False):
            assert coords.shape == (5, 3)
            return _PbRow({"u": True, "v": False})

    fn = make_posebusters_pass_fraction_traj_fn(_Ev())
    out = fn(_mock_traj(5))
    assert out == pytest.approx(0.5)


def test_make_posebusters_cached_column_scalar_fns_single_bust_per_frame() -> None:
    calls = {"n": 0}

    class _Ev:
        n_struct = 5

        def bust_row(self, coords, full_report: bool = False):
            calls["n"] += 1
            return _PbRow({"alpha": True, "beta": False})

    fns = make_posebusters_cached_column_scalar_fns(_Ev(), ["alpha", "beta"])
    traj = _mock_traj(5)
    assert fns[0](traj) == 1.0
    assert fns[1](traj) == 0.0
    assert calls["n"] == 1
    # new frame id -> bust again
    traj2 = _mock_traj(5)
    assert fns[0](traj2) == 1.0
    assert calls["n"] == 2


@patch("genai_tps.evaluation.posebusters.PoseBustersTrajEvaluator")
def test_probe_column_names_uses_evaluator(mock_ev_cls, tmp_path: Path) -> None:
    from genai_tps.evaluation.posebusters import probe_column_names

    inst = mock_ev_cls.return_value
    inst.bust_row.return_value = _PbRow({"chk": True})

    cols = probe_column_names(
        object(),
        3,
        tmp_path,
        np.zeros((3, 3), dtype=np.float32),
        mode="dock",
    )
    assert cols == ["chk"]
    mock_ev_cls.assert_called_once()
    inst.bust_row.assert_called_once()

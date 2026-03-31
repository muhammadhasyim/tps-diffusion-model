"""Run PoseBusters on Boltz trajectory frames (protein–ligand complexes).

Uses ``PoseBusters(..., max_workers=0)`` for single-process evaluation inside OPES loops.
See https://github.com/maabuu/posebusters and :doc:`docs/posebusters_opes_cv.md`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Boltz chain_type_ids: PROTEIN=0, NONPOLYMER=3
_PROTEIN_MOL_TYPE = 0
_NONPOLYMER_MOL_TYPE = 3

POSEBUSTERS_CV_PREFIX = "posebusters__"


def _ligand_atom_indices(structure: Any, ligand_chain: str | None) -> tuple[set[int], set[int]]:
    """Return (protein_atom_indices, ligand_atom_indices) for one ligand chain."""
    protein: set[int] = set()
    ligand: set[int] = set()
    nonpoly_chains: list[tuple[str, int, int]] = []
    for chain in structure.chains:
        start = int(chain["atom_idx"])
        n = int(chain["atom_num"])
        mol_type = int(chain["mol_type"])
        name = str(chain["name"]).strip()
        idxs = set(range(start, start + n))
        if mol_type == _PROTEIN_MOL_TYPE:
            protein |= idxs
        elif mol_type == _NONPOLYMER_MOL_TYPE:
            nonpoly_chains.append((name, start, n))
    if not nonpoly_chains:
        return protein, set()
    if ligand_chain is not None:
        pick = next((c for c in nonpoly_chains if c[0] == ligand_chain), None)
        if pick is None:
            raise ValueError(
                f"ligand_chain {ligand_chain!r} not found among NONPOLYMER chains "
                f"{[c[0] for c in nonpoly_chains]}"
            )
        name, start, n = pick
        ligand = set(range(start, start + n))
    else:
        nonpoly_chains.sort(key=lambda t: t[1])
        _, start, n = nonpoly_chains[0]
        ligand = set(range(start, start + n))
    return protein, ligand


def _complex_pdb_lines(
    coords: np.ndarray,
    structure: Any,
    n_struct: int,
) -> list[str]:
    """Return PDB lines (split) from Boltz ``to_pdb`` for the first ``n_struct`` atoms."""
    from dataclasses import replace  # noqa: PLC0415

    from boltz.data.types import Coords, Interface  # noqa: PLC0415
    from boltz.data.write.pdb import to_pdb  # noqa: PLC0415

    c = np.asarray(coords, dtype=np.float32)[: int(n_struct)]
    atoms = structure.atoms.copy()
    atoms["coords"] = c
    atoms["is_present"] = True
    residues = structure.residues.copy()
    residues["is_present"] = True
    coord_arr = np.array([(x,) for x in c], dtype=Coords)
    interfaces = np.array([], dtype=Interface)
    new_s = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_arr,
    )
    text = to_pdb(new_s, plddts=None, boltz2=True)
    return [ln for ln in text.splitlines() if ln.strip()]


def _pdb_lines_for_atom_subset(
    all_lines: list[str],
    atom_indices: set[int],
) -> str:
    """Keep ATOM/HETATM lines whose running atom index is in ``atom_indices``."""
    out: list[str] = []
    i = 0
    for line in all_lines:
        if line.startswith(("ATOM", "HETATM")):
            if i in atom_indices:
                out.append(line)
            i += 1
    return "\n".join(out) + "\nEND\n"


def _ligand_mol_from_pdb_block(pdb_block: str):
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdmolops  # noqa: PLC0415

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    frag = max(rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False), key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(frag)
    except Exception:  # noqa: BLE001
        return None
    return frag


def _write_ligand_sdf(mol, path: Path) -> None:
    from rdkit.Chem import SDWriter  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    w = SDWriter(str(path))
    w.write(mol)
    w.close()


class PoseBustersTrajEvaluator:
    """Evaluate PoseBusters for coordinates + Boltz topology (writes temp PDB/SDF)."""

    def __init__(
        self,
        structure: Any,
        n_struct: int,
        scratch_dir: Path,
        *,
        mode: str = "redock_fast",
        reference_ligand_sdf: Path | None = None,
        initial_coords_for_reference_ligand: np.ndarray | None = None,
        ligand_chain: str | None = None,
    ) -> None:
        from posebusters import PoseBusters  # noqa: PLC0415

        self.structure = structure
        self.n_struct = int(n_struct)
        self.scratch = Path(scratch_dir)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.ligand_chain = ligand_chain
        self.protein_idx, self.ligand_idx = _ligand_atom_indices(structure, ligand_chain)
        if not self.ligand_idx:
            raise ValueError("No NONPOLYMER (ligand) chain found in Boltz structure.")
        self._pb = PoseBusters(config=mode, max_workers=0)
        self._pred_sdf = self.scratch / "posebusters_pred.sdf"
        self._prot_pdb = self.scratch / "posebusters_protein.pdb"

        self._mol_true_path: Path | None = None
        if mode.startswith("redock"):
            if reference_ligand_sdf is not None:
                p = Path(reference_ligand_sdf).expanduser().resolve()
                if not p.is_file():
                    raise FileNotFoundError(f"reference ligand SDF not found: {p}")
                self._mol_true_path = p
            elif initial_coords_for_reference_ligand is not None:
                block = _pdb_lines_for_atom_subset(
                    _complex_pdb_lines(initial_coords_for_reference_ligand, structure, n_struct),
                    self.ligand_idx,
                )
                mol = _ligand_mol_from_pdb_block(block)
                if mol is None:
                    raise RuntimeError("Could not build reference ligand mol from initial coordinates.")
                self._mol_true_path = self.scratch / "posebusters_ref_ligand.sdf"
                _write_ligand_sdf(mol, self._mol_true_path)
            else:
                raise ValueError(
                    "PoseBusters redock modes require reference_ligand_sdf or "
                    "initial_coords_for_reference_ligand."
                )

    def bust_row(self, coords_angstrom: np.ndarray, *, full_report: bool = False) -> pd.Series:
        """Run PoseBusters for one structure; return the first (only) result row."""
        lines = _complex_pdb_lines(coords_angstrom, self.structure, self.n_struct)
        prot_block = _pdb_lines_for_atom_subset(lines, self.protein_idx)
        lig_block = _pdb_lines_for_atom_subset(lines, self.ligand_idx)
        self._prot_pdb.write_text(prot_block, encoding="utf-8")
        mol = _ligand_mol_from_pdb_block(lig_block)
        if mol is None:
            return pd.Series(dtype=object)
        _write_ligand_sdf(mol, self._pred_sdf)

        if self._mol_true_path is not None:
            df = self._pb.bust(
                str(self._pred_sdf),
                str(self._mol_true_path),
                str(self._prot_pdb),
                full_report=full_report,
            )
        else:
            df = self._pb.bust(
                str(self._pred_sdf),
                mol_cond=str(self._prot_pdb),
                full_report=full_report,
            )
        if df.shape[0] == 0:
            return pd.Series(dtype=object)
        return df.iloc[0]

    def bust_snapshot(self, snapshot, *, full_report: bool = False) -> pd.Series:
        """Run PoseBusters on one snapshot, with host readback isolated here."""
        from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy  # noqa: PLC0415

        coords = snapshot_frame_numpy_copy(snapshot)[: self.n_struct]
        return self.bust_row(coords, full_report=full_report)


def pass_fraction_from_row(row: pd.Series) -> float:
    """Mean of boolean columns in a PoseBusters result row (in ``[0, 1]``)."""
    vals: list[float] = []
    for v in row.values:
        if isinstance(v, (bool, np.bool_)):
            vals.append(float(v))
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def vector_from_row(row: pd.Series, columns: list[str]) -> np.ndarray:
    """Map named columns to floats (bools as 0/1, missing as NaN)."""
    out = np.full(len(columns), np.nan, dtype=np.float64)
    for i, name in enumerate(columns):
        if name not in row.index:
            continue
        v = row[name]
        if isinstance(v, (bool, np.bool_)):
            out[i] = float(v)
        elif isinstance(v, int | float | np.integer | np.floating):
            out[i] = float(v)
    return out


def probe_column_names(
    structure: Any,
    n_struct: int,
    scratch_dir: Path,
    probe_coords: np.ndarray,
    *,
    mode: str = "redock_fast",
    reference_ligand_sdf: Path | None = None,
    initial_coords_for_reference_ligand: np.ndarray | None = None,
    ligand_chain: str | None = None,
    full_report: bool = False,
) -> list[str]:
    """One evaluation to list default output columns for ``mode``."""
    ev = PoseBustersTrajEvaluator(
        structure,
        n_struct,
        scratch_dir,
        mode=mode,
        reference_ligand_sdf=reference_ligand_sdf,
        initial_coords_for_reference_ligand=initial_coords_for_reference_ligand,
        ligand_chain=ligand_chain,
    )
    row = ev.bust_row(probe_coords, full_report=full_report)
    idx = row.index
    col_list = idx.tolist() if hasattr(idx, "tolist") else list(idx)
    return [str(c) for c in col_list]


def cv_name_for_column(col: str) -> str:
    """Stable OPES CV token for a PoseBusters dataframe column."""
    safe = re.sub(r"[^\w%.-]+", "_", str(col).strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"{POSEBUSTERS_CV_PREFIX}{safe}"


def expand_posebusters_all_to_cv_names(
    structure: Any,
    n_struct: int,
    scratch_dir: Path,
    probe_coords: np.ndarray,
    **eval_kw: Any,
) -> tuple[list[str], list[str]]:
    """Return ``(cv_names, raw_column_names)`` for ``posebusters_all`` expansion."""
    cols = probe_column_names(
        structure,
        n_struct,
        scratch_dir,
        probe_coords,
        **eval_kw,
    )
    names: list[str] = []
    seen: set[str] = set()
    for c in cols:
        base = cv_name_for_column(c)
        name = base
        j = 0
        while name in seen:
            j += 1
            name = f"{base}_{j}"
        seen.add(name)
        names.append(name)
    return names, cols


def expand_bias_cv_posebusters_all(
    bias_cv: str,
    structure: Any,
    n_struct: int,
    scratch_dir: Path,
    probe_coords: np.ndarray,
    **eval_kw: Any,
) -> tuple[str, list[str] | None, list[str] | None]:
    """Replace sole ``posebusters_all`` with comma-separated ``posebusters__*`` tokens.

    Returns ``(new_bias_cv_string, raw_columns, cv_names)`` when expansion ran;
    otherwise ``(bias_cv, None, None)``.  ``raw_columns`` aligns index-wise with
    ``cv_names``.
    """
    parts = [p.strip() for p in bias_cv.split(",") if p.strip()]
    if "posebusters_all" not in parts:
        return bias_cv, None, None
    if parts != ["posebusters_all"]:
        raise ValueError(
            "posebusters_all must be the only token in --bias-cv (no comma-separated partners)."
        )
    cv_names, raw_cols = expand_posebusters_all_to_cv_names(
        structure,
        n_struct,
        scratch_dir,
        probe_coords,
        **eval_kw,
    )
    return ",".join(cv_names), raw_cols, cv_names


def validate_posebusters_bias_cv_names(names: list[str]) -> None:
    """Raise ``ValueError`` if PoseBusters tokens are mixed illegally."""
    has_pf = "posebusters_pass_fraction" in names
    pb_cols = [n for n in names if n.startswith(POSEBUSTERS_CV_PREFIX)]
    if has_pf and len(names) > 1:
        raise ValueError(
            "posebusters_pass_fraction cannot be combined with other --bias-cv names; "
            "use a separate run or only posebusters_all / posebusters__* alone."
        )
    if pb_cols and not all(n.startswith(POSEBUSTERS_CV_PREFIX) for n in names):
        raise ValueError(
            "posebusters__* CVs cannot be mixed with non-PoseBusters CVs in one bias."
        )


def make_posebusters_pass_fraction_traj_fn(evaluator: PoseBustersTrajEvaluator) -> Callable[..., float]:
    """``f(traj) -> float`` mean boolean pass rate for one trajectory step."""
    from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy  # noqa: PLC0415

    n = evaluator.n_struct

    def _fn(traj) -> float:
        if hasattr(evaluator, "bust_snapshot"):
            row = evaluator.bust_snapshot(traj[-1])
        else:
            coords = snapshot_frame_numpy_copy(traj[-1])[:n]
            row = evaluator.bust_row(coords)
        return float(pass_fraction_from_row(row))

    return _fn


def make_posebusters_cached_column_scalar_fns(
    evaluator: PoseBustersTrajEvaluator,
    columns: list[str],
) -> list[Callable[..., float]]:
    """One PoseBusters evaluation per trajectory frame; each callable returns one column as 0/1 or float."""
    from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy  # noqa: PLC0415

    n = evaluator.n_struct
    cache: dict[str, Any] = {"snap_id": None, "row": None}

    def _make_fn(col: str) -> Callable[..., float]:
        def _fn(traj) -> float:
            snap = traj[-1]
            sid = id(snap)
            if cache["snap_id"] != sid:
                if hasattr(evaluator, "bust_snapshot"):
                    cache["row"] = evaluator.bust_snapshot(snap)
                else:
                    coords = snapshot_frame_numpy_copy(snap)[:n]
                    cache["row"] = evaluator.bust_row(coords)
                cache["snap_id"] = sid
            row = cache["row"]
            if row is None or len(row) == 0:
                return 0.0
            v = float(vector_from_row(row, [col])[0])
            return 0.0 if np.isnan(v) else v

        return _fn

    return [_make_fn(c) for c in columns]

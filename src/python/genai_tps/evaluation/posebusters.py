"""PoseBusters evaluation: tensor-native GPU checks and upstream PoseBusters on trajectory frames.

This package merges the former :mod:`posebusters_gpu` (geometry on torch tensors) and
:mod:`posebusters_traj` (temp PDB/SDF round-trip via the ``posebusters`` package).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
from genai_tps.backends.boltz.snapshot import snapshot_frame_numpy_copy, snapshot_frame_tensor_view

POSEBUSTERS_GPU_CV_PREFIX = "posebusters_gpu__"
POSEBUSTERS_GPU_PASS_FRACTION = "posebusters_gpu_pass_fraction"

_PROTEIN_MOL_TYPE = 0
_NONPOLYMER_MOL_TYPE = 3

_GPU_CHECK_SPECS: tuple[tuple[str, str], ...] = (
    ("ligand_rmsd_le_2a", "Ligand pose RMSD <= 2 A after protein-frame Kabsch alignment."),
    ("ligand_pocket_dist_le_6a", "Ligand COM to pocket COM distance <= 6 A."),
    ("ligand_contacts_ge_5", "Protein-ligand contact score >= 5."),
    ("ligand_hbonds_ge_1", "At least one protein-ligand N/O contact within cutoff."),
    ("clash_count_le_4", "Steric clash proxy count <= 4."),
    ("ligand_max_extent_le_8a", "Largest ligand pair distance <= 8 A."),
    ("ligand_bbox_volume_le_250a3", "Ligand bounding-box volume <= 250 A^3."),
)


def gpu_check_columns() -> list[str]:
    """Stable ordered GPU check names used by ``posebusters_gpu_all``."""
    return [name for name, _ in _GPU_CHECK_SPECS]


def cv_name_for_gpu_column(col: str) -> str:
    """Stable OPES CV token for one GPU-native check."""
    return f"{POSEBUSTERS_GPU_CV_PREFIX}{col}"


def expand_posebusters_gpu_all_to_cv_names() -> tuple[list[str], list[str]]:
    """Return ``(cv_names, raw_column_names)`` for ``posebusters_gpu_all``."""
    cols = gpu_check_columns()
    return [cv_name_for_gpu_column(c) for c in cols], cols


def expand_bias_cv_posebusters_gpu_all(
    bias_cv: str,
) -> tuple[str, list[str] | None, list[str] | None]:
    """Replace sole ``posebusters_gpu_all`` with comma-separated GPU check tokens."""
    parts = [p.strip() for p in bias_cv.split(",") if p.strip()]
    if "posebusters_gpu_all" not in parts:
        return bias_cv, None, None
    if parts != ["posebusters_gpu_all"]:
        raise ValueError(
            "posebusters_gpu_all must be the only token in --bias-cv (no comma-separated partners)."
        )
    cv_names, raw_cols = expand_posebusters_gpu_all_to_cv_names()
    return ",".join(cv_names), raw_cols, cv_names


def validate_posebusters_gpu_bias_cv_names(names: list[str]) -> None:
    """Raise when GPU PoseBusters names are mixed illegally."""
    has_pf = POSEBUSTERS_GPU_PASS_FRACTION in names
    gpu_cols = [n for n in names if n.startswith(POSEBUSTERS_GPU_CV_PREFIX)]
    if has_pf and len(names) > 1:
        raise ValueError(
            "posebusters_gpu_pass_fraction cannot be combined with other --bias-cv names; "
            "use a separate run or only posebusters_gpu_all / posebusters_gpu__* alone."
        )
    if gpu_cols and not all(n.startswith(POSEBUSTERS_GPU_CV_PREFIX) for n in names):
        raise ValueError(
            "posebusters_gpu__* CVs cannot be mixed with non-GPU-PoseBusters CVs in one bias."
        )


def pass_fraction_from_gpu_row(row: dict[str, float]) -> float:
    """Mean of boolean-like GPU check outputs in ``[0, 1]``."""
    vals = [float(v) for v in row.values() if np.isfinite(v)]
    if not vals:
        return 0.0
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def vector_from_gpu_row(row: dict[str, float], columns: list[str]) -> np.ndarray:
    """Map named GPU check columns to a numeric vector."""
    out = np.full(len(columns), np.nan, dtype=np.float64)
    for i, name in enumerate(columns):
        if name in row:
            out[i] = float(row[name])
    return out


def _ligand_atom_indices(structure: Any, ligand_chain: str | None) -> tuple[np.ndarray, np.ndarray]:
    protein: list[int] = []
    ligand: list[int] = []
    nonpoly_chains: list[tuple[str, int, int]] = []
    for chain in structure.chains:
        start = int(chain["atom_idx"])
        n = int(chain["atom_num"])
        mol_type = int(chain["mol_type"])
        name = str(chain["name"]).strip()
        if mol_type == _PROTEIN_MOL_TYPE:
            protein.extend(range(start, start + n))
        elif mol_type == _NONPOLYMER_MOL_TYPE:
            nonpoly_chains.append((name, start, n))
    if ligand_chain is not None:
        pick = next((c for c in nonpoly_chains if c[0] == ligand_chain), None)
        if pick is None:
            raise ValueError(
                f"ligand_chain {ligand_chain!r} not found among NONPOLYMER chains "
                f"{[c[0] for c in nonpoly_chains]}"
            )
        _, start, n = pick
        ligand.extend(range(start, start + n))
    elif nonpoly_chains:
        nonpoly_chains.sort(key=lambda t: t[1])
        _, start, n = nonpoly_chains[0]
        ligand.extend(range(start, start + n))
    return np.asarray(protein, dtype=np.int64), np.asarray(ligand, dtype=np.int64)


def _kabsch_align_torch(
    mobile: torch.Tensor,
    reference: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Align ``points`` using the Kabsch transform fit from ``mobile`` to ``reference``."""
    c_mob = mobile.mean(dim=0)
    c_ref = reference.mean(dim=0)
    p = mobile - c_mob
    q = reference - c_ref
    h = p.T @ q
    u, _, vt = torch.linalg.svd(h)
    det_val = torch.linalg.det(vt.T @ u.T)
    correction = torch.diag(
        torch.tensor([1.0, 1.0, float(torch.sign(det_val))], device=mobile.device, dtype=mobile.dtype)
    )
    rot = vt.T @ correction @ u.T
    return (points - c_mob) @ rot.T + c_ref


@dataclass
class GPUPoseBustersEvaluator:
    """Evaluate a stable subset of PoseBusters-like geometry checks on tensors."""

    structure: Any
    n_struct: int
    reference_coords: np.ndarray
    pocket_radius: float = 6.0
    contact_r0: float = 3.5
    hbond_cutoff: float = 3.5
    ligand_chain: str | None = None
    backend_mode: str = "gpu_fast"
    cpu_evaluator: Any | None = None
    cpu_fallback_every: int = 0

    def __post_init__(self) -> None:
        self.n_struct = int(self.n_struct)
        self.reference_coords = np.asarray(self.reference_coords, dtype=np.float32)[: self.n_struct]
        protein_idx, ligand_idx = _ligand_atom_indices(self.structure, self.ligand_chain)
        if ligand_idx.size == 0:
            raise ValueError("No NONPOLYMER (ligand) chain found in Boltz structure.")
        self.protein_idx = protein_idx
        self.ligand_idx = ligand_idx
        self.indexer = PoseCVIndexer(
            self.structure,
            self.reference_coords,
            pocket_radius=float(self.pocket_radius),
        )
        self._eval_counter = 0
        self.latest_cpu_row: Any | None = None

    @property
    def ordered_columns(self) -> list[str]:
        return gpu_check_columns()

    def _coords_tensor(self, coords: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(coords, torch.Tensor):
            tc = coords
        else:
            tc = torch.as_tensor(coords, dtype=torch.float32)
        if tc.dim() == 3:
            tc = tc[0]
        return tc[: self.n_struct]

    def _ref_on(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ref = torch.as_tensor(self.reference_coords, dtype=coords.dtype, device=coords.device)
        lig_idx = torch.as_tensor(self.ligand_idx, dtype=torch.long, device=coords.device)
        return ref, ref[lig_idx]

    def _run_cpu_fallback_if_needed(self, snapshot) -> None:
        if self.backend_mode != "hybrid" or self.cpu_evaluator is None:
            return
        every = int(self.cpu_fallback_every)
        if every <= 0 or (self._eval_counter % every) != 0:
            return
        host_coords = snapshot_frame_numpy_copy(snapshot)[: self.n_struct]
        self.latest_cpu_row = self.cpu_evaluator.bust_row(host_coords)

    def evaluate_snapshot(self, snapshot) -> dict[str, float]:
        """Evaluate supported checks directly from a trajectory snapshot."""
        self._eval_counter += 1
        self._run_cpu_fallback_if_needed(snapshot)
        coords = snapshot_frame_tensor_view(snapshot)
        return self.evaluate_coords(coords)

    def evaluate_coords(self, coords: torch.Tensor | np.ndarray) -> dict[str, float]:
        """Evaluate supported checks from a tensor or ndarray of shape ``(N, 3)``."""
        x = self._coords_tensor(coords)
        ref, ref_lig = self._ref_on(x)
        lig_idx = torch.as_tensor(self.ligand_idx, dtype=torch.long, device=x.device)
        lig = x[lig_idx]
        lig_ref = ref_lig
        row: dict[str, float] = {name: 0.0 for name in self.ordered_columns}
        if lig.shape[0] == 0:
            return row

        if self.indexer.protein_ca_idx.size >= 3:
            ca_idx = torch.as_tensor(self.indexer.protein_ca_idx, device=x.device, dtype=torch.long)
            lig_aligned = _kabsch_align_torch(x[ca_idx], ref[ca_idx], lig)
        else:
            lig_aligned = lig
        lig_rmsd = torch.sqrt(((lig_aligned - lig_ref) ** 2).sum(dim=-1).mean() + 1e-24)
        row["ligand_rmsd_le_2a"] = float((lig_rmsd <= 2.0).item())

        if self.indexer.pocket_ca_idx.size > 0:
            pocket_ca_idx = torch.as_tensor(self.indexer.pocket_ca_idx, device=x.device, dtype=torch.long)
            lig_com = lig.mean(dim=0)
            pocket_com = x[pocket_ca_idx].mean(dim=0)
            lig_pocket_dist = torch.linalg.norm(lig_com - pocket_com)
            row["ligand_pocket_dist_le_6a"] = float((lig_pocket_dist <= 6.0).item())
        else:
            row["ligand_pocket_dist_le_6a"] = 0.0

        if self.indexer.pocket_heavy_idx.size > 0:
            pocket_idx = torch.as_tensor(self.indexer.pocket_heavy_idx, device=x.device, dtype=torch.long)
            prot = x[pocket_idx]
            dists = torch.cdist(prot.unsqueeze(0), lig.unsqueeze(0))[0]
            ratio = dists / float(self.contact_r0)
            r6 = ratio.pow(6)
            r12 = ratio.pow(12)
            num = 1.0 - r6
            den = 1.0 - r12
            degenerate = den.abs() < 1e-9
            s = torch.where(degenerate, torch.full_like(num, 0.5), num / (den + 1e-30)).clamp(0.0, 1.0)
            contact_score = s.sum()
            row["ligand_contacts_ge_5"] = float((contact_score >= 5.0).item())
            clash_count = (dists < 2.0).sum()
            row["clash_count_le_4"] = float((clash_count <= 4).item())
        else:
            row["ligand_contacts_ge_5"] = 0.0
            row["clash_count_le_4"] = 0.0

        if self.indexer.ligand_no_idx.size > 0 and self.indexer.pocket_no_idx.size > 0:
            lig_no_idx = torch.as_tensor(self.indexer.ligand_no_idx, device=x.device, dtype=torch.long)
            pocket_no_idx = torch.as_tensor(self.indexer.pocket_no_idx, device=x.device, dtype=torch.long)
            no_dists = torch.cdist(x[pocket_no_idx].unsqueeze(0), x[lig_no_idx].unsqueeze(0))[0]
            hbond_count = (no_dists < float(self.hbond_cutoff)).sum()
            row["ligand_hbonds_ge_1"] = float((hbond_count >= 1).item())
        else:
            row["ligand_hbonds_ge_1"] = 0.0

        if lig.shape[0] >= 2:
            pair_dists = torch.cdist(lig.unsqueeze(0), lig.unsqueeze(0))[0]
            row["ligand_max_extent_le_8a"] = float((pair_dists.max() <= 8.0).item())
        else:
            row["ligand_max_extent_le_8a"] = 1.0

        bbox = lig.max(dim=0).values - lig.min(dim=0).values
        bbox_volume = torch.prod(torch.clamp(bbox, min=0.0))
        row["ligand_bbox_volume_le_250a3"] = float((bbox_volume <= 250.0).item())
        return row


def make_posebusters_gpu_pass_fraction_traj_fn(
    evaluator: GPUPoseBustersEvaluator,
) -> Callable[..., float]:
    """``f(traj) -> float`` mean pass rate over GPU-native boolean checks."""

    def _fn(traj) -> float:
        row = evaluator.evaluate_snapshot(traj[-1])
        return float(pass_fraction_from_gpu_row(row))

    return _fn


def make_posebusters_gpu_cached_column_scalar_fns(
    evaluator: GPUPoseBustersEvaluator,
    columns: list[str],
) -> list[Callable[..., float]]:
    """One GPU evaluation per trajectory frame; each callable returns one GPU check."""
    cache: dict[str, Any] = {"snap_id": None, "row": None}

    def _make_fn(col: str) -> Callable[..., float]:
        def _fn(traj) -> float:
            snap = traj[-1]
            sid = id(snap)
            if cache["snap_id"] != sid:
                cache["row"] = evaluator.evaluate_snapshot(snap)
                cache["snap_id"] = sid
            row = cache["row"]
            if row is None:
                return 0.0
            v = float(vector_from_gpu_row(row, [col])[0])
            return 0.0 if np.isnan(v) else v

        return _fn

    return [_make_fn(c) for c in columns]

# ----- Upstream PoseBusters trajectory evaluation (temp PDB/SDF) -----

POSEBUSTERS_CV_PREFIX = "posebusters__"


def _pb_traj_ligand_atom_indices(structure: Any, ligand_chain: str | None) -> tuple[set[int], set[int]]:
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
    """Build sanitized RDKit mol from PDB block (largest fragment).

    Returns ``None`` when RDKit cannot parse the block.  Sanitization
    failures propagate to make ligand extraction errors visible.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdmolops  # noqa: PLC0415

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    frag = max(rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False), key=lambda m: m.GetNumAtoms())
    Chem.SanitizeMol(frag)
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
        self.protein_idx, self.ligand_idx = _pb_traj_ligand_atom_indices(structure, ligand_chain)
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
        """Run PoseBusters for one structure; return the first (only) result row.

        Raises
        ------
        RuntimeError
            When the ligand cannot be extracted from coordinates.
        """
        lines = _complex_pdb_lines(coords_angstrom, self.structure, self.n_struct)
        prot_block = _pdb_lines_for_atom_subset(lines, self.protein_idx)
        lig_block = _pdb_lines_for_atom_subset(lines, self.ligand_idx)
        self._prot_pdb.write_text(prot_block, encoding="utf-8")
        mol = _ligand_mol_from_pdb_block(lig_block)
        if mol is None:
            raise RuntimeError(
                "PoseBusters: could not extract ligand mol from coordinates. "
                "Check that the Boltz structure has a valid NONPOLYMER chain "
                f"(ligand_chain={self.ligand_chain!r}, "
                f"ligand_idx count={len(self.ligand_idx)})."
            )
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

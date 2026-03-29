"""Incremental Škrinjar-style training similarity (Foldseek + RDKit shape/SuCOS + pocket qcov).

This mirrors the **chemistry stack** in ``papers/runs-n-poses/similarity_scoring.py`` where
possible: Crippen pre-alignment, :func:`rdShapeAlign.AlignMol`, and SuCOS-like
pharmacophore + shape protrusion. **Foldseek** is optional for GPU structural
prefiltering; ligand scoring remains **CPU**-bound.

**Non-parity with the paper / PLINDER (important):**

- :func:`geometric_pocket_qcov_ca` (exposed also as :func:`pocket_qcov_ca`) is a **geometric
  proxy**, not PLINDER’s pocket_qcov from the Runs N’ Poses pipeline (Methods §7.4).
- Upstream ``similarity_scoring.py`` applies **Foldseek’s rigid (u, t)** to the **training
  ligand** before SuCOS when comparing holo systems; this incremental path **does not** —
  it superimposes receptors with a Kabsch on the first ``min(n_q, n_t)`` Cα for the pocket
  term and aligns ligands with RDKit only. Expect **numerical differences** vs Zenodo
  ``all_similarity_scores.parquet`` even when RDKit behaves perfectly.

Set ``SKRINJAR_LOG_ALIGN_FALLBACK=1`` to emit a warning whenever :func:`rdShapeAlign.AlignMol`
fails and the code falls back to Crippen-only alignment.

See ``docs/runs_n_poses_reproduction.md`` for batch reproduction context.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

# --- RDKit SuCOS-style scoring (adapted from runs-n-poses / SuCOS) -----------------

_FD_DEF = None
_FEAT_MAP_PARAMS = None
_PHARMACOPHORE_FEATURES = (
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
)


def _feat_factory():
    global _FD_DEF, _FEAT_MAP_PARAMS
    if _FD_DEF is not None:
        return _FD_DEF, _FEAT_MAP_PARAMS
    from rdkit import Chem, RDConfig  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415
    from rdkit.Chem.FeatMaps import FeatMaps  # noqa: PLC0415

    _FD_DEF = AllChem.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    _FEAT_MAP_PARAMS = {k: FeatMaps.FeatMapParams() for k in _FD_DEF.GetFeatureFamilies()}
    return _FD_DEF, _FEAT_MAP_PARAMS


def align_molecules_crippen(mol_ref, mol_probe, iterations: int = 100) -> None:
    from rdkit.Chem import rdMolAlign  # noqa: PLC0415

    crippen_o3a = rdMolAlign.GetCrippenO3A(mol_probe, mol_ref, maxIters=iterations)
    crippen_o3a.Align()


def _default_conf_id(mol) -> int:
    """First conformer id for shape alignment (0 if any conformers exist, else -1)."""
    n = int(mol.GetNumConformers())
    if n <= 0:
        return -1
    return 0


def align_molecules(
    reference,
    mobile,
    max_preiters: int = 100,
    max_postiters: int = 100,
    *,
    on_align_fallback: Callable[[], None] | None = None,
) -> tuple[float, ...]:
    """Crippen pre-align then :func:`rdShapeAlign.AlignMol` when the binding works.

    Some RDKit conda builds raise ``Boost.Python.ArgumentError`` for
    ``AlignMol``; we fall back to Crippen-only alignment so SuCOS / protrusion
    scores still run (parity with a fully working RDKit may differ slightly).
    """
    from rdkit.Chem import rdShapeAlign  # noqa: PLC0415

    align_molecules_crippen(reference, mobile, iterations=max_preiters)
    ref_cid = _default_conf_id(reference)
    mob_cid = _default_conf_id(mobile)
    try:
        return rdShapeAlign.AlignMol(
            reference,
            mobile,
            refConfId=ref_cid,
            probeConfId=mob_cid,
            max_preiters=max_preiters,
            max_postiters=max_postiters,
        )
    except Exception:  # noqa: BLE001 — Boost.ArgumentError or missing shape backend
        if os.environ.get("SKRINJAR_LOG_ALIGN_FALLBACK", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            _LOG.warning(
                "rdShapeAlign.AlignMol failed; using Crippen-only alignment — "
                "scores may diverge from papers/runs-n-poses similarity_scoring.py"
            )
        if on_align_fallback is not None:
            on_align_fallback()
        return (0.0, 0.0)


def _ensure_ring_info_for_featmaps(mol) -> None:
    """Ring perception required before :meth:`MolChemicalFeatureFactory.GetFeaturesForMol`."""
    from rdkit.Chem import rdmolops  # noqa: PLC0415

    try:
        rdmolops.FastFindRings(mol)
    except Exception:  # noqa: BLE001
        pass


def get_feature_map_score(mol_1, mol_2, score_mode=None) -> float:
    from rdkit.Chem.FeatMaps import FeatMaps  # noqa: PLC0415

    if score_mode is None:
        score_mode = FeatMaps.FeatMapScoreMode.All
    FDEF, FEAT_MAP_PARAMS = _feat_factory()
    feat_lists = []
    for molecule in (mol_1, mol_2):
        _ensure_ring_info_for_featmaps(molecule)
        raw_feats = FDEF.GetFeaturesForMol(molecule)
        feat_lists.append([f for f in raw_feats if f.GetFamily() in _PHARMACOPHORE_FEATURES])

    feat_maps = [
        FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=FEAT_MAP_PARAMS)
        for x in feat_lists
    ]
    feat_maps[0].scoreMode = score_mode

    score = feat_maps[0].ScoreFeats(feat_lists[1])
    denom = min(feat_maps[0].GetNumFeatures(), len(feat_lists[1]))
    return float(score / denom) if denom else 0.0


def get_sucos_score(mol_1, mol_2, score_mode=None) -> float:
    from rdkit.Chem import rdShapeHelpers  # noqa: PLC0415

    if score_mode is None:
        from rdkit.Chem.FeatMaps import FeatMaps  # noqa: PLC0415

        score_mode = FeatMaps.FeatMapScoreMode.All
    try:
        fm_score = float(np.clip(get_feature_map_score(mol_1, mol_2, score_mode), 0.0, 1.0))
        protrude_dist = float(
            np.clip(
                rdShapeHelpers.ShapeProtrudeDist(mol_1, mol_2, allowReordering=False),
                0.0,
                1.0,
            )
        )
        return float(0.5 * fm_score + 0.5 * (1.0 - protrude_dist))
    except Exception as exc:  # noqa: BLE001 — RDKit C++/FeatMaps can throw
        _LOG.debug("get_sucos_score failed (%s): %s", type(exc).__name__, exc)
        return 0.0


# --- Geometry ----------------------------------------------------------------------

def kabsch_rotation_translation(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Superimpose P onto Q (same length). Returns R (3,3), t (3,) with X' = X @ R.T + t."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P and Q must have identical shape (N, 3); got {P.shape}, {Q.shape}")
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T
    t = Qc - (Pc @ R.T)
    return R, t


def apply_rigid(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ R.T + t


def ca_coords_from_pdb(path: Path) -> np.ndarray:
    """Collect ATOM Cα coordinates in file order (Å)."""
    coords: list[list[float]] = []
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])
    return np.asarray(coords, dtype=np.float64)


def ligand_heavy_coords_from_pdb(path: Path) -> np.ndarray:
    """HETATM heavy-atom coordinates, excluding common solvent/residue names."""
    skip = {"HOH", "WAT", "NA", "CL", "K", "BR", "I", "MG", "CA", "ZN", "FE", "SOD", "CLA"}
    coords: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("HETATM"):
            continue
        if len(line) < 54:
            continue
        resname = line[17:20].strip()
        if resname in skip:
            continue
        elem = line[76:78].strip() if len(line) > 77 else ""
        if elem == "H" or (not elem and line[12:16].upper().startswith("H")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])
    return np.asarray(coords, dtype=np.float64)


def geometric_pocket_qcov_ca(
    query_pdb: Path,
    target_pdb: Path,
    *,
    pocket_radius: float = 6.0,
    match_cutoff: float = 2.0,
) -> float:
    """Geometric proxy: fraction of query-pocket Cα matched to target-pocket Cα.

    This is **not** PLINDER pocket_qcov from the benchmark pipeline; see module docstring.
    """
    q_ca = ca_coords_from_pdb(query_pdb)
    t_ca = ca_coords_from_pdb(target_pdb)
    q_lig = ligand_heavy_coords_from_pdb(query_pdb)
    t_lig = ligand_heavy_coords_from_pdb(target_pdb)
    if q_ca.size == 0 or t_ca.size == 0 or q_lig.size == 0 or t_lig.size == 0:
        return 0.0

    d_ql = np.linalg.norm(q_ca[:, None, :] - q_lig[None, :, :], axis=-1)
    q_pocket = np.nonzero(d_ql.min(axis=1) < pocket_radius)[0]
    d_tl = np.linalg.norm(t_ca[:, None, :] - t_lig[None, :, :], axis=-1)
    t_pocket = np.nonzero(d_tl.min(axis=1) < pocket_radius)[0]
    if q_pocket.size == 0:
        return 0.0
    if t_pocket.size == 0:
        return 0.0

    n = int(min(len(q_ca), len(t_ca)))
    if n < 3:
        return 0.0
    R, t = kabsch_rotation_translation(t_ca[:n], q_ca[:n])
    t_pocket_xyz = apply_rigid(t_ca[t_pocket], R, t)
    q_xyz = q_ca[q_pocket]
    dists = np.linalg.norm(q_xyz[:, None, :] - t_pocket_xyz[None, :, :], axis=-1)
    covered = np.any(dists < match_cutoff, axis=1)
    return float(np.count_nonzero(covered) / len(q_pocket))


def pocket_qcov_ca(*args: Any, **kwargs: Any) -> float:
    """Alias for :func:`geometric_pocket_qcov_ca` (backward-compatible name)."""
    return geometric_pocket_qcov_ca(*args, **kwargs)


def pick_largest_organic_fragment(mol):
    from rdkit.Chem import rdmolops  # noqa: PLC0415

    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    best = None
    best_h = 0
    for f in frags:
        nh = f.GetNumHeavyAtoms()
        if nh < 2:
            continue
        if nh > best_h:
            best_h = nh
            best = f
    return best


def mol_from_sdf_path(sdf: Path):
    """Load a training ligand SDF with sanitization so FeatMaps / SuCOS are safe."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromMolFile(str(sdf), sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolFile(str(sdf), sanitize=False, strictParsing=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return None
    _ensure_ring_info_for_featmaps(mol)
    return mol


def ligand_mol_from_pdb(path: Path):
    """Largest organic fragment from PDB, sanitized for FeatMaps / SuCOS.

    Incomplete PDB chemistry often leaves molecules without ring perception;
    :func:`GetFeaturesForMol` then aborts. We sanitize when possible, otherwise
    rebuild from SMILES and embed a conformer for shape scoring.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        return None
    frag = pick_largest_organic_fragment(mol)
    if frag is None:
        return None
    try:
        Chem.SanitizeMol(frag)
        frag_h = Chem.AddHs(frag)
        seed = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) % (2**31)
        if frag_h.GetNumConformers() == 0 and frag_h.GetNumAtoms() > 0:
            AllChem.EmbedMolecule(frag_h, randomSeed=seed)
        _ensure_ring_info_for_featmaps(frag_h)
        return frag_h
    except Exception:  # noqa: BLE001
        pass
    try:
        smi = Chem.MolToSmiles(frag)
        fresh = Chem.MolFromSmiles(smi) if smi else None
        n_h = int(frag.GetNumHeavyAtoms())
        if fresh is None and n_h == 2:
            fresh = Chem.MolFromSmiles("CC")
        if fresh is None and n_h == 1:
            fresh = Chem.MolFromSmiles("C")
        if fresh is None:
            return None
        fresh = Chem.AddHs(fresh)
        seed = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) % (2**31)
        if AllChem.EmbedMolecule(fresh, randomSeed=seed) != 0:
            AllChem.EmbedMolecule(fresh, randomSeed=seed + 1)
        Chem.SanitizeMol(fresh)
        _ensure_ring_info_for_featmaps(fresh)
        return fresh
    except Exception:  # noqa: BLE001
        return None


def sucos_shape_after_align(
    reference,
    mobile,
    *,
    on_align_fallback: Callable[[], None] | None = None,
) -> float:
    """SuCOS score after Crippen + rdShapeAlign (reference fixed, mobile moved)."""
    m_ref = reference.__copy__()
    m_mob = mobile.__copy__()
    align_molecules(m_ref, m_mob, on_align_fallback=on_align_fallback)
    return get_sucos_score(m_ref, m_mob)


# --- Foldseek ----------------------------------------------------------------------

# PDB ids are four chars; avoid ``\\b`` (underscore is a word char in Python).
_PDB_ID_RE = re.compile(r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")


def _pdb_id_from_subject_field(subject: str) -> str | None:
    subject = subject.strip()
    if not subject:
        return None
    m = _PDB_ID_RE.search(subject)
    if m:
        return m.group(1).lower()
    head = subject.split("_", 1)[0].strip()
    if len(head) == 4 and head[0].isdigit() and head[1:].isalnum():
        return head.lower()
    return None


def parse_pdb_ids_from_foldseek_output(text: str, max_hits: int) -> list[str]:
    """Parse target PDB ids from Foldseek/BLAST m8-style output."""
    seen: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = _pdb_id_from_subject_field(parts[1])
            if pid is not None:
                if pid not in seen:
                    seen.append(pid)
                    if len(seen) >= max_hits:
                        return seen
                continue
        for m in _PDB_ID_RE.finditer(line):
            pid = m.group(1).lower()
            if pid not in seen:
                seen.append(pid)
            if len(seen) >= max_hits:
                return seen
    return seen


def foldseek_easy_search(
    query_pdb: Path,
    target_db: Path,
    *,
    foldseek_bin: str | None = None,
    use_gpu: bool = True,
    extra_args: Sequence[str] | None = None,
) -> str:
    """Run ``foldseek easy-search``; return stdout (tabular)."""
    exe = foldseek_bin or os.environ.get("FOLDSEEK_BIN", "foldseek")
    with tempfile.TemporaryDirectory(prefix="foldseek_sk_") as tmp:
        tmp_p = Path(tmp)
        m8_path = tmp_p / "hits.m8"
        tmp_dir = tmp_p / "tmp"
        tmp_dir.mkdir()
        cmd = [
            exe,
            "easy-search",
            str(query_pdb),
            str(target_db),
            str(m8_path),
            str(tmp_dir),
        ]
        if use_gpu:
            cmd.extend(["--gpu", "1"])
        if extra_args:
            cmd.extend(list(extra_args))
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"foldseek easy-search failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        if m8_path.is_file():
            return m8_path.read_text()
        for c in sorted(tmp_p.glob("*.m8")):
            if c.is_file() and c.stat().st_size > 0:
                return c.read_text()
        return proc.stdout or ""


# --- Orchestration -----------------------------------------------------------------


@dataclass
class IncrementalSkrinjarScorer:
    """Score a query complex PDB against training structures / ligands."""

    foldseek_bin: str | None = None
    foldseek_db: Path | None = None
    training_complexes_dir: Path | None = None
    training_ligand_sdfs: tuple[Path, ...] = ()
    use_foldseek_gpu: bool = True
    top_k: int = 5
    pocket_radius: float = 6.0
    pocket_qcov_cutoff: float = 2.0
    enable_coord_cache: bool = True
    scratch_dir: Path | None = None
    align_mol_fallback_count: int = field(init=False, default=0)
    _cache: dict[str, float] = field(default_factory=dict, repr=False)

    def _bump_align_fallback(self) -> None:
        self.align_mol_fallback_count += 1

    def _coord_key(self, coords: np.ndarray) -> str:
        arr = np.asarray(coords, dtype=np.float32).ravel(order="C")
        return hashlib.md5(arr.tobytes()).hexdigest()

    def score_pdb_file(self, query_pdb: Path) -> float:
        """Best ``sucos_shape * pocket_qcov`` over configured training neighbors."""
        key = hashlib.md5(query_pdb.read_bytes()).hexdigest()
        if self.enable_coord_cache and key in self._cache:
            return self._cache[key]
        best = self._score_pdb_uncached(query_pdb)
        if self.enable_coord_cache:
            self._cache[key] = best
        return best

    def score_coords(
        self,
        coords_angstrom: np.ndarray,
        structure: Any,
        n_struct: int,
        *,
        workdir: Path | None = None,
    ) -> float:
        """Write coordinates to a temp PDB via Boltz topology, then score."""
        from dataclasses import replace  # noqa: PLC0415

        from boltz.data.types import Coords, Interface  # noqa: PLC0415
        from boltz.data.write.pdb import to_pdb  # noqa: PLC0415

        c = np.asarray(coords_angstrom, dtype=np.float32)[: int(n_struct)]
        if self.enable_coord_cache:
            ck = self._coord_key(c)
            if ck in self._cache:
                return self._cache[ck]

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
        root = workdir or self.scratch_dir
        if root is None:
            root = Path(tempfile.mkdtemp(prefix="skrinjar_cv_"))
        else:
            root.mkdir(parents=True, exist_ok=True)
        qpath = root / "query.pdb"
        qpath.write_text(to_pdb(new_s, plddts=None, boltz2=True), encoding="utf-8")
        val = self._score_pdb_uncached(qpath)
        if self.enable_coord_cache:
            self._cache[self._coord_key(c)] = val
        return val

    def _training_complex_candidates(self, query_pdb: Path) -> list[Path]:
        hits: list[Path] = []
        if self.training_complexes_dir is None:
            return hits
        d = self.training_complexes_dir
        if self.foldseek_db is not None and self.foldseek_db.exists():
            try:
                out = foldseek_easy_search(
                    query_pdb,
                    self.foldseek_db,
                    foldseek_bin=self.foldseek_bin,
                    use_gpu=self.use_foldseek_gpu,
                )
                pdb_ids = parse_pdb_ids_from_foldseek_output(out, self.top_k)
            except (OSError, RuntimeError):
                pdb_ids = []
            for pid in pdb_ids:
                for pat in (f"{pid}.pdb", f"{pid.upper()}.pdb", f"{pid.lower()}.pdb"):
                    p = d / pat
                    if p.is_file():
                        hits.append(p)
                        break
        if not hits:
            hits = sorted(p for p in d.glob("*.pdb") if p.is_file())[: max(1, self.top_k * 10)]
        return hits[: self.top_k] if hits else []

    def _score_pdb_uncached(self, query_pdb: Path) -> float:
        q_mol = ligand_mol_from_pdb(query_pdb)
        if q_mol is None:
            return 0.0

        best = 0.0
        bump = self._bump_align_fallback
        for tpath in self._training_complex_candidates(query_pdb):
            t_mol = ligand_mol_from_pdb(tpath)
            if t_mol is None:
                continue
            try:
                su = sucos_shape_after_align(q_mol, t_mol, on_align_fallback=bump)
                pq = geometric_pocket_qcov_ca(
                    query_pdb,
                    tpath,
                    pocket_radius=self.pocket_radius,
                    match_cutoff=self.pocket_qcov_cutoff,
                )
                best = max(best, su * pq)
            except (ValueError, OSError, RuntimeError) as exc:
                _LOG.debug("skrinjar pair skip (%s): %s", type(exc).__name__, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                _LOG.debug(
                    "skrinjar pair skip (unexpected %s): %s",
                    type(exc).__name__,
                    exc,
                )
                continue

        for sdf in self.training_ligand_sdfs:
            if not sdf.is_file():
                continue
            try:
                t_mol = mol_from_sdf_path(sdf)
                if t_mol is None:
                    continue
                su = sucos_shape_after_align(q_mol, t_mol, on_align_fallback=bump)
                best = max(best, su)
            except (ValueError, OSError, RuntimeError) as exc:
                _LOG.debug("skrinjar sdf skip (%s): %s", type(exc).__name__, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                _LOG.debug(
                    "skrinjar sdf skip (unexpected %s): %s",
                    type(exc).__name__,
                    exc,
                )
                continue

        return float(best)

    def clear_cache(self) -> None:
        self._cache.clear()


def load_parquet_similarity_column(
    parquet_path: Path,
    column: str,
    row_index: int = 0,
) -> float | None:
    """Read one scalar from a parquet (optional regression helper)."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        return None
    try:
        table = pq.read_table(parquet_path, columns=[column])
    except Exception:  # noqa: BLE001 — missing column or corrupt file
        return None
    if table.num_rows == 0 or row_index >= table.num_rows:
        return None
    try:
        val = table.column(0)[row_index].as_py()
    except Exception:  # noqa: BLE001
        return None
    if val is None:
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN -> None

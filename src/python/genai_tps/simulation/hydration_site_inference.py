"""Infer OneOPES auxiliary hydration spot indices (Boltz global) without a prior MD run.

Two strategies are supported:

1. **Geometric (default, always available)** — rank N/O atoms on the ligand and
   pocket (from :class:`~genai_tps.backends.boltz.cv_pose.PoseCVIndexer`) by
   proximity across the binding interface and take the top *K* candidates.
   This matches the plan's fallback when no solvent-density map is available.

2. **3D-RISM + metatwist (optional)** — if AmberTools ``tleap`` can build an
   AMBER topology from the initial PDB and ``rism3d.snglpnt`` + ``metatwist``
   succeed, high-density water oxygen blobs are converted to spot anchors by
   mapping each blob centroid to the nearest ligand/pocket N/O atom.

The PLUMED deck expects ``oneopes_hydration_spot_plumed_idx`` to be **solute**
atom indices around which water oxygens are coordinated (not water atoms
themselves).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np

__all__ = [
    "default_oneopes_hydration_boltz_indices",
    "geometric_hydration_spot_boltz_indices",
    "map_water_centroids_to_boltz_no_indices",
    "try_infer_hydration_site_coords_3drism",
]


def _distances_to_set(
    ref: np.ndarray, query_idx: np.ndarray, target_idx: np.ndarray
) -> np.ndarray:
    """Minimum distance from each query atom to any target atom (Å)."""
    if query_idx.size == 0 or target_idx.size == 0:
        return np.full(query_idx.shape[0], np.inf, dtype=np.float64)
    q = ref[np.asarray(query_idx, dtype=np.int64)]
    t = ref[np.asarray(target_idx, dtype=np.int64)]
    # (nq, nt, 3) -> (nq, nt)
    d = np.linalg.norm(q[:, None, :] - t[None, :, :], axis=2)
    return d.min(axis=1)


def geometric_hydration_spot_boltz_indices(
    ref_coords_angstrom: np.ndarray,
    indexer: "PoseCVIndexer",
    *,
    max_sites: int = 5,
    interface_cutoff_angstrom: float = 4.5,
) -> list[int]:
    """Pick Boltz indices of ligand/pocket N/O atoms at the binding interface.

    Candidates are ``indexer.ligand_no_idx`` and ``indexer.pocket_no_idx``.
    Atoms within *interface_cutoff_angstrom* of the opposite moiety are ranked
    by ``d_lig + d_pocket`` (smaller = more interfacial) and de-duplicated.
    """
    ref = np.asarray(ref_coords_angstrom, dtype=np.float64)
    max_sites = int(max_sites)
    if max_sites <= 0:
        return []

    lig_all = np.asarray(indexer.ligand_idx, dtype=np.int64)
    pocket_heavy = np.asarray(indexer.pocket_heavy_idx, dtype=np.int64)
    lig_no = np.asarray(indexer.ligand_no_idx, dtype=np.int64)
    pocket_no = np.asarray(indexer.pocket_no_idx, dtype=np.int64)

    scored: list[tuple[float, int]] = []

    for idx in lig_no.tolist():
        d_poc = float(
            np.min(np.linalg.norm(ref[idx] - ref[pocket_heavy], axis=1))
            if pocket_heavy.size
            else np.inf
        )
        if d_poc <= float(interface_cutoff_angstrom):
            d_lig = 0.0  # on ligand
            scored.append((d_lig + d_poc, int(idx)))

    for idx in pocket_no.tolist():
        d_lig = float(
            np.min(np.linalg.norm(ref[idx] - ref[lig_all], axis=1))
            if lig_all.size
            else np.inf
        )
        if d_lig <= float(interface_cutoff_angstrom):
            d_poc = 0.0
            scored.append((d_lig + d_poc, int(idx)))

    scored.sort(key=lambda x: x[0])
    out: list[int] = []
    seen: set[int] = set()
    for _, i in scored:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
        if len(out) >= max_sites:
            break

    # If interface filter removed everyone, fall back to closest N/O pairs
    if not out and (lig_no.size or pocket_no.size):
        extras: list[tuple[float, int]] = []
        if lig_no.size and pocket_heavy.size:
            dl = _distances_to_set(ref, lig_no, pocket_heavy)
            for j, idx in enumerate(lig_no.tolist()):
                extras.append((float(dl[j]), int(idx)))
        if pocket_no.size and lig_all.size:
            dp = _distances_to_set(ref, pocket_no, lig_all)
            for j, idx in enumerate(pocket_no.tolist()):
                extras.append((float(dp[j]), int(idx)))
        extras.sort(key=lambda x: x[0])
        for _, i in extras[:max_sites]:
            if i not in seen:
                out.append(i)
                seen.add(i)
    return out


def map_water_centroids_to_boltz_no_indices(
    centroids_angstrom: np.ndarray,
    ref_coords_angstrom: np.ndarray,
    indexer: "PoseCVIndexer",
    *,
    max_sites: int = 5,
    max_anchor_distance_angstrom: float = 5.0,
) -> list[int]:
    """Map 3D water-blob centroids to nearest ligand/pocket N/O Boltz indices."""
    ref = np.asarray(ref_coords_angstrom, dtype=np.float64)
    cents = np.asarray(centroids_angstrom, dtype=np.float64).reshape(-1, 3)
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray(indexer.ligand_no_idx, dtype=np.int64),
                np.asarray(indexer.pocket_no_idx, dtype=np.int64),
            ]
        )
    )
    if candidates.size == 0 or cents.shape[0] == 0:
        return []

    picked: list[int] = []
    seen: set[int] = set()
    for c in cents:
        d = np.linalg.norm(ref[candidates] - c.reshape(1, 3), axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > float(max_anchor_distance_angstrom):
            continue
        atom = int(candidates[j])
        if atom in seen:
            continue
        seen.add(atom)
        picked.append(atom)
        if len(picked) >= int(max_sites):
            break
    return picked


def _default_rism_xvv_path() -> Path | None:
    """Return packaged SPC/E (CHARMM-like) susceptibility file if present."""
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("AMBERHOME")
    if not prefix:
        return None
    p = Path(prefix) / "dat" / "rism1d" / "cSPCE" / "cSPCE_kh.xvv"
    return p if p.is_file() else None


def try_infer_hydration_site_coords_3drism(
    pdb_path: Path,
    *,
    min_density: float = 2.0,
    rism3d_executable: str = "rism3d.snglpnt",
    tleap_executable: str = "tleap",
    metatwist_executable: str = "metatwist",
) -> np.ndarray | None:
    """Run 3D-RISM + metatwist blob centroids; return an (N, 3) array of O sites (Å) or None.

    This is **best-effort**: many protein–ligand PDBs fail ``tleap`` without a
    bespoke ligand force field. On any failure, returns ``None`` so callers can
    fall back to :func:`geometric_hydration_spot_boltz_indices`.
    """
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.is_file():
        return None
    xvv = _default_rism_xvv_path()
    if xvv is None:
        return None
    for exe in (rism3d_executable, tleap_executable, metatwist_executable):
        if shutil.which(exe) is None:
            return None

    work = Path(tempfile.mkdtemp(prefix="genai_tps_rism_"))
    try:
        complex_pdb = work / "complex.pdb"
        complex_pdb.write_bytes(pdb_path.read_bytes())
        leap_in = work / "tleap.in"
        leap_in.write_text(
            "source leaprc.protein.ff19SB\n"
            f"m = loadpdb {complex_pdb.name}\n"
            "saveamberparm m complex.prmtop complex.rst7\n"
            "quit\n",
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [tleap_executable, "-f", str(leap_in)],
                cwd=str(work),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        prmtop = work / "complex.prmtop"
        if not prmtop.is_file():
            return None

        guv_root = work / "guv"
        try:
            subprocess.run(
                [
                    rism3d_executable,
                    "--pdb",
                    str(complex_pdb),
                    "--prmtop",
                    str(prmtop),
                    "--rst",
                    str(work / "complex.rst7"),
                    "--xvv",
                    str(xvv),
                    "--guv",
                    str(guv_root),
                    "--closure",
                    "kh",
                    "--buffer",
                    "20.0",
                    "--grdspc",
                    "0.5,0.5,0.5",
                    "--tolerance",
                    "1e-3",
                ],
                cwd=str(work),
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        # Oxygen site distribution (AmberTools naming: PREFIX.SITE.FRAME.dx)
        dx_candidates = sorted(work.glob("guv.O.*.dx"))
        if not dx_candidates:
            dx_candidates = sorted(work.glob("*.O.*.dx"))
        if not dx_candidates:
            return None
        dx_in = dx_candidates[0]
        lap_out = work / "laplace.dx"
        try:
            subprocess.run(
                [
                    metatwist_executable,
                    "--dx",
                    str(dx_in),
                    "--species",
                    "O",
                    "--convolve",
                    "4",
                    "--sigma",
                    "1.0",
                    "--odx",
                    str(lap_out),
                ],
                cwd=str(work),
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        blob_pdb = work / "blobs.pdb"
        try:
            subprocess.run(
                [
                    metatwist_executable,
                    "--dx",
                    str(dx_in),
                    "--ldx",
                    str(lap_out),
                    "--map",
                    "blobsper",
                    "--species",
                    "O",
                    "WAT",
                    "--bulk",
                    "55.55",
                    "--threshold",
                    str(float(min_density)),
                ],
                cwd=str(work),
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        # metatwist writes a centroid PDB with a predictable suffix pattern
        centroid_pdbs = sorted(work.glob("*blobs-centroid*.pdb"))
        if not centroid_pdbs:
            centroid_pdbs = sorted(work.glob("*centroid*.pdb"))
        if not centroid_pdbs:
            return None

        coords = _read_oxygen_coords_from_pdb(centroid_pdbs[0])
        return coords if coords.size else None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _read_oxygen_coords_from_pdb(path: Path) -> np.ndarray:
    """Parse ATOM/HETATM lines with element O from a PDB."""
    xs: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            elem = line[76:78].strip() if len(line) > 76 else ""
            if not elem:
                name = line[12:16].strip().upper()
                elem = name[:1] if name else ""
            if elem != "O":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            xs.append([x, y, z])
    return np.asarray(xs, dtype=np.float64)


def default_oneopes_hydration_boltz_indices(
    ref_coords_angstrom: np.ndarray,
    indexer: "PoseCVIndexer",
    *,
    pdb_path: Path | None,
    max_sites: int = 5,
    use_3drism: bool = True,
    rism_min_density: float = 2.0,
) -> list[int]:
    """Return Boltz indices for OneOPES hydration spots (interface N/O by default).

    If *use_3drism* is true and *pdb_path* is set, try 3D-RISM blob centroids
    first; otherwise (or on failure) use :func:`geometric_hydration_spot_boltz_indices`.
    """
    ref = np.asarray(ref_coords_angstrom, dtype=np.float64)
    if use_3drism and pdb_path is not None:
        cents = try_infer_hydration_site_coords_3drism(
            pdb_path, min_density=float(rism_min_density)
        )
        if cents is not None and cents.size:
            mapped = map_water_centroids_to_boltz_no_indices(
                cents,
                ref,
                indexer,
                max_sites=int(max_sites),
            )
            if mapped:
                return mapped
    return geometric_hydration_spot_boltz_indices(
        ref,
        indexer,
        max_sites=int(max_sites),
    )

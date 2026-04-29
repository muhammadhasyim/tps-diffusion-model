"""Boltz NPZ → single-frame PDB export helpers.

This module centralises the logic for converting Boltz checkpoint NPZ files
(``tps_mc_step_*.npz``) into single-frame PDB files so that downstream tools
(OpenMM RMSD watcher, ProDy ensemble builder, etc.) share a single
implementation.

The only heavy dependency is ``boltz`` (available when the Boltz submodule is
on ``sys.path``).  NumPy and standard-library imports are the only hard
requirements at module level.

Usage::

    from genai_tps.io.boltz_npz_export import load_topo, npz_to_pdb, batch_export

    topo, n_struct = load_topo(Path("boltz_results/.../structures/foo.npz"))

    # write a single last-frame PDB
    npz_to_pdb(Path("tps_mc_step_00001000.npz"), topo, n_struct,
               Path("out/step_01000_last.pdb"))

    # batch-convert a whole directory
    pdb_paths = batch_export(
        ckpt_dir=Path("cofolding_tps_out/trajectory_checkpoints"),
        topo_npz=Path("..."),
        out_dir=Path("out/pdbs"),
    )
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np


def _pdb_coord_field_width_ok(value: float) -> bool:
    """Return True if *value* fits Boltz ``to_pdb``'s ``%8.3f`` coordinate columns."""
    return len(f"{float(value):8.3f}") <= 8


def coords_centered_for_pdb_export(
    fc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the centroid so coordinates fit PDB fixed-width ``8.3f`` fields.

    Boltz's :func:`boltz.data.write.pdb.to_pdb` formats each Cartesian component
    with ``%8.3f`` (width 8).  Values outside roughly ``[-999.999, 9999.999]`` produce
    strings longer than eight characters and corrupt adjacent PDB columns, which
    breaks :class:`openmm.app.PDBFile` parsing.

    Translating all atoms by the same vector preserves internal distances and is
    safe for implicit-solvent OpenMM pipelines that depend only on relative
    coordinates.

    Parameters
    ----------
    fc
        Cartesian coordinates in ångströms, shape ``(N, 3)``.

    Returns
    -------
    fc_shifted, com
        *fc_shifted* is ``float32`` ``(N, 3)`` suitable for ``to_pdb``.  *com* is
        ``float64`` ``(3,)``, the subtracted centroid (documented in a REMARK).

    Raises
    ------
    ValueError
        If coordinates still do not fit after centering (pathological / NaN pose).
    """
    fc64 = np.asarray(fc, dtype=np.float64)
    if fc64.ndim != 2 or fc64.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinates, got shape {fc64.shape}")
    if not np.isfinite(fc64).all():
        raise ValueError("Non-finite coordinates in frame; cannot write PDB.")
    com = fc64.mean(axis=0)
    centered = (fc64 - com).astype(np.float32)
    bad = [float(v) for v in centered.ravel() if not _pdb_coord_field_width_ok(float(v))]
    if bad:
        r = float(np.nanmax(np.abs(centered)))
        raise ValueError(
            "Coordinates do not fit PDB 8.3f columns even after COM centering "
            f"(max|x|≈{r:.1f} Å). Check Boltz sampling quality (official Boltz CLI "
            "defaults: --sampling_steps 200, --recycling_steps 3; see boltz "
            "docs/prediction.md)."
        )
    return centered, com


def coords_frame_from_npz(npz, *, frame_idx: int, n_struct: int) -> np.ndarray:
    """Extract one frame of Cartesian coordinates as ``(n_struct, 3)`` float32.

    Supports TPS checkpoint NPZs, Boltz ``processed/structures/*.npz`` topology
    bundles, and plain single-snapshot layouts:

    * Trajectory: ``coords`` or ``atom_coords`` with shape ``(T, N, 3)``.
      ``frame_idx`` follows NumPy negative indexing (``-1`` = last frame).
    * Boltz :class:`boltz.data.types.Coords` on disk: ``coords`` is structured
      ``(N,)`` with field ``coords`` of shape ``(3,)``.
    * Snapshot: ``coords`` with shape ``(N, 3)``.

    Parameters
    ----------
    npz:
        Object returned by :func:`numpy.load` on an ``.npz`` file.
    frame_idx:
        Time index for trajectory layouts. Ignored for single-structure NPZs.
    n_struct:
        Number of heavy atoms to keep (padding is dropped when present).

    Returns
    -------
    numpy.ndarray
        ``(min(N, n_struct), 3)``, dtype ``float32``.
    """
    files = npz.files
    if "coords" in files:
        key = "coords"
    elif "atom_coords" in files:
        key = "atom_coords"
    else:
        raise KeyError(
            "NPZ has neither 'coords' nor 'atom_coords'; "
            f"keys={sorted(files)}"
        )

    raw = npz[key]
    n_keep = int(n_struct)

    if raw.ndim == 3:
        t_frames = raw.shape[0]
        idx = frame_idx if frame_idx >= 0 else t_frames + frame_idx
        fc = np.asarray(raw[int(idx)], dtype=np.float32)[:n_keep]
        return fc

    if raw.ndim == 1 and raw.dtype.names and "coords" in raw.dtype.names:
        fc = np.asarray(raw["coords"], dtype=np.float32)[:n_keep]
        return fc

    if raw.ndim == 2 and raw.shape[1] == 3:
        fc = np.asarray(raw, dtype=np.float32)[:n_keep]
        return fc

    raise ValueError(
        f"Unsupported {key!r} layout: shape={raw.shape}, dtype={raw.dtype}"
    )


def load_topo(topo_npz: Path):
    """Load Boltz ``StructureV2`` topology and return ``(structure, n_atoms)``.

    Parameters
    ----------
    topo_npz:
        Path to the Boltz ``processed/structures/*.npz`` file.

    Returns
    -------
    tuple[StructureV2, int]
        The topology object and the number of heavy atoms it describes.

    Raises
    ------
    ImportError
        When the ``boltz`` package is not importable (Boltz submodule absent).
    FileNotFoundError
        When *topo_npz* does not exist.
    """
    from boltz.data.types import StructureV2  # noqa: PLC0415

    if not topo_npz.is_file():
        raise FileNotFoundError(f"Topology NPZ not found: {topo_npz}")
    structure = StructureV2.load(topo_npz).remove_invalid_chains()
    n_struct = int(structure.atoms.shape[0])
    return structure, n_struct


def npz_to_pdb(
    ckpt_npz: Path,
    topo_structure,
    n_struct: int,
    out_pdb: Path,
    *,
    frame_idx: int = -1,
) -> None:
    """Extract one diffusion frame from *ckpt_npz* and write a PDB file.

    Parameters
    ----------
    ckpt_npz:
        Path to a ``tps_mc_step_*.npz`` checkpoint file.  The ``coords``
        array has shape ``(n_diffusion_frames, n_atoms_padded, 3)`` in
        Angstroms.
    topo_structure:
        Boltz ``StructureV2`` topology object (from :func:`load_topo`).
    n_struct:
        Number of heavy atoms described by *topo_structure*.  Coordinates
        beyond index *n_struct* are padding and are discarded.
    out_pdb:
        Destination PDB file.  Parent directories are created automatically.
    frame_idx:
        Which diffusion frame to export.  ``-1`` (default) selects the last
        frame, i.e. the fully denoised terminal structure.
    """
    from boltz.data.types import Coords, Interface  # noqa: PLC0415
    from boltz.data.write.pdb import to_pdb  # noqa: PLC0415

    with np.load(ckpt_npz) as d:
        fc = coords_frame_from_npz(d, frame_idx=frame_idx, n_struct=n_struct)

    fc, com_shift_angstrom = coords_centered_for_pdb_export(fc)

    atoms = topo_structure.atoms.copy()
    atoms["coords"] = fc
    atoms["is_present"] = True
    residues = topo_structure.residues.copy()
    residues["is_present"] = True
    coord_arr = np.array([(x,) for x in fc], dtype=Coords)
    interfaces = np.array([], dtype=Interface)

    new_s = replace(
        topo_structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_arr,
    )
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    body = to_pdb(new_s, plddts=None, boltz2=True)
    remark = (
        "REMARK 999 GENAI_TPS PDB export: subtracted COM (Angstrom) "
        f"{com_shift_angstrom[0]:.6f} {com_shift_angstrom[1]:.6f} "
        f"{com_shift_angstrom[2]:.6f} so coordinates fit fixed-width fields.\n"
    )
    out_pdb.write_text(remark + body)


def batch_export(
    ckpt_dir: Path,
    topo_npz: Path,
    out_dir: Path,
    *,
    frame_idx: int = -1,
    glob: str = "tps_mc_step_*.npz",
    skip_patterns: Tuple[str, ...] = ("last_frame", "latest"),
    max_files: int | None = None,
) -> list[Path]:
    """Convert every checkpoint NPZ in *ckpt_dir* to a last-frame PDB.

    Parameters
    ----------
    ckpt_dir:
        Directory containing ``tps_mc_step_*.npz`` checkpoint files.
    topo_npz:
        Path to Boltz ``processed/structures/*.npz`` for topology.
    out_dir:
        Directory where PDB files will be written (created if absent).
    frame_idx:
        Diffusion frame to export from each NPZ; ``-1`` = last (terminal).
    glob:
        Glob pattern for checkpoint files within *ckpt_dir*.
    skip_patterns:
        NPZ stems containing any of these substrings are skipped.
    max_files:
        If set and positive, only process the first *max_files* checkpoints (after
        sort and skip filter).

    Returns
    -------
    list[Path]
        Sorted list of written PDB paths.
    """
    topo, n_struct = load_topo(topo_npz)
    ckpts = sorted(
        p for p in ckpt_dir.glob(glob)
        if not any(pat in p.stem for pat in skip_patterns)
    )
    if max_files is not None and max_files > 0:
        ckpts = ckpts[:max_files]
    written: list[Path] = []
    for ckpt in ckpts:
        out_pdb = out_dir / (ckpt.stem + "_last.pdb")
        npz_to_pdb(ckpt, topo, n_struct, out_pdb, frame_idx=frame_idx)
        written.append(out_pdb)
    return written

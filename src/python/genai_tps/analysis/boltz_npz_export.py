"""Boltz NPZ → single-frame PDB export helpers.

This module centralises the logic for converting Boltz checkpoint NPZ files
(``tps_mc_step_*.npz``) into single-frame PDB files so that downstream tools
(OpenMM RMSD watcher, ProDy ensemble builder, etc.) share a single
implementation.

The only heavy dependency is ``boltz`` (available when the Boltz submodule is
on ``sys.path``).  NumPy and standard-library imports are the only hard
requirements at module level.

Usage::

    from genai_tps.analysis.boltz_npz_export import load_topo, npz_to_pdb, batch_export

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

    d = np.load(ckpt_npz)
    fc = np.asarray(d["coords"][frame_idx], dtype=np.float32)[:n_struct]

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
    out_pdb.write_text(to_pdb(new_s, plddts=None, boltz2=True))


def batch_export(
    ckpt_dir: Path,
    topo_npz: Path,
    out_dir: Path,
    *,
    frame_idx: int = -1,
    glob: str = "tps_mc_step_*.npz",
    skip_patterns: Tuple[str, ...] = ("last_frame", "latest"),
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
    written: list[Path] = []
    for ckpt in ckpts:
        out_pdb = out_dir / (ckpt.stem + "_last.pdb")
        npz_to_pdb(ckpt, topo, n_struct, out_pdb, frame_idx=frame_idx)
        written.append(out_pdb)
    return written

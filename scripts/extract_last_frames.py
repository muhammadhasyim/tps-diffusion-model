#!/usr/bin/env python3
"""Extract the last frame from each TPS trajectory checkpoint and write PDB files.

Reads every ``tps_mc_step_*.npz`` in ``--checkpoint-dir``, takes the final
frame of each accepted trajectory (``coords[-1]``), and writes it as a PDB
file using the Boltz topology.  The resulting PDB files can be passed directly
to ``compute_cv_rmsd.py`` for energy-minimisation and RMSD analysis.

Usage::

    python scripts/extract_last_frames.py \\
        --checkpoint-dir opes_tps_out/trajectory_checkpoints \\
        --topo-npz opes_tps_out/boltz_results_cofolding_multimer_msa_empty/processed/structures/cofolding_multimer_msa_empty.npz \\
        --out-dir opes_tps_out/checkpoint_last_frames
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_topo(topo_npz: Path):
    """Load Boltz topology from a processed structures npz file.

    Parameters
    ----------
    topo_npz:
        Path to the ``processed/structures/*.npz`` file.

    Returns
    -------
    tuple[Structure, int]
        Boltz Structure object and the number of atoms.
    """
    from genai_tps.analysis.boltz_npz_export import load_topo as _load_topo
    return _load_topo(topo_npz)


def coords_to_pdb_string(topo, n_struct: int, coords_angstrom: np.ndarray) -> str:
    """Convert a ``(n_atoms, 3)`` Å coordinate array to a PDB string.

    Parameters
    ----------
    topo:
        Boltz Structure object (topology).
    n_struct:
        Number of atoms in the structure.
    coords_angstrom:
        Shape ``(n_atoms, 3)`` in Ångström.

    Returns
    -------
    str
        PDB-format string.
    """
    from boltz.data.types import Coords, Interface
    from boltz.data.write.pdb import to_pdb
    from dataclasses import replace

    fc = np.asarray(coords_angstrom[:n_struct], dtype=np.float32)

    atoms = topo.atoms.copy()
    atoms["coords"] = fc
    atoms["is_present"] = True
    residues = topo.residues.copy()
    residues["is_present"] = True
    coord_arr = np.array([(x,) for x in fc], dtype=Coords)
    interfaces = np.array([], dtype=Interface)

    new_s = replace(
        topo,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coord_arr,
    )
    return to_pdb(new_s, plddts=None, boltz2=True)


def extract_last_frames(
    checkpoint_dir: Path,
    topo_npz: Path,
    out_dir: Path,
    glob: str = "tps_mc_step_*.npz",
) -> int:
    """Extract last frames from all checkpoint files and write PDBs.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing ``tps_mc_step_*.npz`` files.
    topo_npz:
        Path to the Boltz topology npz.
    out_dir:
        Output directory for PDB files.
    glob:
        Filename pattern to match within ``checkpoint_dir``.

    Returns
    -------
    int
        Number of PDB files written.
    """
    checkpoints = sorted(checkpoint_dir.glob(glob))
    if not checkpoints:
        logger.error("No files matching '%s' in %s", glob, checkpoint_dir)
        return 0

    logger.info("Loading topology from %s …", topo_npz)
    topo, n_struct = load_topo(topo_npz)
    logger.info("Topology: %d atoms", n_struct)

    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ckpt_path in checkpoints:
        data = np.load(ckpt_path, allow_pickle=True)
        coords = data["coords"]  # (n_frames, n_atoms, 3)
        mc_step = int(data["mc_step"])

        last_frame = coords[-1]  # (n_atoms, 3)

        pdb_str = coords_to_pdb_string(topo, n_struct, last_frame)

        stem = f"step_{mc_step:08d}_last_frame"
        out_path = out_dir / f"{stem}.pdb"
        out_path.write_text(pdb_str)
        written += 1
        logger.info("  %s → %s  (%d frames, wrote last)", ckpt_path.name, out_path.name, len(coords))

    logger.info("Wrote %d PDB files to %s", written, out_dir)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing tps_mc_step_*.npz checkpoint files.",
    )
    p.add_argument(
        "--topo-npz",
        type=Path,
        required=True,
        help="Path to the Boltz processed/structures/*.npz topology file.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for extracted PDB files.",
    )
    p.add_argument(
        "--glob",
        default="tps_mc_step_*.npz",
        metavar="PATTERN",
        help="Glob pattern for checkpoint files (default: 'tps_mc_step_*.npz').",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    args = parse_args(argv)
    n = extract_last_frames(
        checkpoint_dir=args.checkpoint_dir,
        topo_npz=args.topo_npz,
        out_dir=args.out_dir,
        glob=args.glob,
    )
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

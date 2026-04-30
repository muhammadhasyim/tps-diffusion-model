"""Helpers for real OPES-MD through ``openmm-plumed``.

This module builds the PLUMED deck used by the OpenMM MD runner and adds the
resulting :class:`openmmplumed.PlumedForce` to an OpenMM ``System``.  The CVs
mirror the existing Python diagnostics:

* ligand pose RMSD after optimal alignment on pocket C-alpha atoms;
* ligand-to-pocket geometric-center distance.

All atom indices passed to this module must already be PLUMED indices, i.e.
1-based OpenMM particle indices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "add_plumed_opes_to_system",
    "generate_plumed_opes_script",
    "write_rmsd_reference_pdb",
]


def _format_float(value: float) -> str:
    """Return a compact deterministic decimal representation for PLUMED input."""
    return f"{float(value):.12g}"


def _plumed_kbt_kjmol(temperature_k: float) -> float:
    """Return *k* B T in kJ/mol, matching :mod:`genai_tps.simulation.openmm_md_runner`.

    Uses the gas constant *R* = 8.314e-3 kJ/(mol·K) times absolute temperature.
    """
    return float(8.314e-3 * float(temperature_k))


def _opes_metad_default_kernel_cutoff_sigma(
    barrier_kjmol: float, biasfactor: float, temperature_k: float
) -> float:
    """PLUMED ``OPES_METAD`` default ``KERNEL_CUTOFF`` in Gaussian sigma units.

    Mirrors ``plumed2/src/opes/OPESmetad.cpp`` for well-tempered OPES:
    ``sqrt(2 * BARRIER / (1 - 1/gamma) / kbt)`` with ``gamma = BIASFACTOR``.
    """
    gamma = float(biasfactor)
    if gamma <= 1.0:
        raise ValueError("OPES BIASFACTOR must be greater than 1.")
    kbt = _plumed_kbt_kjmol(temperature_k)
    if kbt <= 0.0:
        raise ValueError("Temperature must be positive for OPES kernel cutoff.")
    bias_pref = 1.0 - 1.0 / gamma
    return math.sqrt(2.0 * float(barrier_kjmol) / bias_pref / kbt)


def _format_indices(indices: Sequence[int]) -> str:
    """Format a non-empty 1-based atom-index list for PLUMED ``ATOMS=`` fields."""
    values = [int(i) for i in indices]
    if not values:
        raise ValueError("PLUMED atom-index list must not be empty.")
    if min(values) < 1:
        raise ValueError("PLUMED atom indices must be 1-based positive integers.")
    return ",".join(str(i) for i in values)


def _positions_to_angstrom(positions: Any) -> np.ndarray:
    """Convert OpenMM positions to an ``(N, 3)`` float64 array in Angstrom."""
    try:
        import openmm.unit as unit

        arr = positions.value_in_unit(unit.angstrom)
    except AttributeError:
        arr = np.asarray(positions, dtype=np.float64)
        # OpenMM's raw Vec3 positions are conventionally in nm.
        arr = arr * 10.0
    return np.asarray(arr, dtype=np.float64)


def _residue_sequence_number(residue: Any) -> int:
    """Return a PDB-compatible residue sequence number from an OpenMM residue."""
    rid = residue.id
    if isinstance(rid, tuple):
        return int(rid[0])
    try:
        return int(rid)
    except (TypeError, ValueError):
        return int(residue.index) + 1


def _format_pdb_atom_name(name: str, element_symbol: str) -> str:
    """Format a PDB atom name field using common left/right alignment rules."""
    name = name[:4]
    if len(name) < 4 and len(element_symbol) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def write_rmsd_reference_pdb(
    topology: Any,
    positions: Any,
    ligand_plumed_idx: Sequence[int],
    align_plumed_idx: Sequence[int],
    out_path: Path,
) -> Path:
    """Write a PLUMED RMSD reference PDB with alignment/displacement weights.

    Parameters
    ----------
    topology:
        OpenMM topology corresponding to the running simulation.
    positions:
        OpenMM positions for *topology*.  Quantity values are converted to
        Angstrom; plain arrays are assumed to be in nanometers.
    ligand_plumed_idx:
        PLUMED 1-based atom indices whose displacement defines the ligand RMSD.
    align_plumed_idx:
        PLUMED 1-based atom indices used for optimal alignment.  For the TPS
        diagnostic systems this is the pocket C-alpha set, falling back to all
        protein C-alpha atoms when the pocket set is too small.
    out_path:
        Destination PDB path.

    Returns
    -------
    Path
        The resolved path that was written.

    Notes
    -----
    PLUMED's ``RMSD TYPE=OPTIMAL`` uses the reference PDB occupancy column as
    the alignment weight and the beta column as the displacement weight.
    """
    ligand_set = {int(i) for i in ligand_plumed_idx}
    align_set = {int(i) for i in align_plumed_idx}
    if not ligand_set:
        raise ValueError("Cannot write PLUMED RMSD reference without ligand atoms.")
    if len(align_set) < 3:
        raise ValueError(
            "PLUMED RMSD TYPE=OPTIMAL requires at least three alignment atoms."
        )

    coords = _positions_to_angstrom(positions)
    out = out_path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as handle:
        for atom in topology.atoms():
            plumed_idx = int(atom.index) + 1
            if plumed_idx not in ligand_set and plumed_idx not in align_set:
                continue

            xyz = coords[int(atom.index)]
            residue = atom.residue
            chain_id = (residue.chain.id or "A").strip()[:1] or "A"
            res_name = residue.name.strip()[:3] or "UNK"
            res_seq = _residue_sequence_number(residue)
            element_symbol = atom.element.symbol if atom.element is not None else atom.name[:1]
            atom_name = _format_pdb_atom_name(atom.name.strip(), element_symbol)
            occupancy = 1.0 if plumed_idx in align_set else 0.0
            beta = 1.0 if plumed_idx in ligand_set else 0.0
            record = "HETATM" if residue.name.strip() not in {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"} else "ATOM  "

            handle.write(
                f"{record}{plumed_idx:5d} {atom_name}{res_name:>4} {chain_id}"
                f"{res_seq:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"{occupancy:6.2f}{beta:6.2f}          {element_symbol:>2}\n"
            )
        handle.write("END\n")

    return out


def generate_plumed_opes_script(
    *,
    ligand_plumed_idx: Sequence[int],
    pocket_ca_plumed_idx: Sequence[int],
    rmsd_reference_pdb: Path,
    sigma: Sequence[float],
    pace: int,
    barrier: float,
    biasfactor: float,
    temperature: float,
    save_opes_every: int,
    progress_every: int,
    out_dir: Path,
    state_rfile: Path | None = None,
    whole_molecule_plumed_idx: Sequence[int] | None = None,
    kernel_cutoff: float | None = None,
    nlist_parameters: tuple[float, float] | None = None,
    print_colvar_heavy_flush: bool = False,
) -> str:
    """Generate a two-dimensional PLUMED ``OPES_METAD`` input script.

    Parameters use PLUMED units: Angstrom for lengths, Kelvin for temperature,
    and kJ/mol for energies.  ``sigma`` must contain two values corresponding
    to ligand RMSD and ligand-pocket distance.

    ``KERNEL_CUTOFF`` controls truncated Gaussian kernel support (in units of
    per-dimension ``SIGMA``).  PLUMED's internal default follows
    ``sqrt(2 * BARRIER / (1 - 1/BIASFACTOR) / kBT)`` and can fall below 3.5,
    which triggers PLUMED's "kernels are truncated too much" warning.  When
    *kernel_cutoff* is ``None``, this function uses the same formula then takes
    ``max(3.5, ...)`` so typical decks avoid that warning without changing the
    collective variables.  ``BARRIER`` still controls ``EPSILON`` inside
    ``OPES_METAD`` as in PLUMED.  Pass an explicit *kernel_cutoff* to override.

    When *nlist_parameters* is ``(a, b)``, ``NLIST_PARAMETERS=a,b`` is emitted
    after ``NLIST`` for neighbor-list tuning without hand-editing the deck.

    When *print_colvar_heavy_flush* is ``True``, the ``PRINT`` line for ``COLVAR``
    includes PLUMED's ``HEAVY_FLUSH`` flag (requires a PLUMED build that contains
    the ``PRINT`` patch, e.g. the ``tps/v2.9.2-print-heavy-flush`` branch on the
    forked submodule). Stock conda PLUMED without that patch will error on the
    unknown keyword.
    """
    if len(sigma) != 2:
        raise ValueError("OPES-MD requires exactly two sigma values.")
    if int(pace) <= 0:
        raise ValueError("OPES deposition pace must be positive.")
    if int(progress_every) <= 0:
        raise ValueError("PLUMED PRINT stride must be positive.")
    if int(save_opes_every) <= 0:
        raise ValueError("PLUMED STATE_WSTRIDE must be positive.")
    if nlist_parameters is not None and len(nlist_parameters) != 2:
        raise ValueError("nlist_parameters must be a (cutoff, stride) pair of floats.")

    colvar_print_suffix = " HEAVY_FLUSH" if print_colvar_heavy_flush else ""

    out = out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ligand_atoms = _format_indices(ligand_plumed_idx)
    pocket_atoms = _format_indices(pocket_ca_plumed_idx)
    kernels_path = out / "KERNELS"
    state_path = out / "STATE"
    colvar_path = out / "COLVAR"

    cutoff_plumed_default = _opes_metad_default_kernel_cutoff_sigma(
        barrier, biasfactor, temperature
    )
    if kernel_cutoff is None:
        kernel_cutoff_resolved = max(3.5, cutoff_plumed_default)
    else:
        kernel_cutoff_resolved = float(kernel_cutoff)
        if kernel_cutoff_resolved <= 0.0:
            raise ValueError("kernel_cutoff must be positive when set explicitly.")

    lines = [
        "UNITS LENGTH=A",
        "",
    ]
    if whole_molecule_plumed_idx is not None:
        lines.append(
            "WHOLEMOLECULES ENTITY0="
            f"{_format_indices(list(whole_molecule_plumed_idx))}"
        )
        lines.append("")

    # CENTER computes an unweighted geometric center, matching the current
    # NumPy CVs that use mean coordinates rather than true mass-weighted COMs.
    lines.extend(
        [
            f"lig_rmsd: RMSD TYPE=OPTIMAL NOPBC REFERENCE={rmsd_reference_pdb.expanduser().resolve()}",
            f"lig_com: CENTER NOPBC ATOMS={ligand_atoms}",
            f"pocket_com: CENTER NOPBC ATOMS={pocket_atoms}",
            "lig_dist: DISTANCE NOPBC ATOMS=lig_com,pocket_com",
            "",
            "opes: OPES_METAD ...",
            "  ARG=lig_rmsd,lig_dist",
            f"  SIGMA={_format_float(sigma[0])},{_format_float(sigma[1])}",
            f"  PACE={int(pace)}",
            f"  BARRIER={_format_float(barrier)}",
            f"  BIASFACTOR={_format_float(biasfactor)}",
            f"  TEMP={_format_float(temperature)}",
            f"  KERNEL_CUTOFF={_format_float(kernel_cutoff_resolved)}",
            f"  FILE={kernels_path}",
            f"  STATE_WFILE={state_path}",
            f"  STATE_WSTRIDE={int(save_opes_every)}",
            "  NLIST",
        ]
    )
    if nlist_parameters is not None:
        a, b = (float(nlist_parameters[0]), float(nlist_parameters[1]))
        lines.append(
            f"  NLIST_PARAMETERS={_format_float(a)},{_format_float(b)}"
        )
    if state_rfile is not None:
        lines.append(f"  STATE_RFILE={state_rfile.expanduser().resolve()}")
    lines.extend(
        [
            "...",
            "",
            "PRINT "
            f"STRIDE={int(progress_every)} FILE={colvar_path} "
            "ARG=lig_rmsd,lig_dist,opes.bias,opes.rct,opes.nker,opes.zed,opes.neff"
            f"{colvar_print_suffix}",
            "",
        ]
    )
    return "\n".join(lines)


def add_plumed_opes_to_system(
    system: Any,
    script: str,
    *,
    temperature: float,
    force_group: int | None = None,
    masses: Iterable[float] | None = None,
    restart: bool = False,
) -> tuple[Any, int]:
    """Create a ``PlumedForce``, add it to *system*, and return it with its index.

    ``openmm-plumed`` is imported lazily so environments without the compiled
    plugin can still import this module and run pure unit tests.
    """
    try:
        from openmmplumed import PlumedForce
    except ImportError as exc:  # pragma: no cover - depends on optional plugin
        raise ImportError(
            "Real OPES-MD requires the openmm-plumed plugin. Install it with "
            "`conda install -c conda-forge openmm-plumed` or build the vendored "
            "`openmm-plumed/` source tree."
        ) from exc

    force = PlumedForce(script)
    force.setTemperature(float(temperature))
    force.setRestart(bool(restart))
    if masses is not None:
        force.setMasses([float(m) for m in masses])
    if force_group is not None:
        force.setForceGroup(int(force_group))
    force_index = int(system.addForce(force))
    return force, force_index

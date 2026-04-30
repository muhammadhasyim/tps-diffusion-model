"""Helpers for real OPES-MD through ``openmm-plumed``.

This module builds the PLUMED deck used by the OpenMM MD runner and adds the
resulting :class:`openmmplumed.PlumedForce` to an OpenMM ``System``.  The CVs
mirror the existing Python diagnostics:

* ligand pose RMSD after optimal alignment on pocket C-alpha atoms;
* ligand-to-pocket geometric-center distance;
* optional ligand–pocket heavy-atom coordination (``COORDINATION``), matching
  :func:`genai_tps.backends.boltz.collective_variables.protein_ligand_contacts`;
* optional ``cv_mode="oneopes"``: ``PROJECTION_ON_AXIS`` (``pp.proj`` / ``pp.ext``),
  ``CONTACTMAP SUM``, optional water ``COORDINATION`` sites with auxiliary
  ``OPES_METAD_EXPLORE`` biases (OneOPES / JPCL 2024 style).

The deck can use ``OPES_METAD`` (default) or ``OPES_METAD_EXPLORE`` and optional
``UPPER_WALLS`` on ``lig_dist`` with ``EXTRA_BIAS`` so OPES accounts for the wall.
Optionally, ``OPES_EXPANDED`` + ``ECV_MULTITHERMAL`` on ``ENERGY`` targets a
multithermal expanded ensemble in a single replica (replica-exchange-like sampling
without multiple walkers; see PLUMED ``OPES_EXPANDED`` documentation).
With periodic explicit solvent, pass ``use_pbc=True`` and ``whole_molecule_plumed_idx``
so ``WHOLEMOLECULES`` unwraps the solute and CV lines omit ``NOPBC`` for
minimum-image distances.

All atom indices passed to this module must already be PLUMED indices, i.e.
1-based OpenMM particle indices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = [
    "OpesPlumedDeckConfig",
    "add_plumed_opes_to_system",
    "compute_oneopes_pp_ext_angstrom",
    "compute_oneopes_pp_proj_cmap_from_boltz_coords",
    "default_oneopes_axis_boltz_indices",
    "default_oneopes_contact_pairs_boltz",
    "default_oneopes_contact_pairs_plumed",
    "generate_plumed_opes_script",
    "generate_plumed_opes_script_from_config",
    "write_rmsd_reference_pdb",
]

OpesCvMode = Literal["2d", "3d", "oneopes"]
OpesMetadVariant = Literal["metad", "explore"]


@dataclass(frozen=True)
class OpesPlumedDeckConfig:
    """Frozen configuration for :func:`generate_plumed_opes_script_from_config`."""

    ligand_plumed_idx: Sequence[int]
    pocket_ca_plumed_idx: Sequence[int]
    rmsd_reference_pdb: Path
    sigma: Sequence[float]
    pace: int
    barrier: float
    biasfactor: float
    temperature: float
    save_opes_every: int
    progress_every: int
    out_dir: Path
    state_rfile: Path | None = None
    whole_molecule_plumed_idx: Sequence[int] | None = None
    kernel_cutoff: float | None = None
    nlist_parameters: tuple[float, float] | None = None
    print_colvar_heavy_flush: bool = False
    cv_mode: OpesCvMode = "2d"
    pocket_heavy_plumed_idx: Sequence[int] | None = None
    coordination_r0: float = 4.5
    coordination_nn: int = 6
    coordination_mm: int = 12
    opes_variant: OpesMetadVariant = "metad"
    upper_wall_dist: float | None = None
    upper_wall_kappa: float = 200.0
    use_pbc: bool = False
    opes_expanded_temp_max: float | None = None
    opes_expanded_pace: int = 50
    opes_expanded_observation_steps: int = 100
    opes_expanded_print_stride: int = 100
    opes_expanded_state_rfile: Path | None = None
    oneopes_axis_p0_plumed_idx: Sequence[int] | None = None
    oneopes_axis_p1_plumed_idx: Sequence[int] | None = None
    oneopes_contactmap_pairs_plumed: Sequence[tuple[int, int]] | None = None
    contactmap_switch_r0: float = 5.5
    contactmap_switch_d0: float = 0.0
    contactmap_switch_nn: int = 4
    contactmap_switch_mm: int = 10
    oneopes_hydration_spot_plumed_idx: Sequence[int] | None = None
    water_oxygen_plumed_idx: Sequence[int] | None = None
    oneopes_water_pace: int = 40_000
    oneopes_water_barrier: float = 3.0
    oneopes_water_biasfactor: float = 5.0
    oneopes_water_sigma: float = 0.15
    oneopes_water_switch_r0: float = 2.5
    oneopes_water_switch_d0: float = 0.0
    oneopes_water_switch_nn: int = 6
    oneopes_water_switch_mm: int = 10
    oneopes_water_switch_d_max: float = 6.0
    oneopes_water_nl_cutoff: float = 16.0
    oneopes_water_nl_stride: int = 20
    oneopes_water_kernel_cutoff: float | None = None


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


def _com_mean(coords: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Unweighted geometric mean of *coords* at *indices* (0-based global atom rows)."""
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        raise ValueError("COM index list must be non-empty.")
    pts = np.asarray(coords, dtype=np.float64)[idx]
    return pts.mean(axis=0)


def compute_oneopes_pp_ext_angstrom(
    ref_coords_angstrom: np.ndarray,
    *,
    axis_p0_boltz: Sequence[int],
    axis_p1_boltz: Sequence[int],
    ligand_boltz: Sequence[int],
) -> float:
    """Return ``pp.ext`` (Å): orthogonal distance from ligand COM to pocket axis.

    The axis passes through the geometric centers of *axis_p0_boltz* and
    *axis_p1_boltz*; the ligand position is the COM of *ligand_boltz* rows in
    *ref_coords_angstrom* (Boltz-ordered, same layout as :class:`PoseCVIndexer`).
    """
    ref = np.asarray(ref_coords_angstrom, dtype=np.float64)
    p0 = _com_mean(ref, axis_p0_boltz)
    p1 = _com_mean(ref, axis_p1_boltz)
    lig = _com_mean(ref, ligand_boltz)
    axis = p1 - p0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("OneOPES axis anchors are degenerate (zero-length axis).")
    axis_u = axis / norm
    v = lig - p0
    # extension = | v - (v·û) û |
    along = float(np.dot(v, axis_u))
    perp = v - along * axis_u
    return float(np.linalg.norm(perp))


def _do_rational_plumed(rdist: float, nn: int, mm: int) -> float:
    """Reproduce PLUMED ``SwitchingFunction::do_rational`` (scalar value only)."""
    if 2 * nn == mm:
        r_n = rdist ** (nn - 1)
        return 1.0 / (1.0 + r_n * rdist)
    epsilon = 1e-14
    if (1.0 - 5.0e10 * epsilon) < rdist < (1.0 + 5.0e10 * epsilon):
        x = rdist - 1.0
        sec_dev = (nn * (mm * mm - 3.0 * mm * (-1 + nn) + nn * (-3 + 2 * nn))) / (6.0 * mm)
        dfunc = 0.5 * nn * float(nn - mm) / mm
        return float(nn) / float(mm) + x * (dfunc + 0.5 * x * sec_dev)
    r_n = rdist ** (nn - 1)
    r_m = rdist ** (mm - 1)
    num = 1.0 - r_n * rdist
    iden = 1.0 / (1.0 - r_m * rdist)
    return float(num * iden)


def _plumed_rational_contact_switch(
    distance: float,
    *,
    r0: float,
    d0: float,
    nn: int,
    mm: int,
) -> float:
    """MATCH PLUMED ``SwitchingFunction::calculate`` for ``SWITCH={RATIONAL ...}``."""
    nn_i, mm_i = int(nn), int(mm)
    if mm_i == 0:
        mm_i = 2 * nn_i
    if r0 <= 0.0:
        raise ValueError("contactmap_switch_r0 must be positive.")
    invr0 = 1.0 / r0
    dmax = d0 + r0 * (0.00001 ** (1.0 / float(nn_i - mm_i)))
    if distance > dmax:
        return 0.0
    rdist = (distance - d0) * invr0
    if rdist <= 0.0:
        return 1.0
    return _do_rational_plumed(rdist, nn_i, mm_i)


def compute_oneopes_pp_proj_cmap_from_boltz_coords(
    coords_angstrom: np.ndarray,
    *,
    axis_p0_boltz: Sequence[int],
    axis_p1_boltz: Sequence[int],
    ligand_boltz: Sequence[int],
    contact_pairs_boltz: Sequence[tuple[int, int]],
    contactmap_switch_r0: float = 5.5,
    contactmap_switch_d0: float = 0.0,
    contactmap_switch_nn: int = 4,
    contactmap_switch_mm: int = 10,
) -> np.ndarray:
    """Return ``[pp.proj, cmap]`` in Å / unitless sum, mirroring the PLUMED OneOPES deck.

    *coords_angstrom* must be Boltz-ordered solute heavy-atom rows (same layout as
    :func:`ligand_pose_rmsd` / the MD runner's ``get_coords_boltz_order`` slice).
    """
    cr = np.asarray(coords_angstrom, dtype=np.float64)
    p0 = _com_mean(cr, axis_p0_boltz)
    p1 = _com_mean(cr, axis_p1_boltz)
    lig = _com_mean(cr, ligand_boltz)
    axis = p1 - p0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("OneOPES axis anchors are degenerate (zero-length axis).")
    axis_u = axis / norm
    pp_proj = float(np.dot(lig - p0, axis_u))
    acc = 0.0
    for pr, li in contact_pairs_boltz:
        d = float(np.linalg.norm(cr[int(pr)] - cr[int(li)]))
        acc += _plumed_rational_contact_switch(
            d,
            r0=float(contactmap_switch_r0),
            d0=float(contactmap_switch_d0),
            nn=int(contactmap_switch_nn),
            mm=int(contactmap_switch_mm),
        )
    return np.array([pp_proj, acc], dtype=np.float64)


def default_oneopes_axis_boltz_indices(
    pocket_ca_boltz: Sequence[int],
    ref_coords_angstrom: np.ndarray,
    ligand_boltz: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Split pocket Cα atoms into proximal / distal anchors along the binding axis.

    Proximal (*p0*) = pocket Cα atoms closer than the median distance to the
    ligand COM; distal (*p1*) = the remainder.  This mirrors the PLUMED-NEST
    OneOPES ABFE convention (deep pocket vs solvent-facing pocket mouth).
    """
    ca = np.asarray(list(pocket_ca_boltz), dtype=np.int64)
    if ca.size < 2:
        raise ValueError(
            "default_oneopes_axis_boltz_indices requires at least two pocket Cα atoms."
        )
    ref = np.asarray(ref_coords_angstrom, dtype=np.float64)
    lig_com = _com_mean(ref, ligand_boltz)
    d = np.linalg.norm(ref[ca] - lig_com.reshape(1, 3), axis=1)
    med = float(np.median(d))
    p0 = ca[d <= med]
    p1 = ca[d > med]
    if p0.size == 0 or p1.size == 0:
        # Degenerate median split: fall back to first half / second half ordering.
        k = max(1, ca.size // 2)
        p0 = ca[:k]
        p1 = ca[k:]
    if p0.size == 0 or p1.size == 0:
        raise ValueError(
            "Could not form two non-empty pocket Cα groups for OneOPES axis anchors."
        )
    return p0, p1


def default_oneopes_contact_pairs_boltz(
    pocket_heavy_boltz: Sequence[int],
    ligand_boltz: Sequence[int],
    *,
    max_pairs: int = 6,
) -> list[tuple[int, int]]:
    """Zip pocket-heavy vs ligand **Boltz** global indices into (protein, ligand) pairs."""
    ph = [int(i) for i in pocket_heavy_boltz]
    lg = [int(i) for i in ligand_boltz]
    if not ph or not lg:
        raise ValueError(
            "default_oneopes_contact_pairs_boltz requires non-empty pocket and ligand lists."
        )
    n = min(len(ph), len(lg), int(max_pairs))
    return [(ph[i], lg[i]) for i in range(n)]


def default_oneopes_contact_pairs_plumed(
    pocket_heavy_plumed_idx: Sequence[int],
    ligand_plumed_idx: Sequence[int],
    *,
    max_pairs: int = 6,
) -> list[tuple[int, int]]:
    """Zip pocket-heavy vs ligand PLUMED indices into (protein, ligand) pairs.

    This is a **deterministic fallback** for automated decks when no explicit
    contact map is supplied; for production runs, prefer hand-curated pairs from
    crystal-structure chemistry.
    """
    ph = [int(i) for i in pocket_heavy_plumed_idx]
    lg = [int(i) for i in ligand_plumed_idx]
    if not ph or not lg:
        raise ValueError(
            "default_oneopes_contact_pairs_plumed requires non-empty pocket and "
            "ligand PLUMED index lists."
        )
    n = min(len(ph), len(lg), int(max_pairs))
    return [(ph[i], lg[i]) for i in range(n)]


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


def generate_plumed_opes_script_from_config(cfg: OpesPlumedDeckConfig) -> str:
    """Generate a PLUMED ``OPES_METAD`` or ``OPES_METAD_EXPLORE`` input script.

    Parameters use PLUMED units: Angstrom for lengths, Kelvin for temperature,
    and kJ/mol for energies.  The length of *sigma* must match the number of
    biased collective variables: two for ``cv_mode="2d"`` (``lig_rmsd``,
    ``lig_dist``), three for ``cv_mode="3d"`` (adds ``lig_contacts``), two for
    ``cv_mode="oneopes"`` (``pp.proj`` + ``cmap`` from ``PROJECTION_ON_AXIS`` and
    ``CONTACTMAP SUM``).

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

    *opes_variant* selects ``OPES_METAD`` (``"metad"``) or ``OPES_METAD_EXPLORE``
    (``"explore"``).  When *upper_wall_dist* is set, an ``UPPER_WALLS`` restraint
    is added and referenced via ``EXTRA_BIAS`` on the main OPES line: for
    ``cv_mode`` ``"2d"`` / ``"3d"`` the wall acts on ``lig_dist``; for
    ``cv_mode="oneopes"`` it acts on ``pp.ext`` (orthogonal distance from the
    ligand COM to the pocket axis), i.e. a funnel on lateral excursions.  PLUMED's
    ``OPES_METAD_EXPLORE`` does not register ``EXTRA_BIAS`` (only
    ``OPES_METAD`` does), so if both *opes_variant* ``"explore"`` and a wall are
    requested, the emitted action is ``OPES_METAD`` with a comment explaining
    the downgrade.

    When *use_pbc* is ``True`` (explicit solvent / periodic box): emit
    ``WHOLEMOLECULES`` for *whole_molecule_plumed_idx* and omit ``NOPBC`` on
    ``RMSD``, ``CENTER``, ``DISTANCE``, and ``COORDINATION`` so minimum-image
    distances apply.  *whole_molecule_plumed_idx* must be non-empty when
    *use_pbc* is ``True``.  When *use_pbc* is ``False`` (legacy implicit solvent),
    ``NOPBC`` is kept and ``WHOLEMOLECULES`` is not emitted.

    When *opes_expanded_temp_max* is set, the deck adds ``ene: ENERGY``,
    ``ECV_MULTITHERMAL`` over ``[TEMP, TEMP_MAX]``, and ``OPES_EXPANDED`` with
    ``PACE=opes_expanded_pace``.  This is additive to ``OPES_METAD`` /
    ``OPES_METAD_EXPLORE`` and uses a fixed-volume potential energy expansion
    (``U = E``).  For NPT, build a custom deck with ``U = E + pV`` via
    ``CUSTOM`` and ``VOLUME`` instead of using this shortcut.

    *opes_expanded_state_rfile* is passed as ``STATE_RFILE`` on
    ``OPES_EXPANDED`` when restarting from a saved expanded state (typically
    ``STATE_EXPANDED`` beside the main ``STATE`` file).

    ``cv_mode="oneopes"`` emits the literature OneOPES-style ligand–pocket CVs:
    ``PROJECTION_ON_AXIS`` (``pp.proj`` / ``pp.ext``) with axis anchors
    *oneopes_axis_p0_plumed_idx* / *oneopes_axis_p1_plumed_idx*, ``CONTACTMAP SUM``
    over *oneopes_contactmap_pairs_plumed* (each tuple is PLUMED 1-based
    protein–ligand atom indices), optional auxiliary ``OPES_METAD_EXPLORE``
    biases on per-site water ``COORDINATION`` CVs for each index in
    *oneopes_hydration_spot_plumed_idx* with water oxygens in
    *water_oxygen_plumed_idx* (a ``GROUP`` labeled ``WO``).
    """
    ligand_plumed_idx = cfg.ligand_plumed_idx
    pocket_ca_plumed_idx = cfg.pocket_ca_plumed_idx
    rmsd_reference_pdb = cfg.rmsd_reference_pdb
    sigma = cfg.sigma
    pace = cfg.pace
    barrier = cfg.barrier
    biasfactor = cfg.biasfactor
    temperature = cfg.temperature
    save_opes_every = cfg.save_opes_every
    progress_every = cfg.progress_every
    out_dir = cfg.out_dir
    state_rfile = cfg.state_rfile
    whole_molecule_plumed_idx = cfg.whole_molecule_plumed_idx
    kernel_cutoff = cfg.kernel_cutoff
    nlist_parameters = cfg.nlist_parameters
    print_colvar_heavy_flush = cfg.print_colvar_heavy_flush
    cv_mode = cfg.cv_mode
    pocket_heavy_plumed_idx = cfg.pocket_heavy_plumed_idx
    coordination_r0 = cfg.coordination_r0
    coordination_nn = cfg.coordination_nn
    coordination_mm = cfg.coordination_mm
    opes_variant = cfg.opes_variant
    upper_wall_dist = cfg.upper_wall_dist
    upper_wall_kappa = cfg.upper_wall_kappa
    use_pbc = cfg.use_pbc
    opes_expanded_temp_max = cfg.opes_expanded_temp_max
    opes_expanded_pace = cfg.opes_expanded_pace
    opes_expanded_observation_steps = cfg.opes_expanded_observation_steps
    opes_expanded_print_stride = cfg.opes_expanded_print_stride
    opes_expanded_state_rfile = cfg.opes_expanded_state_rfile
    oneopes_axis_p0_plumed_idx = cfg.oneopes_axis_p0_plumed_idx
    oneopes_axis_p1_plumed_idx = cfg.oneopes_axis_p1_plumed_idx
    oneopes_contactmap_pairs_plumed = cfg.oneopes_contactmap_pairs_plumed
    contactmap_switch_r0 = cfg.contactmap_switch_r0
    contactmap_switch_d0 = cfg.contactmap_switch_d0
    contactmap_switch_nn = cfg.contactmap_switch_nn
    contactmap_switch_mm = cfg.contactmap_switch_mm
    oneopes_hydration_spot_plumed_idx = cfg.oneopes_hydration_spot_plumed_idx
    water_oxygen_plumed_idx = cfg.water_oxygen_plumed_idx
    oneopes_water_pace = cfg.oneopes_water_pace
    oneopes_water_barrier = cfg.oneopes_water_barrier
    oneopes_water_biasfactor = cfg.oneopes_water_biasfactor
    oneopes_water_sigma = cfg.oneopes_water_sigma
    oneopes_water_switch_r0 = cfg.oneopes_water_switch_r0
    oneopes_water_switch_d0 = cfg.oneopes_water_switch_d0
    oneopes_water_switch_nn = cfg.oneopes_water_switch_nn
    oneopes_water_switch_mm = cfg.oneopes_water_switch_mm
    oneopes_water_switch_d_max = cfg.oneopes_water_switch_d_max
    oneopes_water_nl_cutoff = cfg.oneopes_water_nl_cutoff
    oneopes_water_nl_stride = cfg.oneopes_water_nl_stride
    oneopes_water_kernel_cutoff = cfg.oneopes_water_kernel_cutoff
    if cv_mode not in ("2d", "3d", "oneopes"):
        raise ValueError('cv_mode must be "2d", "3d", or "oneopes".')
    if opes_variant not in ("metad", "explore"):
        raise ValueError('opes_variant must be "metad" or "explore".')
    if cv_mode == "3d":
        if pocket_heavy_plumed_idx is None or len(pocket_heavy_plumed_idx) == 0:
            raise ValueError(
                'cv_mode="3d" requires a non-empty pocket_heavy_plumed_idx list.'
            )
    oneopes_hydr_spots: tuple[int, ...] = ()
    if cv_mode == "oneopes":
        oneopes_hydr_spots = tuple(int(x) for x in (oneopes_hydration_spot_plumed_idx or ()))
        if oneopes_axis_p0_plumed_idx is None or len(list(oneopes_axis_p0_plumed_idx)) == 0:
            raise ValueError(
                'cv_mode="oneopes" requires non-empty oneopes_axis_p0_plumed_idx.'
            )
        if oneopes_axis_p1_plumed_idx is None or len(list(oneopes_axis_p1_plumed_idx)) == 0:
            raise ValueError(
                'cv_mode="oneopes" requires non-empty oneopes_axis_p1_plumed_idx.'
            )
        if oneopes_contactmap_pairs_plumed is None or len(oneopes_contactmap_pairs_plumed) == 0:
            raise ValueError(
                'cv_mode="oneopes" requires at least one entry in '
                "oneopes_contactmap_pairs_plumed."
            )
        if oneopes_hydr_spots:
            if water_oxygen_plumed_idx is None or len(list(water_oxygen_plumed_idx)) == 0:
                raise ValueError(
                    "oneopes_hydration_spot_plumed_idx is non-empty but "
                    "water_oxygen_plumed_idx is missing or empty (define a WO GROUP)."
                )
            if int(oneopes_water_pace) <= 0:
                raise ValueError("oneopes_water_pace must be positive.")
            if float(oneopes_water_barrier) <= 0.0:
                raise ValueError("oneopes_water_barrier must be positive.")
            if float(oneopes_water_biasfactor) <= 1.0:
                raise ValueError("oneopes_water_biasfactor must exceed 1.")
            if float(oneopes_water_sigma) <= 0.0:
                raise ValueError("oneopes_water_sigma must be positive.")

    cv_names: tuple[str, ...]
    if cv_mode == "oneopes":
        cv_names = ("pp.proj", "cmap")
    elif cv_mode == "3d":
        cv_names = ("lig_rmsd", "lig_dist", "lig_contacts")
    else:
        cv_names = ("lig_rmsd", "lig_dist")
    n_cv = len(cv_names)
    sigma_t = tuple(float(s) for s in sigma)
    if len(sigma_t) != n_cv:
        raise ValueError(
            f"OPES-MD requires {n_cv} sigma values for cv_mode={cv_mode!r} "
            f"(got {len(sigma_t)})."
        )
    if int(pace) <= 0:
        raise ValueError("OPES deposition pace must be positive.")
    if int(progress_every) <= 0:
        raise ValueError("PLUMED PRINT stride must be positive.")
    if int(save_opes_every) <= 0:
        raise ValueError("PLUMED STATE_WSTRIDE must be positive.")
    if nlist_parameters is not None and len(nlist_parameters) != 2:
        raise ValueError("nlist_parameters must be a (cutoff, stride) pair of floats.")
    if upper_wall_dist is not None and float(upper_wall_dist) <= 0.0:
        raise ValueError("upper_wall_dist must be positive when set.")
    if float(upper_wall_kappa) <= 0.0:
        raise ValueError("upper_wall_kappa must be positive.")
    oneopes_n_hydr = len(oneopes_hydr_spots)
    if use_pbc:
        if whole_molecule_plumed_idx is None or len(list(whole_molecule_plumed_idx)) == 0:
            raise ValueError(
                "use_pbc=True requires a non-empty whole_molecule_plumed_idx list "
                "for WHOLEMOLECULES unwrapping."
            )

    use_expanded = opes_expanded_temp_max is not None
    if use_expanded:
        t_max = float(opes_expanded_temp_max)
        t0 = float(temperature)
        if t_max <= t0 + 1e-9:
            raise ValueError(
                "opes_expanded_temp_max must be strictly greater than simulation "
                f"TEMP (got TEMP_MAX={t_max:g} K vs TEMP={t0:g} K)."
            )
        ep = int(opes_expanded_pace)
        if ep <= 0:
            raise ValueError("opes_expanded_pace must be positive.")
        obs = int(opes_expanded_observation_steps)
        if obs < 1:
            raise ValueError("opes_expanded_observation_steps must be >= 1.")
        ps = int(opes_expanded_print_stride)
        if ps < 1:
            raise ValueError("opes_expanded_print_stride must be >= 1.")

    colvar_print_suffix = " HEAVY_FLUSH" if print_colvar_heavy_flush else ""

    out = out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ligand_atoms = _format_indices(ligand_plumed_idx)
    pocket_atoms = _format_indices(pocket_ca_plumed_idx)
    kernels_path = out / "KERNELS"
    state_path = out / "STATE"
    colvar_path = out / "COLVAR"
    expanded_delta_path = out / "OPES_EXPANDED_DELTAFS"
    expanded_state_path = out / "STATE_EXPANDED"

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
    if use_pbc and whole_molecule_plumed_idx is not None:
        lines.append(
            "WHOLEMOLECULES ENTITY0="
            f"{_format_indices(list(whole_molecule_plumed_idx))}"
        )
        lines.append("")

    pbc_token = "" if use_pbc else " NOPBC"

    # CENTER computes an unweighted geometric center, matching the current
    # NumPy CVs that use mean coordinates rather than true mass-weighted COMs.
    lines.extend(
        [
            f"lig_rmsd: RMSD TYPE=OPTIMAL{pbc_token} REFERENCE={rmsd_reference_pdb.expanduser().resolve()}",
            f"lig_com: CENTER{pbc_token} ATOMS={ligand_atoms}",
            f"pocket_com: CENTER{pbc_token} ATOMS={pocket_atoms}",
            f"lig_dist: DISTANCE{pbc_token} ATOMS=lig_com,pocket_com",
        ]
    )
    if cv_mode == "3d":
        assert pocket_heavy_plumed_idx is not None
        pocket_heavy = _format_indices(pocket_heavy_plumed_idx)
        lines.append(
            "lig_contacts: COORDINATION GROUPA="
            f"{ligand_atoms} GROUPB={pocket_heavy} "
            f"R_0={_format_float(float(coordination_r0))} "
            f"NN={int(coordination_nn)} MM={int(coordination_mm)}{pbc_token}"
        )
    if cv_mode == "oneopes":
        assert oneopes_axis_p0_plumed_idx is not None and oneopes_axis_p1_plumed_idx is not None
        assert oneopes_contactmap_pairs_plumed is not None
        p0_atoms = _format_indices(list(oneopes_axis_p0_plumed_idx))
        p1_atoms = _format_indices(list(oneopes_axis_p1_plumed_idx))
        lines.extend(
            [
                f"oneopes_axis_p0: CENTER{pbc_token} ATOMS={p0_atoms}",
                f"oneopes_axis_p1: CENTER{pbc_token} ATOMS={p1_atoms}",
                (
                    "pp: PROJECTION_ON_AXIS AXIS_ATOMS=oneopes_axis_p0,oneopes_axis_p1 "
                    f"ATOM=lig_com{pbc_token}"
                ),
            ]
        )
        cmap_atoms_parts: list[str] = []
        for pi, li in oneopes_contactmap_pairs_plumed:
            cmap_atoms_parts.append(f"ATOMS{len(cmap_atoms_parts) + 1}={int(pi)},{int(li)}")
        sw = (
            "SWITCH={RATIONAL "
            f"R_0={_format_float(float(contactmap_switch_r0))} "
            f"D_0={_format_float(float(contactmap_switch_d0))} "
            f"NN={int(contactmap_switch_nn)} MM={int(contactmap_switch_mm)}}}"
        )
        cmap_line = "cmap: CONTACTMAP SUM " + " ".join(cmap_atoms_parts) + f" {sw}{pbc_token}"
        lines.append(cmap_line)
        if oneopes_hydr_spots:
            assert water_oxygen_plumed_idx is not None
            wo = _format_indices(list(water_oxygen_plumed_idx))
            lines.append(f"WO: GROUP ATOMS={wo}")
            for si, spot in enumerate(oneopes_hydr_spots):
                w_sw = (
                    "SWITCH={RATIONAL "
                    f"D_0={_format_float(float(oneopes_water_switch_d0))} "
                    f"R_0={_format_float(float(oneopes_water_switch_r0))} "
                    f"NN={int(oneopes_water_switch_nn)} MM={int(oneopes_water_switch_mm)} "
                    f"D_MAX={_format_float(float(oneopes_water_switch_d_max))}"
                    "}"
                )
                lines.append(
                    f"hydr_{si}: COORDINATION GROUPA={int(spot)} GROUPB=WO {w_sw}"
                    f" NLIST NL_CUTOFF={_format_float(float(oneopes_water_nl_cutoff))} "
                    f"NL_STRIDE={int(oneopes_water_nl_stride)}{pbc_token}"
                )
    lines.append("")
    if use_expanded:
        lines.extend(
            [
                "# Multithermal expanded ensemble (OPES_EXPANDED + ECV_MULTITHERMAL).",
                "ene: ENERGY",
                "",
            ]
        )

    wall_label = "pp_ext_uwall" if cv_mode == "oneopes" else "lig_dist_uwall"
    wall_arg = "pp.ext" if cv_mode == "oneopes" else "lig_dist"
    if upper_wall_dist is not None:
        # Action label is the leading ``name:`` token; UPPER_WALLS has no LABEL= keyword.
        lines.append(
            f"{wall_label}: UPPER_WALLS ARG={wall_arg} "
            f"AT={_format_float(float(upper_wall_dist))} "
            f"KAPPA={_format_float(float(upper_wall_kappa))}"
        )
        lines.append("")

    # PLUMED OPESmetad.cpp registers EXTRA_BIAS only for OPES_METAD, not EXPLORE.
    opes_action = (
        "OPES_METAD_EXPLORE" if opes_variant == "explore" else "OPES_METAD"
    )
    if upper_wall_dist is not None and opes_variant == "explore":
        lines.append(
            "# OPES_METAD_EXPLORE has no EXTRA_BIAS in PLUMED; using OPES_METAD with wall."
        )
        lines.append("")
        opes_action = "OPES_METAD"
    arg_str = ",".join(cv_names)
    sigma_str = ",".join(_format_float(s) for s in sigma_t)
    if cv_mode == "oneopes":
        print_cv_parts = ["lig_rmsd", "lig_dist", "pp.proj", "pp.ext", "cmap"]
        print_cv_parts += [f"hydr_{i}" for i in range(oneopes_n_hydr)]
    else:
        print_cv_parts = list(cv_names)
    print_cv_str = ",".join(print_cv_parts) + ",opes.bias,opes.rct,opes.nker,opes.zed,opes.neff"
    for hi in range(oneopes_n_hydr):
        print_cv_str += (
            f",opes_hydr_{hi}.bias,opes_hydr_{hi}.rct,opes_hydr_{hi}.nker,"
            f"opes_hydr_{hi}.zed,opes_hydr_{hi}.neff"
        )
    if use_expanded:
        print_cv_str += ",ene,opes_expanded.bias"

    opes_lines = [
        f"opes: {opes_action} ...",
        f"  ARG={arg_str}",
    ]
    if upper_wall_dist is not None:
        opes_lines.append(f"  EXTRA_BIAS={wall_label}.bias")
    opes_lines.extend(
        [
            f"  SIGMA={sigma_str}",
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
    lines.extend(opes_lines)
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
        ]
    )

    if cv_mode == "oneopes" and oneopes_n_hydr > 0:
        w_barrier = float(oneopes_water_barrier)
        w_bf = float(oneopes_water_biasfactor)
        w_temp = float(temperature)
        w_cutoff_default = _opes_metad_default_kernel_cutoff_sigma(
            w_barrier, w_bf, w_temp
        )
        if oneopes_water_kernel_cutoff is None:
            w_kernel_cut = max(3.5, w_cutoff_default)
        else:
            w_kernel_cut = float(oneopes_water_kernel_cutoff)
            if w_kernel_cut <= 0.0:
                raise ValueError("oneopes_water_kernel_cutoff must be positive when set.")
        w_sig = _format_float(float(oneopes_water_sigma))
        lines.append("# Auxiliary low-barrier OPES on hydration COORDINATION CVs (OneOPES-style).")
        lines.append("")
        for hi in range(oneopes_n_hydr):
            w_kernels = out / f"KERNELS_HYDR_{hi}"
            w_state = out / f"STATE_HYDR_{hi}"
            lines.extend(
                [
                    f"opes_hydr_{hi}: OPES_METAD_EXPLORE ...",
                    f"  ARG=hydr_{hi}",
                    f"  SIGMA={w_sig}",
                    f"  PACE={int(oneopes_water_pace)}",
                    f"  BARRIER={_format_float(w_barrier)}",
                    f"  BIASFACTOR={_format_float(w_bf)}",
                    f"  TEMP={_format_float(w_temp)}",
                    f"  KERNEL_CUTOFF={_format_float(w_kernel_cut)}",
                    f"  FILE={w_kernels}",
                    f"  STATE_WFILE={w_state}",
                    f"  STATE_WSTRIDE={int(save_opes_every)}",
                    "  NLIST",
                ]
            )
            if nlist_parameters is not None:
                a, b = (float(nlist_parameters[0]), float(nlist_parameters[1]))
                lines.append(
                    f"  NLIST_PARAMETERS={_format_float(a)},{_format_float(b)}"
                )
            lines.extend(
                [
                    "...",
                    "",
                ]
            )

    if use_expanded:
        # STATE_WSTRIDE must be >= PACE (MD steps) and align with bias PACE cycles.
        ep = int(opes_expanded_pace)
        wstride = max(ep, int(save_opes_every) // ep * ep)
        if wstride < int(save_opes_every):
            wstride += ep
        lines.extend(
            [
                (
                    f"opes_ecv: ECV_MULTITHERMAL ARG=ene TEMP={_format_float(float(temperature))} "
                    f"TEMP_MIN={_format_float(float(temperature))} "
                    f"TEMP_MAX={_format_float(float(opes_expanded_temp_max))}"
                ),
                "opes_expanded: OPES_EXPANDED ...",
                "  ARG=opes_ecv.*",
                f"  PACE={ep}",
                f"  OBSERVATION_STEPS={int(opes_expanded_observation_steps)}",
                f"  FILE={expanded_delta_path}",
                f"  PRINT_STRIDE={int(opes_expanded_print_stride)}",
                f"  STATE_WFILE={expanded_state_path}",
                f"  STATE_WSTRIDE={int(wstride)}",
            ]
        )
        if opes_expanded_state_rfile is not None:
            lines.append(
                f"  STATE_RFILE={opes_expanded_state_rfile.expanduser().resolve()}"
            )
        lines.extend(
            [
                "...",
                "",
            ]
        )

    lines.extend(
        [
            "PRINT "
            f"STRIDE={int(progress_every)} FILE={colvar_path} "
            f"ARG={print_cv_str}"
            f"{colvar_print_suffix}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_plumed_opes_script(**kwargs: Any) -> str:
    """Generate a PLUMED deck from keyword arguments (backward-compatible API).

    Prefer :func:`generate_plumed_opes_script_from_config` with an
    :class:`OpesPlumedDeckConfig` instance for clearer call sites.
    """
    return generate_plumed_opes_script_from_config(OpesPlumedDeckConfig(**kwargs))


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

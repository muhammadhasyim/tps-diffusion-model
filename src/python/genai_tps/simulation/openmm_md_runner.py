"""OPES-biased OpenMM MD driver for Boltz-aligned WDSM ground-truth samples."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _read_latest_plain_colvar_row(
    colvar_path: Path, *, max_chars: int = 220
) -> str | None:
    """Return the last non-comment row from a PLUMED COLVAR file (truncated)."""
    if not colvar_path.is_file() or colvar_path.stat().st_size < 8:
        return None
    try:
        text = colvar_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s if len(s) <= max_chars else s[: max_chars - 3] + "..."
    return None


def _log_coord_stats_np(tag: str, coords: np.ndarray, *, unit_label: str) -> None:
    """Print finite-ness and axis bounds for a coordinate array (diagnostics)."""
    a = np.asarray(coords, dtype=np.float64)
    finite = bool(np.isfinite(a).all())
    if a.size == 0:
        print(f"[MD-OPES] {tag}: empty array ({unit_label})", flush=True)
        return
    print(
        f"[MD-OPES] {tag}: finite={finite} shape={a.shape} ({unit_label}) "
        f"min=({a[:, 0].min():.4f},{a[:, 1].min():.4f},{a[:, 2].min():.4f}) "
        f"max=({a[:, 0].max():.4f},{a[:, 1].max():.4f},{a[:, 2].max():.4f})",
        flush=True,
    )


def _diagnostic_energy_and_large_forces(
    sim,
    tag: str,
    *,
    large_force_threshold: float = 100_000.0,
    max_print: int = 30,
) -> None:
    """Log potential energy and OpenMM particles with very large forces (OpenMM FAQ)."""
    from math import sqrt

    import openmm.unit as u

    st = sim.context.getState(getEnergy=True, getForces=True)
    pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f"[MD-OPES] {tag}: potential_energy={pe:.3f} kJ/mol", flush=True)
    forces = st.getForces().value_in_unit(u.kilojoules_per_mole / u.nanometer)
    n_big = 0
    for i, f in enumerate(forces):
        mag = sqrt(float(f.x) ** 2 + float(f.y) ** 2 + float(f.z) ** 2)
        if mag > large_force_threshold:
            if n_big < max_print:
                print(
                    f"[MD-OPES] {tag}: large_force atom_index={i} |F|={mag:.3e} kJ/mol/nm",
                    flush=True,
                )
            n_big += 1
    if n_big > max_print:
        print(
            f"[MD-OPES] {tag}: ({n_big - max_print} more atoms with |F| > "
            f"{large_force_threshold:g} ...)",
            flush=True,
        )


_DEFAULT_OPES_WALL_MARGIN_ANGSTROM = 10.0


def default_opes_upper_wall_dist_angstrom(
    initial_ligand_pocket_dist_angstrom: float,
    *,
    margin_angstrom: float = _DEFAULT_OPES_WALL_MARGIN_ANGSTROM,
) -> float:
    """Return UPPER_WALLS ``AT`` distance (Å) from initial pocket COM distance + margin.

    Used when ``--opes-wall-dist`` is omitted so OPES exploration is bounded
    relative to the starting pose without unbinding to arbitrary distances.
    """
    return float(initial_ligand_pocket_dist_angstrom) + float(margin_angstrom)


def _parse_opes_nlist_parameters(value: str) -> tuple[float, float]:
    """Parse ``'a,b'`` into PLUMED ``NLIST_PARAMETERS`` components."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated floats, e.g. '4.0,0.4'"
        )
    return float(parts[0]), float(parts[1])


def _parse_comma_int_list(value: str) -> list[int]:
    """Parse ``'1,2,3'`` into a list of integers (empty string → empty list)."""
    value = value.strip()
    if not value:
        return []
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def _parse_oneopes_contact_pairs_boltz(value: str) -> list[tuple[int, int]]:
    """Parse ``'p-l,p-l'`` Boltz global atom index pairs (hyphen-separated)."""
    out: list[tuple[int, int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise argparse.ArgumentTypeError(
                f"invalid OneOPES contact pair {chunk!r}; expected 'protein_boltz-ligand_boltz'"
            )
        a, b = chunk.split("-", 1)
        out.append((int(a.strip()), int(b.strip())))
    return out


def collect_water_oxygen_plumed_1based(topology: Any) -> list[int]:
    """Return PLUMED 1-based indices of explicit-solvent water oxygen atoms.

    Matches common Amber/OpenMM TIP3P residue names and oxygen atom labels.
    """
    water_like = {
        "HOH",
        "WAT",
        "SOL",
        "TIP3",
        "TIP3P",
        "SPC",
        "SPC/E",
        "OPC",
        "OPC3",
        "H2O",
    }
    o_names = {"O", "OW", "OH2"}
    out: list[int] = []
    for atom in topology.atoms():
        res = str(atom.residue.name).strip().upper()
        if res not in water_like:
            continue
        nm = str(atom.name).strip().upper()
        if nm in o_names:
            out.append(int(atom.index) + 1)
    return out


def _require_finite_positions(pos_nm: np.ndarray, *, context_msg: str) -> None:
    a = np.asarray(pos_nm, dtype=np.float64)
    if not np.isfinite(a).all():
        n_bad = int(np.sum(~np.isfinite(a)))
        raise RuntimeError(
            f"[MD-OPES] Non-finite particle positions after {context_msg} "
            f"({n_bad} non-finite values in position array). "
            "See OpenMM NaN FAQ: https://github.com/openmm/openmm/wiki/"
            "Frequently-Asked-Questions#nan"
        )


def _next_observer_event_step(
    completed: int,
    n_steps: int,
    periods: tuple[int, ...],
) -> int:
    """Return the smallest step index in ``(completed, n_steps]`` on any period boundary.

    If no boundary lies before ``n_steps``, returns ``n_steps`` so the remaining
    trajectory can be integrated in one batch without host checkpoints.

    Parameters
    ----------
    completed
        Number of MD steps already taken (0 before the first batch).
    n_steps
        Total planned MD steps.
    periods
        Positive integers (e.g. deposit pace, save interval, progress interval).
    """
    if completed >= n_steps:
        return n_steps
    m = completed + 1
    nxt: int | None = None
    for p in periods:
        if p <= 0:
            raise ValueError("observer checkpoint periods must be positive")
        k = ((m + p - 1) // p) * p
        if k <= n_steps:
            nxt = k if nxt is None else min(nxt, k)
    return nxt if nxt is not None else n_steps


def build_opes_md_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPES-biased OpenMM MD for WDSM data collection")
    parser.add_argument("--topo-npz", type=Path, required=True, help="Boltz processed structures NPZ")
    parser.add_argument("--frame-npz", type=Path, default=None, help="NPZ with initial coords (default: same as topo)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index in frame-npz")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-steps", type=int, default=5_000_000, help="Total MD steps (default: 5M = 10ns at 2fs)")
    parser.add_argument("--save-every", type=int, default=1000, help="Save structure every N steps (default: 1000 = 2ps)")
    parser.add_argument("--deposit-pace", type=int, default=500, help="OPES kernel deposit every N steps")
    parser.add_argument(
        "--opes-barrier",
        type=float,
        default=40.0,
        help="OPES BARRIER (kJ/mol); should exceed the largest FES barrier you target.",
    )
    parser.add_argument("--opes-biasfactor", type=float, default=10.0)
    parser.add_argument(
        "--opes-sigma",
        type=str,
        default="0.3,0.5",
        help=(
            "Comma-separated kernel sigmas (Å): two for --bias-cv 2d or oneopes, "
            "three for 3d."
        ),
    )
    parser.add_argument(
        "--bias-cv",
        type=str,
        default="2d",
        choices=["2d", "3d", "oneopes"],
        help=(
            "PLUMED CV set: 2d = lig_rmsd+lig_dist; 3d adds lig_contacts (COORDINATION "
            "ligand vs pocket heavy atoms); oneopes = PROJECTION_ON_AXIS (pp.proj) + "
            "CONTACTMAP SUM (requires --opes-mode plumed; see --oneopes-* flags)."
        ),
    )
    parser.add_argument(
        "--opes-explore",
        action="store_true",
        help="Use OPES_METAD_EXPLORE instead of OPES_METAD (more aggressive exploration).",
    )
    parser.add_argument(
        "--opes-wall-dist",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help=(
            "UPPER_WALLS at this distance (Å) with EXTRA_BIAS in OPES: on lig_dist "
            "for --bias-cv 2d/3d; on pp.ext (orthogonal extension from the OneOPES "
            "axis) for --bias-cv oneopes. If omitted in PLUMED mode, the wall is set "
            "adaptively from the initial pose plus "
            f"{_DEFAULT_OPES_WALL_MARGIN_ANGSTROM:g} Å."
        ),
    )
    parser.add_argument(
        "--opes-wall-kappa",
        type=float,
        default=200.0,
        help="Harmonic constant (kJ/mol/Å^2) for --opes-wall-dist UPPER_WALLS.",
    )
    parser.add_argument(
        "--coordination-r0",
        type=float,
        default=4.5,
        help="COORDINATION R_0 (Å) for lig_contacts; matches protein_ligand_contacts r0 in observer mode.",
    )
    parser.add_argument("--opes-restart", type=Path, default=None, help="Resume OPES from saved state")
    parser.add_argument(
        "--opes-kernel-cutoff",
        type=float,
        default=None,
        metavar="SIGMA",
        help=(
            "PLUMED OPES_METAD KERNEL_CUTOFF in units of per-CV SIGMA. "
            "If omitted, uses PLUMED's default formula max(3.5, sqrt(...)) "
            "to avoid the 'kernels are truncated too much' warning at typical "
            "BARRIER/BIASFACTOR/T."
        ),
    )
    parser.add_argument(
        "--opes-nlist-parameters",
        type=_parse_opes_nlist_parameters,
        default=None,
        metavar="A,B",
        help=(
            "Optional PLUMED NLIST_PARAMETERS for OPES_METAD neighbor list, "
            "e.g. '4.0,0.4' (comma-separated floats)."
        ),
    )
    parser.add_argument(
        "--opes-mode",
        type=str,
        default="plumed",
        choices=["plumed", "observer"],
        help=(
            "OPES implementation: 'plumed' applies OPES_METAD / OPES_METAD_EXPLORE "
            "bias forces through openmm-plumed (requires a PLUMED build with the "
            "'opes' module — not included in default conda-forge plumed). "
            "'observer' runs unbiased MD and updates Python OPESBias only "
            "(different physics than PLUMED OPES)."
        ),
    )
    parser.add_argument(
        "--plumed-force-group",
        type=int,
        default=30,
        help="OpenMM force group used for the PLUMED bias energy.",
    )
    parser.add_argument("--temperature", type=float, default=300.0, help="Langevin temperature (K)")
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument("--platform", type=str, default="CUDA", choices=["CUDA", "OpenCL", "CPU"])
    parser.add_argument(
        "--openmm-device-index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CUDA/OpenCL GPU ordinal for OpenMM (platform property DeviceIndex). "
            "Ignored when the resolved platform is CPU."
        ),
    )
    parser.add_argument("--minimize-steps", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--save-opes-every", type=int, default=50000)
    parser.add_argument(
        "--plumed-colvar-heavy-flush",
        action="store_true",
        help=(
            "Emit PRINT ... HEAVY_FLUSH for COLVAR (PLUMED reopens the file after each line). "
            "Requires PLUMED built from the patched plumed2 submodule "
            "(see scripts/build_plumed_opes.sh); omit for stock PLUMED."
        ),
    )
    parser.add_argument("--ligand-smiles", type=str, default=None, help="chain:SMILES (e.g. B:CC...)")
    parser.add_argument("--mol-dir", type=Path, default=None, help="Boltz CCD mol dir for SMILES lookup")
    parser.add_argument(
        "--log-gpu-util",
        action="store_true",
        help=(
            "Background thread: poll nvidia-smi for GPU utilization (CUDA/OpenCL). "
            "Requires nvidia-smi on PATH; ignored for CPU platform."
        ),
    )
    parser.add_argument(
        "--gpu-util-interval",
        type=float,
        default=0.1,
        metavar="SEC",
        help="Wall-clock seconds between nvidia-smi samples when --log-gpu-util is set.",
    )
    parser.add_argument(
        "--gpu-util-out",
        type=Path,
        default=None,
        metavar="CSV",
        help="GPU utilization CSV (default: <out>/gpu_utilization.csv).",
    )
    parser.add_argument(
        "--opes-expanded-temp-max",
        type=float,
        default=None,
        metavar="K",
        help=(
            "If set, append PLUMED OPES_EXPANDED + ECV_MULTITHERMAL on ENERGY with "
            "TEMP_MAX=K (Kelvin; must exceed --temperature). Single-replica multithermal "
            "expanded ensemble (replica-exchange-like target without multiple replicas)."
        ),
    )
    parser.add_argument(
        "--opes-expanded-pace",
        type=int,
        default=50,
        help=(
            "PLUMED OPES_EXPANDED PACE in MD steps (only used when "
            "--opes-expanded-temp-max is set)."
        ),
    )
    parser.add_argument(
        "--oneopes-axis-p0-boltz",
        type=_parse_comma_int_list,
        default=None,
        help=(
            "Optional comma-separated Boltz global atom indices defining the deep-pocket "
            "anchor COM for --bias-cv oneopes. Omit both axis flags to auto-split pocket Cα."
        ),
    )
    parser.add_argument(
        "--oneopes-axis-p1-boltz",
        type=_parse_comma_int_list,
        default=None,
        help=(
            "Optional comma-separated Boltz indices for the pocket-mouth anchor COM "
            "(--bias-cv oneopes). Omit both axis flags to auto-split pocket Cα."
        ),
    )
    parser.add_argument(
        "--oneopes-contact-pairs-boltz",
        type=_parse_oneopes_contact_pairs_boltz,
        default=None,
        help=(
            "OneOPES CONTACTMAP pairs as 'prot-lig,prot-lig,...' using Boltz global "
            "0-based atom indices. If omitted, zips pocket heavy vs ligand atoms (up to 6)."
        ),
    )
    parser.add_argument(
        "--oneopes-hydration-boltz",
        type=_parse_comma_int_list,
        default=None,
        help=(
            "Optional Boltz global atom indices for auxiliary water COORDINATION OPES "
            "(each spot vs TIP3P oxygens). When set, water oxygens are taken from the "
            "hydrogenated OpenMM topology unless --oneopes-water-oxygen-plumed is set."
        ),
    )
    parser.add_argument(
        "--oneopes-water-oxygen-plumed",
        type=_parse_comma_int_list,
        default=None,
        help=(
            "Optional comma-separated PLUMED 1-based water oxygen indices for the WO "
            "GROUP (overrides auto-detection when --oneopes-hydration-boltz is set)."
        ),
    )
    parser.add_argument("--oneopes-water-pace", type=int, default=40_000)
    parser.add_argument("--oneopes-water-barrier", type=float, default=3.0)
    parser.add_argument("--oneopes-water-biasfactor", type=float, default=5.0)
    parser.add_argument("--oneopes-water-sigma", type=float, default=0.15)
    parser.add_argument(
        "--oneopes-auto-hydration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --bias-cv oneopes: if --oneopes-hydration-boltz is omitted, infer "
            "hydration spot Boltz indices (3D-RISM+blobs when possible, else geometric "
            "N/O interface atoms). Use --no-oneopes-auto-hydration to disable auxiliary "
            "water OPES unless you pass explicit --oneopes-hydration-boltz."
        ),
    )
    parser.add_argument(
        "--oneopes-hydration-max-sites",
        type=int,
        default=5,
        help="Cap on inferred hydration spots (--oneopes-auto-hydration).",
    )
    parser.add_argument(
        "--oneopes-hydration-use-3drism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Try AmberTools tleap+rism3d+metatwist before geometric fallback when "
            "auto-inferring hydration spots. Use --no-oneopes-hydration-use-3drism to "
            "skip RISM (faster, topology-independent)."
        ),
    )
    parser.add_argument(
        "--oneopes-hydration-min-density",
        type=float,
        default=0.5,
        help=(
            "metatwist --threshold for Laplacian blob picking when using 3D-RISM "
            "(larger = fewer blobs; Amber tutorial often uses ~0.5)."
        ),
    )
    return parser


def parse_opes_md_args(argv: list[str] | None = None) -> Namespace:
    """Parse CLI arguments for :func:`run_opes_md` (used by ``run_openmm_opes_md.py``)."""
    parser = build_opes_md_argument_parser()
    args = parser.parse_args(argv)

    if args.bias_cv == "oneopes" and args.opes_mode != "plumed":
        parser.error("--bias-cv oneopes requires --opes-mode plumed")

    if args.opes_expanded_temp_max is not None:
        if args.opes_mode != "plumed":
            parser.error("--opes-expanded-temp-max requires --opes-mode plumed")
        if float(args.opes_expanded_temp_max) <= float(args.temperature):
            parser.error(
                "--opes-expanded-temp-max must be strictly greater than --temperature"
            )

    if args.opes_mode == "plumed":
        from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

        assert_plumed_opes_metad_available()

    if int(args.oneopes_hydration_max_sites) < 1:
        parser.error("--oneopes-hydration-max-sites must be >= 1")

    return args


def build_plumed_extra_forces_callback(
    args: Namespace,
    *,
    structure: Any,
    indexer: Any,
    ref_coords: np.ndarray,
    n_s: int,
    pdb_path: Path,
    md_out_dir: Path,
    opes_wall_dist_resolved: float | None,
    plumed_context: dict[str, object],
    sigma: list[float],
    force_empty_oneopes_hydration: bool = False,
    oneopes_hydration_site_cap: int | None = None,
    opes_expanded_temp_max_override: float | None = None,
    opes_expanded_pace_override: int | None = None,
    plumed_state_rfile_override: Path | None = None,
    oneopes_protocol: str = "legacy_boltz",
    paper_replica_index: int | None = None,
) -> Callable[[Any, Any, Any, dict], None]:
    """Return the ``extra_forces`` callback used by :func:`run_opes_md` for PLUMED OPES.

    Parameters
    ----------
    md_out_dir
        Per-replica MD output root (e.g. ``<run>/rep000``). PLUMED files are written to
        ``md_out_dir / "plumed_opes.dat"``, ``md_out_dir / "plumed_rmsd_reference.pdb"``,
        and ``md_out_dir / "opes_states"``.
    force_empty_oneopes_hydration
        When ``True`` and ``--bias-cv oneopes``, omit auxiliary hydration ``OPES_METAD``
        blocks even if ``--oneopes-auto-hydration`` would otherwise add sites. Used for
        stratified replica 0 in OneOPES Hamiltonian exchange ladders.
    oneopes_hydration_site_cap
        When set, keep at most this many auxiliary hydration sites (ordered as inferred
        or as listed in ``--oneopes-hydration-boltz``). Ignored when
        *force_empty_oneopes_hydration* is ``True``.
    opes_expanded_temp_max_override, opes_expanded_pace_override
        Optional overrides for multithermal ``OPES_EXPANDED`` (replicas 4–7 in the
        literature ladder). When *opes_expanded_temp_max_override* is ``None``,
        ``args.opes_expanded_temp_max`` is used.
    plumed_state_rfile_override
        When set, passed as ``STATE_RFILE`` to the PLUMED deck instead of
        ``args.opes_restart``. Used for evaluator contexts that must restart from a
        copied ``STATE`` snapshot without mutating the production replica tree.
    oneopes_protocol
        ``legacy_boltz`` (default) or ``paper_host_guest`` for Febrer Martinez-style
        ``cyl_z`` + ``cosang`` CVs and Pefema auxiliary ladders.
    paper_replica_index
        Replica index ``0…7`` for paper stratified decks; defaults to ``0`` when unset.
    """

    def _add_plumed_force(system: Any, h_topology: Any, h_positions: Any, meta: dict) -> None:
        """Build atom selections after hydrogenation and inject PLUMED force."""
        import openmm.unit as u

        from genai_tps.simulation.openmm_boltz_bridge import (
            boltz_to_plumed_indices,
            build_openmm_indices_for_boltz_atoms,
        )
        from genai_tps.simulation.plumed_opes import (
            PaperHostGuestOpesDeckConfig,
            add_plumed_opes_to_system,
            default_oneopes_axis_boltz_indices,
            default_oneopes_contact_pairs_boltz,
            generate_paper_host_guest_plumed_opes_script_from_config,
            generate_plumed_opes_script,
            write_rmsd_reference_pdb,
        )

        out = md_out_dir.expanduser().resolve()
        opes_dir = out / "opes_states"

        h_pos_nm = np.asarray(h_positions.value_in_unit(u.nanometer), dtype=np.float64)
        omm_idx_cb = build_openmm_indices_for_boltz_atoms(
            structure,
            h_topology,
            ref_coords_angstrom=ref_coords,
            omm_positions_nm=h_pos_nm,
        )
        if int(np.min(omm_idx_cb)) < 0:
            raise RuntimeError(
                "[MD-OPES] Invalid Boltz→OpenMM index map before PLUMED setup "
                f"(min={int(np.min(omm_idx_cb))})."
            )

        rmsd_align_boltz = (
            indexer.pocket_ca_idx
            if len(indexer.pocket_ca_idx) >= 3
            else indexer.protein_ca_idx
        )
        if len(rmsd_align_boltz) < 3:
            raise RuntimeError(
                "[MD-OPES] PLUMED ligand RMSD requires at least three protein "
                "C-alpha alignment atoms."
            )
        if len(indexer.pocket_ca_idx) == 0:
            raise RuntimeError(
                "[MD-OPES] PLUMED ligand-pocket distance requires at least one "
                "pocket C-alpha atom."
            )
        if args.bias_cv == "3d" and len(indexer.pocket_heavy_idx) == 0:
            raise RuntimeError(
                "[MD-OPES] --bias-cv 3d requires pocket heavy atoms; "
                "increase --pocket-radius or check the reference structure."
            )
        if args.bias_cv == "oneopes":
            if len(indexer.pocket_ca_idx) < 2:
                raise RuntimeError(
                    "[MD-OPES] --bias-cv oneopes requires at least two pocket Cα atoms."
                )
            if args.oneopes_contact_pairs_boltz is None and len(indexer.pocket_heavy_idx) == 0:
                raise RuntimeError(
                    "[MD-OPES] --bias-cv oneopes needs pocket heavy atoms for default "
                    "CONTACTMAP pairs unless --oneopes-contact-pairs-boltz is set; "
                    "increase --pocket-radius or supply explicit pairs."
                )

        ligand_plumed_idx = boltz_to_plumed_indices(indexer.ligand_idx, omm_idx_cb)
        pocket_plumed_idx = boltz_to_plumed_indices(indexer.pocket_ca_idx, omm_idx_cb)
        pocket_heavy_plumed_idx = (
            boltz_to_plumed_indices(indexer.pocket_heavy_idx, omm_idx_cb)
            if args.bias_cv in ("3d", "oneopes")
            else None
        )
        align_plumed_idx = boltz_to_plumed_indices(rmsd_align_boltz, omm_idx_cb)
        solute_boltz = np.unique(
            np.concatenate([indexer.protein_idx, indexer.ligand_idx])
        )
        whole_molecule_plumed_idx = sorted(
            boltz_to_plumed_indices(solute_boltz, omm_idx_cb)
        )
        ref_pdb = out / "plumed_rmsd_reference.pdb"
        script_path = out / "plumed_opes.dat"
        write_rmsd_reference_pdb(
            h_topology,
            h_positions,
            ligand_plumed_idx,
            align_plumed_idx,
            ref_pdb,
        )
        expanded_state_rfile: Path | None = None
        opes_expanded_temp_max_resolved = (
            opes_expanded_temp_max_override
            if opes_expanded_temp_max_override is not None
            else args.opes_expanded_temp_max
        )
        opes_expanded_pace_resolved = (
            int(opes_expanded_pace_override)
            if opes_expanded_pace_override is not None
            else int(args.opes_expanded_pace)
        )
        if opes_expanded_temp_max_resolved is not None and args.opes_restart is not None:
            cand_exp = opes_dir / "STATE_EXPANDED"
            if cand_exp.is_file():
                expanded_state_rfile = cand_exp

        state_rfile_resolved = (
            plumed_state_rfile_override
            if plumed_state_rfile_override is not None
            else args.opes_restart
        )

        oneopes_kw: dict = {}
        protocol = str(oneopes_protocol).replace("-", "_")
        if args.bias_cv == "oneopes":
            assert pocket_heavy_plumed_idx is not None
            p_axes = args.oneopes_axis_p0_boltz
            q_axes = args.oneopes_axis_p1_boltz
            if (p_axes is None) ^ (q_axes is None):
                raise RuntimeError(
                    "[MD-OPES] Supply both --oneopes-axis-p0-boltz and "
                    "--oneopes-axis-p1-boltz, or omit both for automatic pocket Cα split."
                )
            if p_axes is None:
                bp0, bp1 = default_oneopes_axis_boltz_indices(
                    indexer.pocket_ca_idx, ref_coords[:n_s], indexer.ligand_idx
                )
            else:
                bp0 = np.asarray(p_axes, dtype=np.int64)
                bp1 = np.asarray(q_axes, dtype=np.int64)
            p0_pl = boltz_to_plumed_indices(bp0.tolist(), omm_idx_cb)
            p1_pl = boltz_to_plumed_indices(bp1.tolist(), omm_idx_cb)
            if protocol == "paper_host_guest":
                pap0 = np.asarray(args.paper_oneopes_ligand_axis_p0_boltz, dtype=np.int64)
                pap1 = np.asarray(args.paper_oneopes_ligand_axis_p1_boltz, dtype=np.int64)
                la0_pl = boltz_to_plumed_indices(pap0.tolist(), omm_idx_cb)
                la1_pl = boltz_to_plumed_indices(pap1.tolist(), omm_idx_cb)
                aux_guest_b = list(args.paper_oneopes_aux_guest_boltz)
                aux_guest_pl = tuple(
                    int(boltz_to_plumed_indices([int(x)], omm_idx_cb)[0]) for x in aux_guest_b
                )
                if args.oneopes_water_oxygen_plumed is not None and len(
                    args.oneopes_water_oxygen_plumed
                ) > 0:
                    wo_paper = [int(x) for x in args.oneopes_water_oxygen_plumed]
                else:
                    wo_paper = collect_water_oxygen_plumed_1based(h_topology)
                if not wo_paper:
                    raise RuntimeError(
                        "[MD-OPES] paper_host_guest requires explicit solvent: pass "
                        "--oneopes-water-oxygen-plumed or use a solvated topology."
                    )
                sig_parts = [float(x.strip()) for x in str(args.opes_sigma).split(",") if x.strip()]
                if len(sig_parts) != 2:
                    raise RuntimeError(
                        "[MD-OPES] paper_host_guest expects two --opes-sigma values "
                        "(main cyl_z, cosang kernel widths)."
                    )
                ridx = int(paper_replica_index) if paper_replica_index is not None else 0
                paper_cfg = PaperHostGuestOpesDeckConfig(
                    ligand_plumed_idx=ligand_plumed_idx,
                    pocket_ca_plumed_idx=pocket_plumed_idx,
                    rmsd_reference_pdb=ref_pdb,
                    sigma_main=(sig_parts[0], sig_parts[1]),
                    biasfactor=float(args.opes_biasfactor),
                    temperature=float(args.temperature),
                    save_opes_every=int(args.save_opes_every),
                    progress_every=int(args.progress_every),
                    out_dir=opes_dir,
                    replica_index=ridx,
                    funnel_axis_p0_plumed_idx=p0_pl,
                    funnel_axis_p1_plumed_idx=p1_pl,
                    ligand_axis_p0_plumed_idx=la0_pl,
                    ligand_axis_p1_plumed_idx=la1_pl,
                    auxiliary_guest_plumed_idx=aux_guest_pl,
                    water_oxygen_plumed_idx=wo_paper,
                    state_rfile=state_rfile_resolved,
                    whole_molecule_plumed_idx=whole_molecule_plumed_idx,
                    kernel_cutoff=args.opes_kernel_cutoff,
                    nlist_parameters=args.opes_nlist_parameters,
                    print_colvar_heavy_flush=bool(args.plumed_colvar_heavy_flush),
                    use_pbc=True,
                    main_pace=int(getattr(args, "paper_oneopes_main_pace", 10_000)),
                    main_barrier_kjmol=float(getattr(args, "paper_oneopes_main_barrier", 100.0)),
                    auxiliary_pace=int(getattr(args, "paper_oneopes_aux_pace", 20_000)),
                    auxiliary_barrier_kjmol=float(
                        getattr(args, "paper_oneopes_aux_barrier", 3.0)
                    ),
                    multithermal_pace=int(getattr(args, "paper_oneopes_multithermal_pace", 100)),
                    opes_expanded_temp_max=float(opes_expanded_temp_max_resolved)
                    if opes_expanded_temp_max_resolved is not None
                    else None,
                    opes_expanded_state_rfile=expanded_state_rfile,
                )
                script = generate_paper_host_guest_plumed_opes_script_from_config(paper_cfg)
                script_path.write_text(script, encoding="utf-8")
                force, force_index = add_plumed_opes_to_system(
                    system,
                    script,
                    temperature=args.temperature,
                    force_group=args.plumed_force_group,
                    restart=state_rfile_resolved is not None,
                )
                plumed_context.update(
                    {
                        "omm_idx": omm_idx_cb,
                        "force": force,
                        "force_index": force_index,
                        "force_group": int(args.plumed_force_group),
                        "script_path": script_path,
                        "reference_pdb": ref_pdb,
                    }
                )
                meta["plumed_force_index"] = force_index
                meta["plumed_force_group"] = int(args.plumed_force_group)
                meta["plumed_script"] = str(script_path)
                meta["plumed_reference_pdb"] = str(ref_pdb)
                meta["oneopes_cv_params"] = {
                    "oneopes_protocol": "paper_host_guest",
                    "paper_replica_index": ridx,
                    "funnel_axis_p0_boltz": np.asarray(bp0, dtype=np.int64).tolist(),
                    "funnel_axis_p1_boltz": np.asarray(bp1, dtype=np.int64).tolist(),
                    "ligand_axis_p0_boltz": pap0.tolist(),
                    "ligand_axis_p1_boltz": pap1.tolist(),
                    "aux_guest_boltz": [int(x) for x in aux_guest_b],
                }
                return

            if args.oneopes_contact_pairs_boltz is not None:
                pairs_b = args.oneopes_contact_pairs_boltz
            else:
                pairs_b = default_oneopes_contact_pairs_boltz(
                    indexer.pocket_heavy_idx, indexer.ligand_idx, max_pairs=6
                )
            cmap_pairs: list[tuple[int, int]] = []
            for pr, li in pairs_b:
                pp_i = boltz_to_plumed_indices([int(pr)], omm_idx_cb)[0]
                li_i = boltz_to_plumed_indices([int(li)], omm_idx_cb)[0]
                cmap_pairs.append((int(pp_i), int(li_i)))
            if force_empty_oneopes_hydration:
                hydr_b = []
            else:
                hydr_explicit = args.oneopes_hydration_boltz
                if hydr_explicit is not None and len(hydr_explicit) > 0:
                    hydr_b = [int(x) for x in hydr_explicit]
                elif bool(args.oneopes_auto_hydration):
                    from genai_tps.simulation.hydration_site_inference import (
                        default_oneopes_hydration_boltz_indices,
                    )

                    hydr_b = default_oneopes_hydration_boltz_indices(
                        ref_coords[:n_s],
                        indexer,
                        pdb_path=pdb_path,
                        max_sites=int(args.oneopes_hydration_max_sites),
                        use_3drism=bool(args.oneopes_hydration_use_3drism),
                        rism_min_density=float(args.oneopes_hydration_min_density),
                    )
                    if hydr_b:
                        print(
                            "[MD-OPES] OneOPES auto hydration spots (Boltz global indices): "
                            f"{hydr_b}",
                            flush=True,
                        )
                else:
                    hydr_b = []
            if (
                not force_empty_oneopes_hydration
                and oneopes_hydration_site_cap is not None
                and hydr_b
            ):
                cap = max(0, int(oneopes_hydration_site_cap))
                hydr_b = hydr_b[:cap]
            hydr_pl = (
                boltz_to_plumed_indices(hydr_b, omm_idx_cb) if hydr_b else []
            )
            wo_pl: list[int] | None = None
            if hydr_b:
                if args.oneopes_water_oxygen_plumed is not None and len(
                    args.oneopes_water_oxygen_plumed
                ) > 0:
                    wo_pl = [int(x) for x in args.oneopes_water_oxygen_plumed]
                else:
                    wo_pl = collect_water_oxygen_plumed_1based(h_topology)
                if not wo_pl:
                    raise RuntimeError(
                        "[MD-OPES] OneOPES hydration CVs requested but no water "
                        "oxygens found in topology; pass --oneopes-water-oxygen-plumed."
                    )
            oneopes_kw = {
                "oneopes_axis_p0_plumed_idx": p0_pl,
                "oneopes_axis_p1_plumed_idx": p1_pl,
                "oneopes_contactmap_pairs_plumed": cmap_pairs,
                "oneopes_hydration_spot_plumed_idx": hydr_pl if hydr_pl else None,
                "water_oxygen_plumed_idx": wo_pl,
                "oneopes_water_pace": int(args.oneopes_water_pace),
                "oneopes_water_barrier": float(args.oneopes_water_barrier),
                "oneopes_water_biasfactor": float(args.oneopes_water_biasfactor),
                "oneopes_water_sigma": float(args.oneopes_water_sigma),
            }
            meta["oneopes_cv_params"] = {
                "axis_p0_boltz": np.asarray(bp0, dtype=np.int64).tolist(),
                "axis_p1_boltz": np.asarray(bp1, dtype=np.int64).tolist(),
                "ligand_boltz": np.asarray(indexer.ligand_idx, dtype=np.int64).tolist(),
                "contact_pairs_boltz": [(int(pr), int(li)) for pr, li in pairs_b],
                "contactmap_switch_r0": 5.5,
                "contactmap_switch_d0": 0.0,
                "contactmap_switch_nn": 4,
                "contactmap_switch_mm": 10,
            }

        script = generate_plumed_opes_script(
            ligand_plumed_idx=ligand_plumed_idx,
            pocket_ca_plumed_idx=pocket_plumed_idx,
            rmsd_reference_pdb=ref_pdb,
            sigma=sigma,
            pace=args.deposit_pace,
            barrier=args.opes_barrier,
            biasfactor=args.opes_biasfactor,
            temperature=args.temperature,
            save_opes_every=args.save_opes_every,
            progress_every=args.progress_every,
            out_dir=opes_dir,
            state_rfile=state_rfile_resolved,
            kernel_cutoff=args.opes_kernel_cutoff,
            nlist_parameters=args.opes_nlist_parameters,
            print_colvar_heavy_flush=bool(args.plumed_colvar_heavy_flush),
            cv_mode=args.bias_cv,
            pocket_heavy_plumed_idx=pocket_heavy_plumed_idx,
            coordination_r0=float(args.coordination_r0),
            opes_variant="explore" if args.opes_explore else "metad",
            upper_wall_dist=opes_wall_dist_resolved,
            upper_wall_kappa=float(args.opes_wall_kappa),
            use_pbc=True,
            whole_molecule_plumed_idx=whole_molecule_plumed_idx,
            opes_expanded_temp_max=opes_expanded_temp_max_resolved,
            opes_expanded_pace=opes_expanded_pace_resolved,
            opes_expanded_state_rfile=expanded_state_rfile,
            **oneopes_kw,
        )
        script_path.write_text(script, encoding="utf-8")
        force, force_index = add_plumed_opes_to_system(
            system,
            script,
            temperature=args.temperature,
            force_group=args.plumed_force_group,
            restart=state_rfile_resolved is not None,
        )
        plumed_context.update(
            {
                "omm_idx": omm_idx_cb,
                "force": force,
                "force_index": force_index,
                "force_group": int(args.plumed_force_group),
                "script_path": script_path,
                "reference_pdb": ref_pdb,
            }
        )
        meta["plumed_force_index"] = force_index
        meta["plumed_force_group"] = int(args.plumed_force_group)
        meta["plumed_script"] = str(script_path)
        meta["plumed_reference_pdb"] = str(ref_pdb)

    return _add_plumed_force


def run_opes_md(
    args: argparse.Namespace,
    *,
    md_out_dir: Path | None = None,
    plumed_factory_extra_kwargs: dict[str, Any] | None = None,
    stop_after_initialization: bool = False,
    return_after_initialization: bool = False,
) -> Any:
    """Run OPES-biased OpenMM MD.

    Parameters
    ----------
    md_out_dir
        Optional override for the run output root (defaults to ``args.out``).
        Used by Hamiltonian replica exchange drivers so each replica writes under
        ``<ensemble>/repNNN/``.
    plumed_factory_extra_kwargs
        Forwarded into :func:`build_plumed_extra_forces_callback` (e.g. stratified
        hydration caps).
    stop_after_initialization
        When ``True``, return immediately after system build, minimization, and
        coordinate validation (skips the production MD loops). Intended for PLUMED
        deck dry-runs.
    return_after_initialization
        When ``True``, return a dictionary with ``sim``, ``meta``, ``plumed_context``,
        ``structure``, ``indexer``, ``ref_coords``, ``n_s``, ``pdb_path``, ``out``,
        ``opes_wall_dist_resolved``, and ``kbt_kjmol`` instead of entering the MD loops.
        Incompatible with *stop_after_initialization* (mutually exclusive).
    """
    from genai_tps.backends.boltz.collective_variables import PoseCVIndexer
    from genai_tps.io.boltz_npz_export import coords_frame_from_npz, load_topo, npz_to_pdb
    from genai_tps.simulation.openmm_boltz_bridge import (
        boltz_to_plumed_indices,
        build_openmm_indices_for_boltz_atoms,
        load_build_md_simulation_from_pdb,
    )
    from genai_tps.utils.compute_device import openmm_device_index_properties
    from genai_tps.simulation.gpu_util_csv_logger import GpuUtilCsvLogger

    out = (md_out_dir if md_out_dir is not None else args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wdsm_dir = out / "wdsm_samples"
    wdsm_dir.mkdir(parents=True, exist_ok=True)
    opes_dir = out / "opes_states"
    opes_dir.mkdir(parents=True, exist_ok=True)

    structure, n_struct = load_topo(args.topo_npz)
    n_s = int(n_struct)
    print(f"[MD-OPES] Loaded topology: {n_s} atoms", flush=True)

    pdb_path = out / "initial_structure.pdb"
    frame_npz = args.frame_npz or args.topo_npz
    with np.load(str(frame_npz)) as ref_data:
        ref_coords = coords_frame_from_npz(
            ref_data, frame_idx=args.frame_idx, n_struct=n_s
        )
    _log_coord_stats_np("checkpoint_after_reference_npz", ref_coords, unit_label="Å")
    npz_to_pdb(frame_npz, structure, n_struct, pdb_path, frame_idx=args.frame_idx)
    print(f"[MD-OPES] Wrote initial PDB: {pdb_path}", flush=True)
    if pdb_path.is_file():
        print(f"[MD-OPES] checkpoint_after_pdb_write: path_ok size={pdb_path.stat().st_size} B", flush=True)
    indexer = PoseCVIndexer(structure, ref_coords, pocket_radius=args.pocket_radius)
    print(
        f"[MD-OPES] PoseCVIndexer: {len(indexer.ligand_idx)} ligand atoms, "
        f"{len(indexer.pocket_ca_idx)} pocket Cas, "
        f"{len(indexer.pocket_heavy_idx)} pocket heavy atoms",
        flush=True,
    )

    opes_wall_dist_resolved: float | None = None
    if args.opes_mode == "plumed":
        if args.opes_wall_dist is not None:
            opes_wall_dist_resolved = float(args.opes_wall_dist)
        elif args.bias_cv == "oneopes":
            from genai_tps.simulation.plumed_opes import (
                compute_oneopes_pp_ext_angstrom,
                default_oneopes_axis_boltz_indices,
            )

            p_axes = args.oneopes_axis_p0_boltz
            q_axes = args.oneopes_axis_p1_boltz
            if (p_axes is None) ^ (q_axes is None):
                raise ValueError(
                    "Supply both --oneopes-axis-p0-boltz and --oneopes-axis-p1-boltz, "
                    "or omit both for automatic pocket Cα median split."
                )
            if p_axes is None:
                bp0, bp1 = default_oneopes_axis_boltz_indices(
                    indexer.pocket_ca_idx, ref_coords[:n_s], indexer.ligand_idx
                )
            else:
                bp0 = np.asarray(p_axes, dtype=np.int64)
                bp1 = np.asarray(q_axes, dtype=np.int64)
            ext0 = compute_oneopes_pp_ext_angstrom(
                ref_coords[:n_s],
                axis_p0_boltz=bp0.tolist(),
                axis_p1_boltz=bp1.tolist(),
                ligand_boltz=indexer.ligand_idx.tolist(),
            )
            opes_wall_dist_resolved = default_opes_upper_wall_dist_angstrom(ext0)
            print(
                f"[MD-OPES] Adaptive --opes-wall-dist (oneopes pp.ext): initial extension="
                f"{ext0:.4f} Å + margin={_DEFAULT_OPES_WALL_MARGIN_ANGSTROM:g} Å "
                f"-> AT={opes_wall_dist_resolved:.4f} Å",
                flush=True,
            )
        else:
            from types import SimpleNamespace

            from genai_tps.backends.boltz.collective_variables import (
                ligand_pocket_distance,
            )

            x0 = torch.from_numpy(ref_coords[:n_s]).float().unsqueeze(0)
            snap0 = SimpleNamespace(_tensor_coords_gpu=x0)
            d0 = float(ligand_pocket_distance(snap0, indexer))
            opes_wall_dist_resolved = default_opes_upper_wall_dist_angstrom(d0)
            print(
                f"[MD-OPES] Adaptive --opes-wall-dist: initial ligand-pocket COM "
                f"distance={d0:.4f} Å + margin={_DEFAULT_OPES_WALL_MARGIN_ANGSTROM:g} Å "
                f"-> AT={opes_wall_dist_resolved:.4f} Å",
                flush=True,
            )

    ligand_smiles = None
    if args.ligand_smiles:
        parts = args.ligand_smiles.split(":", 1)
        ligand_smiles = {parts[0]: parts[1]}
    elif args.mol_dir:
        from genai_tps.simulation.openmm_cv import _detect_ligand_smiles

        ligand_smiles = _detect_ligand_smiles(structure, args.mol_dir)

    kbt_kjmol = 8.314e-3 * args.temperature
    sigma = [float(s) for s in args.opes_sigma.split(",")]
    if args.bias_cv == "3d":
        expected_sigmas = 3
    else:
        expected_sigmas = 2
    if len(sigma) != expected_sigmas:
        raise ValueError(
            f"--opes-sigma must contain {expected_sigmas} comma-separated values "
            f"for --bias-cv {args.bias_cv!r} (got {len(sigma)})."
        )

    opes = None
    if args.opes_mode == "observer":
        from genai_tps.simulation import OPESBias

        if args.opes_restart:
            print(f"[MD-OPES] Restarting observer OPES from {args.opes_restart}", flush=True)
            opes = OPESBias.load_state(args.opes_restart)
        else:
            opes = OPESBias(
                ndim=expected_sigmas,
                kbt=kbt_kjmol,
                barrier=args.opes_barrier,
                biasfactor=args.opes_biasfactor,
                pace=args.deposit_pace,
                fixed_sigma=np.array(sigma),
            )
        print(
            f"[MD-OPES] Observer OPES: barrier={opes.barrier:.1f} "
            f"biasfactor={opes.biasfactor:.1f} kbt={opes.kbt:.3f} kJ/mol "
            f"sigma={sigma}",
            flush=True,
        )
    else:
        print(
            f"[MD-OPES] PLUMED OPES: barrier={args.opes_barrier:.1f} "
            f"biasfactor={args.opes_biasfactor:.1f} kbt={kbt_kjmol:.3f} kJ/mol "
            f"bias_cv={args.bias_cv} explore={bool(args.opes_explore)} sigma={sigma}",
            flush=True,
        )

    if args.opes_mode == "plumed" and args.opes_explore and opes_wall_dist_resolved is not None:
        print(
            "[MD-OPES] PLUMED: UPPER_WALLS uses EXTRA_BIAS, which is only available "
            "for OPES_METAD (not OPES_METAD_EXPLORE); the generated deck uses OPES_METAD.",
            flush=True,
        )
    if args.opes_mode == "plumed" and args.opes_expanded_temp_max is not None:
        print(
            f"[MD-OPES] PLUMED: OPES_EXPANDED multithermal TEMP_MAX="
            f"{float(args.opes_expanded_temp_max):.1f} K, PACE={int(args.opes_expanded_pace)}",
            flush=True,
        )
    print(
        f"[MD-OPES] Building OpenMM simulation (AMBER14 + explicit TIP3P, PME, "
        f"{args.platform})...",
        flush=True,
    )
    build_md = load_build_md_simulation_from_pdb()
    plumed_context: dict[str, object] = {}
    _factory_kw: dict[str, Any] = dict(
        args=args,
        structure=structure,
        indexer=indexer,
        ref_coords=ref_coords,
        n_s=n_s,
        pdb_path=pdb_path,
        md_out_dir=out,
        opes_wall_dist_resolved=opes_wall_dist_resolved,
        plumed_context=plumed_context,
        sigma=sigma,
        force_empty_oneopes_hydration=False,
        oneopes_hydration_site_cap=None,
        opes_expanded_temp_max_override=None,
        opes_expanded_pace_override=None,
        plumed_state_rfile_override=None,
        oneopes_protocol=str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace(
            "-", "_"
        ),
        paper_replica_index=None,
    )
    if plumed_factory_extra_kwargs:
        _factory_kw.update(plumed_factory_extra_kwargs)
    _add_plumed_force = build_plumed_extra_forces_callback(**_factory_kw)

    md_boltz_pose: dict = {}
    if args.mol_dir is not None:
        md_boltz_pose = {
            "boltz_structure": structure,
            "boltz_coords_angstrom": ref_coords,
            "boltz_mol_dir": args.mol_dir.expanduser().resolve(),
            "ligand_pose_policy": "boltz_first",
        }
    omm_props = openmm_device_index_properties(
        args.platform, args.openmm_device_index
    )
    sim, meta = build_md(
        pdb_path,
        platform_name=args.platform,
        temperature_k=args.temperature,
        ligand_smiles=ligand_smiles,
        extra_forces=_add_plumed_force if args.opes_mode == "plumed" else None,
        platform_properties=omm_props if omm_props else None,
        **md_boltz_pose,
    )
    print(f"[MD-OPES] Platform: {meta['platform_used']}", flush=True)

    if args.opes_mode == "plumed":
        colvar_p = opes_dir / "COLVAR"
        if args.bias_cv == "oneopes":
            _op = str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace("-", "_")
            if _op == "paper_host_guest":
                cv_fes = "cyl_z,cosang"
            else:
                cv_fes = "pp.proj,cmap"
            grid_hint = "100,100"
            fes_name = "fes_reweighted_2d.dat"
        elif args.bias_cv == "3d":
            cv_fes = "lig_rmsd,lig_dist,lig_contacts"
            grid_hint = "40,40,40"
            fes_name = "fes_reweighted_3d.dat"
        else:
            cv_fes = "lig_rmsd,lig_dist"
            grid_hint = "100,100"
            fes_name = "fes_reweighted_2d.dat"
        print(
            "[MD-OPES] PLUMED COLVAR (PRINT every "
            f"{int(args.progress_every)} steps) -> {colvar_p}",
            flush=True,
        )
        print(
            "[MD-OPES] FES reweighting (after data accumulates; outfile next to COLVAR): "
            f"cv_names={cv_fes!r}  sigma={args.opes_sigma!r}  bias_name=\"opes.bias\"  "
            f"temperature_k={float(args.temperature)}  grid_bin={grid_hint!r}  "
            f"-> {fes_name}",
            flush=True,
        )
        print(
            "[MD-OPES] Note: COLVAR column opes.nker is the merged kernel count in CV "
            "space; reweighted FES KDEs are smooth even when KERNELS lists many "
            "historical depositions.",
            flush=True,
        )
        print(
            "[MD-OPES] Python API: "
            "genai_tps.simulation.plumed_colvar_fes.run_fes_from_reweighting_script("
            f"colvar_path=Path(...), outfile=Path(...)/{fes_name}, ...)",
            flush=True,
        )
        print(
            "[MD-OPES] During the run, latest plain COLVAR row is echoed on each "
            f"progress line (every {int(args.progress_every)} MD steps).",
            flush=True,
        )

    state0 = sim.context.getState(getPositions=True)
    pos_nm = state0.getPositions(asNumpy=True)
    _log_coord_stats_np("checkpoint_after_build_set_positions", pos_nm, unit_label="nm")
    _diagnostic_energy_and_large_forces(sim, "after_build_set_positions")

    if args.opes_mode == "plumed":
        if "omm_idx" not in plumed_context:
            raise RuntimeError("[MD-OPES] PLUMED setup did not produce an atom map.")
        omm_idx = np.asarray(plumed_context["omm_idx"], dtype=np.int64)
    else:
        omm_idx = build_openmm_indices_for_boltz_atoms(
            structure,
            sim.topology,
            ref_coords_angstrom=ref_coords,
            omm_positions_nm=pos_nm,
        )
    if int(np.min(omm_idx)) < 0:
        raise RuntimeError(
            "[MD-OPES] Invalid Boltz→OpenMM index map (negative OpenMM index). "
            f"min(omm_idx)={int(np.min(omm_idx))}"
        )
    n_boltz = int(structure.atoms.shape[0])

    if args.minimize_steps > 0:
        print(f"[MD-OPES] Minimizing ({args.minimize_steps} steps)...", flush=True)
        sim.minimizeEnergy(maxIterations=args.minimize_steps)

    state_post = sim.context.getState(getPositions=True)
    pos_post_nm = state_post.getPositions(asNumpy=True)
    if args.minimize_steps > 0:
        _log_coord_stats_np("checkpoint_after_minimize", pos_post_nm, unit_label="nm")
        _require_finite_positions(pos_post_nm, context_msg="energy minimization")
        _diagnostic_energy_and_large_forces(sim, "after_minimize")
    else:
        _log_coord_stats_np("checkpoint_skip_minimize", pos_post_nm, unit_label="nm")
        _require_finite_positions(pos_post_nm, context_msg="initial context (minimize_steps=0)")

    # WDSM shards (and CV tensors) use Boltz-ordered **solute-only** coordinates:
    # explicit TIP3P and ions exist only in the OpenMM topology.  ``n_boltz`` is
    # the Boltz-processed heavy-atom count from the topology NPZ; ``omm_idx`` maps
    # each Boltz row to one OpenMM particle.  Bulk solvent is never written to
    # ``wdsm_step_*.npz``.
    def get_coords_boltz_order(state=None):
        import openmm.unit as u

        if state is None:
            state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(u.angstrom)
        coords = np.zeros((n_boltz, 3), dtype=np.float32)
        for i in range(n_boltz):
            coords[i] = pos[int(omm_idx[i])]
        return coords

    _probe_boltz = get_coords_boltz_order()
    _log_coord_stats_np("checkpoint_first_boltz_ordered_coords", _probe_boltz, unit_label="Å")
    _require_finite_positions(
        _probe_boltz,
        context_msg="first Boltz-ordered coordinate extraction",
    )
    del _probe_boltz

    oneopes_cv_params = meta.get("oneopes_cv_params")

    def compute_cv(coords):
        if oneopes_cv_params is not None:
            from genai_tps.simulation.plumed_opes import (
                compute_oneopes_pp_proj_cmap_from_boltz_coords,
            )

            p = oneopes_cv_params
            return compute_oneopes_pp_proj_cmap_from_boltz_coords(
                coords,
                axis_p0_boltz=p["axis_p0_boltz"],
                axis_p1_boltz=p["axis_p1_boltz"],
                ligand_boltz=p["ligand_boltz"],
                contact_pairs_boltz=p["contact_pairs_boltz"],
                contactmap_switch_r0=float(p["contactmap_switch_r0"]),
                contactmap_switch_d0=float(p["contactmap_switch_d0"]),
                contactmap_switch_nn=int(p["contactmap_switch_nn"]),
                contactmap_switch_mm=int(p["contactmap_switch_mm"]),
            )
        x = torch.from_numpy(coords[:n_s]).float().unsqueeze(0)
        from types import SimpleNamespace

        snap = SimpleNamespace(_tensor_coords_gpu=x)
        from genai_tps.backends.boltz.collective_variables import (
            ligand_pocket_distance,
            ligand_pose_rmsd,
            protein_ligand_contacts,
        )

        rmsd = float(ligand_pose_rmsd(snap, indexer))
        dist = float(ligand_pocket_distance(snap, indexer))
        if args.bias_cv == "3d":
            contacts = float(
                protein_ligand_contacts(snap, indexer, r0=float(args.coordination_r0))
            )
            return np.array([rmsd, dist, contacts], dtype=np.float64)
        return np.array([rmsd, dist], dtype=np.float64)

    if args.bias_cv == "3d":
        _observer_bias_cv = "ligand_rmsd,ligand_pocket_dist,ligand_contacts"
        _observer_bias_names = [
            "ligand_rmsd",
            "ligand_pocket_dist",
            "ligand_contacts",
        ]
    else:
        _observer_bias_cv = "ligand_rmsd,ligand_pocket_dist"
        _observer_bias_names = ["ligand_rmsd", "ligand_pocket_dist"]

    print(f"\n{'='*60}")
    print(f"  OpenMM OPES-MD: {args.n_steps:,} steps ({args.n_steps * 2e-6:.1f} ns at 2fs)")
    print(f"  Save every {args.save_every} steps | Deposit every {args.deposit_pace} steps")
    print(f"  Temperature: {args.temperature} K")
    print(f"{'='*60}\n", flush=True)

    cv_log = []
    n_saved = 0
    t0 = time.time()

    gpu_logger: GpuUtilCsvLogger | None = None
    gpu_csv: Path | None = None
    if args.log_gpu_util:
        if args.platform == "CPU":
            print(
                "[MD-OPES] --log-gpu-util ignored for --platform CPU.",
                flush=True,
            )
        elif shutil.which("nvidia-smi") is None:
            print(
                "[MD-OPES] --log-gpu-util: nvidia-smi not found on PATH; skipping.",
                flush=True,
            )
        else:
            gpu_ord = (
                int(args.openmm_device_index)
                if args.openmm_device_index is not None
                else 0
            )
            gpu_csv = (
                args.gpu_util_out.expanduser().resolve()
                if args.gpu_util_out is not None
                else (out / "gpu_utilization.csv")
            )
            gpu_logger = GpuUtilCsvLogger(
                gpu_csv,
                float(args.gpu_util_interval),
                gpu_ord,
            )
            if gpu_logger.start():
                print(
                    f"[MD-OPES] GPU util log: device {gpu_ord}, every "
                    f"{float(args.gpu_util_interval):.3f}s -> {gpu_csv}",
                    flush=True,
                )
            else:
                gpu_logger = None
                gpu_csv = None

    if stop_after_initialization and return_after_initialization:
        raise ValueError("stop_after_initialization and return_after_initialization are mutually exclusive.")

    if stop_after_initialization or return_after_initialization:
        if gpu_logger is not None:
            gpu_logger.stop()
        if stop_after_initialization:
            print("[MD-OPES] stop_after_initialization: skipping MD loops.", flush=True)
            with open(out / "cv_log.json", "w") as f:
                json.dump([], f)
            with open(out / "md_summary.json", "w") as f:
                json.dump(
                    {
                        "total_steps": 0,
                        "n_saved": 0,
                        "stopped": "after_initialization",
                        "md_out": str(out),
                    },
                    f,
                    indent=2,
                )
            return None
        print("[MD-OPES] return_after_initialization: returning simulation bundle.", flush=True)
        return {
            "sim": sim,
            "meta": meta,
            "plumed_context": plumed_context,
            "structure": structure,
            "indexer": indexer,
            "ref_coords": ref_coords,
            "n_s": n_s,
            "pdb_path": pdb_path,
            "out": out,
            "opes_wall_dist_resolved": opes_wall_dist_resolved,
            "kbt_kjmol": kbt_kjmol,
        }

    if args.opes_mode == "observer":
        if opes is None:
            raise RuntimeError("[MD-OPES] Observer mode did not initialize OPESBias.")
        _observer_periods = (
            int(args.deposit_pace),
            int(args.save_every),
            int(args.progress_every),
            int(args.save_opes_every),
        )
        completed = 0
        while completed < args.n_steps:
            next_at = _next_observer_event_step(
                completed, int(args.n_steps), _observer_periods
            )
            batch = int(next_at - completed)
            assert batch > 0
            sim.step(batch)
            completed = int(next_at)

            coords = None
            cv = None

            def _ensure_cv_once() -> None:
                nonlocal coords, cv
                if cv is not None:
                    return
                st = sim.context.getState(getPositions=True)
                coords = get_coords_boltz_order(st)
                cv = compute_cv(coords)

            if completed % args.deposit_pace == 0:
                _ensure_cv_once()
                assert cv is not None
                opes.update(cv, completed)

            if completed % args.save_every == 0:
                _ensure_cv_once()
                assert cv is not None
                logw = float(opes.evaluate(cv)) / opes.kbt

                n_saved += 1
                npz_path = wdsm_dir / f"wdsm_step_{n_saved:08d}.npz"
                np.savez_compressed(
                    str(npz_path),
                    coords=coords,
                    cv=cv,
                    logw=np.float64(logw),
                    md_step=np.int64(completed),
                )
                cv_log.append({"step": completed, "cv": cv.tolist(), "logw": logw})

            if completed % args.save_opes_every == 0:
                opes.save_state(
                    opes_dir / f"opes_state_{completed:010d}.json",
                    bias_cv=_observer_bias_cv,
                    bias_cv_names=_observer_bias_names,
                )

            if completed % args.progress_every == 0:
                _ensure_cv_once()
                assert cv is not None
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (args.n_steps - completed) / rate if rate > 0 else float("inf")
                cv_str = ", ".join(f"{v:.3f}" for v in cv)
                print(
                    f"[MD-OPES] step {completed:>10,}/{args.n_steps:,} | "
                    f"{rate:.0f} steps/s | ETA {eta/60:.1f}min | "
                    f"CV=[{cv_str}] | "
                    f"kernels={opes.n_kernels} | saved={n_saved}",
                    flush=True,
                )

        opes.save_state(
            out / "opes_state_final.json",
            bias_cv=_observer_bias_cv,
            bias_cv_names=_observer_bias_names,
        )
    else:
        import openmm.unit as u

        force_group = int(plumed_context["force_group"])
        groups = {force_group}
        step = 0
        next_progress = int(args.progress_every)
        while step < args.n_steps:
            chunk = min(int(args.save_every), args.n_steps - step)
            sim.step(chunk)
            step += chunk

            state = sim.context.getState(
                getPositions=True,
                getEnergy=True,
                groups=groups,
            )
            coords = get_coords_boltz_order(state)
            cv = compute_cv(coords)
            bias_energy_kj = float(
                state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            )
            logw = bias_energy_kj / kbt_kjmol

            n_saved += 1
            npz_path = wdsm_dir / f"wdsm_step_{n_saved:08d}.npz"
            np.savez_compressed(
                str(npz_path),
                coords=coords,
                cv=cv,
                logw=np.float64(logw),
                md_step=np.int64(step),
                bias_energy_kj_mol=np.float64(bias_energy_kj),
            )
            cv_log.append(
                {
                    "step": step,
                    "cv": cv.tolist(),
                    "logw": logw,
                    "bias_energy_kj_mol": bias_energy_kj,
                }
            )

            if step >= next_progress or step == args.n_steps:
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (args.n_steps - step) / rate
                cv_str = ", ".join(f"{v:.3f}" for v in cv)
                print(
                    f"[MD-OPES] step {step:>10,}/{args.n_steps:,} | "
                    f"{rate:.0f} steps/s | ETA {eta/60:.1f}min | "
                    f"CV=[{cv_str}] | "
                    f"bias={bias_energy_kj:.3f} kJ/mol | saved={n_saved}",
                    flush=True,
                )
                tail = _read_latest_plain_colvar_row(opes_dir / "COLVAR")
                if tail:
                    print(f"[MD-OPES] COLVAR (latest row): {tail}", flush=True)
                while next_progress <= step:
                    next_progress += int(args.progress_every)

    if gpu_logger is not None:
        gpu_logger.stop()

    with open(out / "cv_log.json", "w") as f:
        json.dump(cv_log, f)

    elapsed = time.time() - t0
    summary = {
        "total_steps": args.n_steps,
        "total_time_s": elapsed,
        "n_saved": n_saved,
        "opes_mode": args.opes_mode,
        "n_kernels": opes.n_kernels if opes is not None else None,
        "temperature_k": args.temperature,
        "opes_barrier": opes.barrier if opes is not None else args.opes_barrier,
        "opes_biasfactor": (
            opes.biasfactor if opes is not None else args.opes_biasfactor
        ),
        "plumed_force_group": plumed_context.get("force_group"),
        "plumed_script": str(plumed_context.get("script_path"))
        if "script_path" in plumed_context
        else None,
        "plumed_reference_pdb": str(plumed_context.get("reference_pdb"))
        if "reference_pdb" in plumed_context
        else None,
        "gpu_utilization_csv": str(gpu_csv) if gpu_csv is not None else None,
        "gpu_util_interval_s": float(args.gpu_util_interval) if gpu_csv else None,
    }
    with open(out / "md_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(
        f"  MD Complete: {args.n_steps:,} steps in {elapsed:.0f}s ({args.n_steps/elapsed:.0f} steps/s)"
    )
    print(f"  Saved {n_saved} structures to {wdsm_dir}")
    if opes is not None:
        print(f"  OPES: {opes.n_kernels} kernels, {opes.counter} depositions")
    else:
        print(f"  OPES: PLUMED force group {plumed_context.get('force_group')}")
    print(f"{'='*60}", flush=True)
    return None


def main() -> None:
    """Backward-compatible CLI entrypoint (prefer ``scripts/run_openmm_opes_md.py``)."""
    run_opes_md(parse_opes_md_args(sys.argv[1:]))


if __name__ == "__main__":
    main()

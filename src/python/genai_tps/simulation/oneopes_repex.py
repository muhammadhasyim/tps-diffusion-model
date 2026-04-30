"""OneOPES-style stratified Hamiltonian replica exchange helpers for OpenMM + PLUMED.

Stratification follows the **OneOPES** replica ladder described in the original
method paper and the automated host–guest protocol of Febrer Martinez *et al.*
(bioRxiv 2024.08.23.609378, doi:10.1101/2024.08.23.609378): replica **0** is
convergence-focused (main ``OPES_METAD_EXPLORE`` only); replicas **1–7** add
progressive weaker auxiliary ``OPES_METAD_EXPLORE``-style biases on extra CVs
(water coordination, etc.; in this codebase mapped to optional hydration-site
``COORDINATION`` OPES blocks); replicas **4–7** additionally add
``OPES_EXPANDED`` + ``ECV_MULTITHERMAL`` with **PACE = 100** MD steps and
``TEMP_MAX`` increasing toward the hottest replica. **Hamiltonian exchange**
attempts every **1000** MD steps in that protocol (see
:const:`LITERATURE_EXCHANGE_ATTEMPT_STRIDE_STEPS`); GROMACS reference inputs and
automation scripts are published at https://github.com/Pefema/OneOpes_protocol .

This module implements **two-replica** and **eight-replica** neighbor Hamiltonian
exchange (alternating bond phases, ``gmx mdrun -multidir -replex … -hrex`` parity)
with Metropolis acceptance at a shared thermostat temperature when only PLUMED
Hamiltonians differ.

Exact exchange requires evaluating each Hamiltonian on the other replica's
coordinates without corrupting production PLUMED ``STATE`` / ``KERNELS`` files.
The prototype copies per-replica ``opes_states`` trees into scratch directories and
rebuilds lightweight OpenMM contexts from a serialized system template (see
:func:`refresh_evaluator_context`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from genai_tps.simulation.oneopes_upstream_reference import neighbor_hrex_pairs_for_phase

ReplicaDevicePolicy = Literal["round-robin", "packed", "explicit"]
EvaluatorPlacement = Literal["same-device", "dedicated", "serial"]
OneOpesProtocolFlavor = Literal["legacy_boltz", "paper_host_guest"]
PaperOneOpesReferenceCards = Literal["cb8", "temoa", "plumed_nest_23_011", "none"]

# Febrer Martinez et al., bioRxiv 2024.08.23.609378 (OneOPES host–guest protocol).
LITERATURE_EXCHANGE_ATTEMPT_STRIDE_STEPS = 1000
LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS = 100
LITERATURE_AUXILIARY_OPES_PACE_STEPS = 20_000
LITERATURE_MAIN_OPES_PACE_STEPS_HOSTGUEST = 10_000


def register_oneopes_repex_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for :func:`validate_oneopes_repex_cli_args` / the REX driver."""
    from genai_tps.simulation.openmm_md_runner import _parse_comma_int_list

    def _parse_seven_boltz_guests(value: str) -> tuple[int, ...]:
        parts = _parse_comma_int_list(value)
        if len(parts) != 7:
            raise argparse.ArgumentTypeError(
                "expected exactly seven comma-separated Boltz indices "
                "(Pefema order L4,V6,L1,V8,V4,V10,V2 guest atoms)"
            )
        return tuple(parts)

    g = parser.add_argument_group("OneOPES Hamiltonian replica exchange (prototype)")
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Build solvated systems and PLUMED decks for each replica, then exit.",
    )
    g.add_argument(
        "--dry-run-use-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --dry-run: force --platform CPU for deck generation (default: true).",
    )
    g.add_argument(
        "--exchange-every",
        type=int,
        default=LITERATURE_EXCHANGE_ATTEMPT_STRIDE_STEPS,
        help=(
            "Hamiltonian exchange attempt period in MD steps (two-replica run mode; "
            f"bioRxiv 2024.08.23.609378 uses {LITERATURE_EXCHANGE_ATTEMPT_STRIDE_STEPS})."
        ),
    )
    g.add_argument(
        "--n-replicas",
        type=int,
        default=2,
        choices=[2, 8],
        help=(
            "Stratified ladder size. Dynamics: 2 (legacy) or 8 with "
            "--oneopes-protocol paper-host-guest."
        ),
    )
    g.add_argument(
        "--oneopes-protocol",
        type=str,
        default="legacy-boltz",
        choices=["legacy-boltz", "paper-host-guest"],
        help=(
            "PLUMED OneOPES deck flavor: legacy Boltz pp.proj+CONTACTMAP+hydration, "
            "or paper host–guest cyl_z+cosang with Pefema-style WL/WH auxiliaries."
        ),
    )
    g.add_argument(
        "--paper-oneopes-main-pace",
        type=int,
        default=LITERATURE_MAIN_OPES_PACE_STEPS_HOSTGUEST,
        help="Paper ladder: main OPES_METAD_EXPLORE PACE (MD steps).",
    )
    g.add_argument(
        "--paper-oneopes-main-barrier",
        type=float,
        default=100.0,
        help="Paper ladder: main OPES_METAD_EXPLORE BARRIER (kJ/mol).",
    )
    g.add_argument(
        "--paper-oneopes-aux-pace",
        type=int,
        default=LITERATURE_AUXILIARY_OPES_PACE_STEPS,
        help="Paper ladder: auxiliary OPES PACE (MD steps).",
    )
    g.add_argument(
        "--paper-oneopes-aux-barrier",
        type=float,
        default=3.0,
        help="Paper ladder: auxiliary OPES BARRIER (kJ/mol).",
    )
    g.add_argument(
        "--paper-oneopes-multithermal-pace",
        type=int,
        default=LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS,
        help="Paper ladder: OPES_EXPANDED PACE on replicas 4–7.",
    )
    g.add_argument(
        "--paper-oneopes-temp-max",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Hottest multithermal ceiling (K) for replica 7; replicas 4–6 interpolate. "
            "When omitted, use the Pefema fixed ladder (310,330,350,370 K) at 298.15 K."
        ),
    )
    g.add_argument(
        "--paper-oneopes-reference-cards",
        type=str,
        default="none",
        choices=["cb8", "temoa", "plumed_nest_23_011", "none"],
        help="Dry-run only: which reference fixture set to compare deck snippets against.",
    )
    g.add_argument(
        "--paper-oneopes-ligand-axis-p0-boltz",
        type=_parse_comma_int_list,
        default=None,
        help=(
            "Paper protocol: comma-separated Boltz indices for ligand-axis anchor COM "
            "(proximal ligand end)."
        ),
    )
    g.add_argument(
        "--paper-oneopes-ligand-axis-p1-boltz",
        type=_parse_comma_int_list,
        default=None,
        help="Paper protocol: Boltz indices for ligand-axis anchor COM (distal end).",
    )
    g.add_argument(
        "--paper-oneopes-aux-guest-boltz",
        type=_parse_seven_boltz_guests,
        default=None,
        help=(
            "Paper protocol: seven Boltz global guest heavy-atom indices for auxiliary "
            "water coordination CVs (L4,V6,L1,V8,V4,V10,V2 order)."
        ),
    )
    g.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated CUDA/OpenCL device ordinals, e.g. '0,1'.",
    )
    g.add_argument(
        "--replica-device-map",
        type=str,
        default="round-robin",
        choices=["round-robin", "packed", "explicit"],
        help="How production replicas map onto --devices.",
    )
    g.add_argument(
        "--replica-devices",
        type=str,
        default=None,
        help="Comma-separated device ordinals (explicit policy; length must match replicas).",
    )
    g.add_argument(
        "--max-active-contexts-per-device",
        type=int,
        default=4,
        help="Fail if production+evaluator OpenMM contexts would exceed this per device.",
    )
    g.add_argument(
        "--evaluator-placement",
        type=str,
        default="same-device",
        choices=["same-device", "dedicated", "serial"],
        help="Heuristic mapping of cross-Hamiltonian evaluator contexts to GPUs.",
    )


def validate_oneopes_repex_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate combined OpenMM-OPES + OneOPES-REX CLI arguments."""
    if args.bias_cv != "oneopes" or args.opes_mode != "plumed":
        parser.error("OneOPES REX requires --bias-cv oneopes and --opes-mode plumed.")
    if int(args.exchange_every) < 1:
        parser.error("--exchange-every must be >= 1.")
    proto = str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace("-", "_")
    if proto not in ("legacy_boltz", "paper_host_guest"):
        parser.error(f"Unknown --oneopes-protocol {args.oneopes_protocol!r}.")
    if proto == "paper_host_guest" and int(args.n_replicas) != 8:
        parser.error("Paper host–guest protocol requires --n-replicas 8.")
    if bool(getattr(args, "dry_run", False)):
        if int(args.n_replicas) not in (2, 8):
            parser.error("--dry-run supports --n-replicas 2 or 8.")
    elif proto == "legacy_boltz" and int(args.n_replicas) != 2:
        parser.error(
            "Only --n-replicas 2 is supported for legacy-boltz dynamics (or use --dry-run)."
        )
    if args.opes_expanded_temp_max is not None:
        if float(args.opes_expanded_temp_max) <= float(args.temperature):
            parser.error("--opes-expanded-temp-max must exceed --temperature.")
    if args.opes_mode == "plumed":
        from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available

        assert_plumed_opes_metad_available()
    if int(args.oneopes_hydration_max_sites) < 1:
        parser.error("--oneopes-hydration-max-sites must be >= 1")
    if proto == "paper_host_guest":
        lp0 = getattr(args, "paper_oneopes_ligand_axis_p0_boltz", None)
        lp1 = getattr(args, "paper_oneopes_ligand_axis_p1_boltz", None)
        ag = getattr(args, "paper_oneopes_aux_guest_boltz", None)
        if lp0 is None or len(lp0) < 1:
            parser.error(
                "Paper protocol requires --paper-oneopes-ligand-axis-p0-boltz "
                "(non-empty comma-separated Boltz indices)."
            )
        if lp1 is None or len(lp1) < 1:
            parser.error(
                "Paper protocol requires --paper-oneopes-ligand-axis-p1-boltz "
                "(non-empty comma-separated Boltz indices)."
            )
        if ag is None or len(ag) != 7:
            parser.error(
                "Paper protocol requires --paper-oneopes-aux-guest-boltz with "
                "exactly seven Boltz global indices (L4…V2 order)."
            )
        ptm = getattr(args, "paper_oneopes_temp_max", None)
        if ptm is not None and float(ptm) <= float(args.temperature):
            parser.error("--paper-oneopes-temp-max must exceed --temperature.")


def _parse_devices_csv(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("devices list is empty.")
    return [int(p) for p in parts]


__all__ = [
    "ExchangeAttempt",
    "OneOpesAuxiliaryCvSpec",
    "OneOpesProtocolFlavor",
    "OneOpesReplicaSpec",
    "PaperHostGuestReplicaStrat",
    "PaperOneOpesReferenceCards",
    "ReplicaPlumedTuning",
    "ReplicaRuntimeState",
    "assign_replicas_to_devices",
    "build_stratified_replica_plumed_kwargs",
    "copy_opes_state_snapshot",
    "LITERATURE_AUXILIARY_OPES_PACE_STEPS",
    "LITERATURE_EXCHANGE_ATTEMPT_STRIDE_STEPS",
    "LITERATURE_MAIN_OPES_PACE_STEPS_HOSTGUEST",
    "LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS",
    "default_expanded_temp_max_for_replica",
    "multithermal_temp_max_k_for_replica",
    "kbt_kj_per_mol",
    "literature_oneopes_replica_specs",
    "paper_host_guest_replica_strats",
    "log_acceptance_two_replica_hrex",
    "metropolis_accept_two_replica_hrex",
    "opes_states_fingerprint",
    "plan_evaluator_device_assignments",
    "plan_evaluator_devices_for_neighbor_pair",
    "refresh_evaluator_context",
    "register_oneopes_repex_arguments",
    "run_multi_replica_neighbor_oneopes_hrex",
    "run_two_replica_oneopes_repex",
    "validate_oneopes_repex_cli_args",
    "write_repex_config_json",
]


def kbt_kj_per_mol(temperature_k: float) -> float:
    """Return *k* B T in kJ/mol (same convention as :mod:`openmm_md_runner`)."""
    return float(8.314e-3 * float(temperature_k))


@dataclass(frozen=True)
class OneOpesReplicaSpec:
    """Static description of one replica in a OneOPES-style stratified ladder."""

    replica_index: int
    label: str
    force_empty_oneopes_hydration: bool
    oneopes_hydration_site_cap: int | None
    multithermal: bool


@dataclass(frozen=True)
class OneOpesAuxiliaryCvSpec:
    """Paper-ladder auxiliary coordination CV (Pefema-style label on a guest atom).

    *wl_style* selects WL vs WH rational switching parameters on water coordination.
    """

    label: str
    guest_atom_boltz_index: int
    wl_style: bool


@dataclass(frozen=True)
class PaperHostGuestReplicaStrat:
    """Paper host–guest eight-replica stratification row."""

    replica_index: int
    auxiliary_cv_count: int
    multithermal: bool
    multithermal_temp_max_k: float | None


@dataclass(frozen=True)
class ReplicaPlumedTuning:
    """Keyword bundle for :func:`openmm_md_runner.build_plumed_extra_forces_callback`."""

    force_empty_oneopes_hydration: bool = False
    oneopes_hydration_site_cap: int | None = None
    opes_expanded_temp_max_override: float | None = None
    opes_expanded_pace_override: int | None = None
    plumed_state_rfile_override: Path | None = None


@dataclass
class ExchangeAttempt:
    """One Hamiltonian exchange attempt record (two-replica case)."""

    md_step: int
    u00: float
    u01: float
    u10: float
    u11: float
    log_accept: float
    accepted: bool
    rng_uniform: float


@dataclass
class ReplicaRuntimeState:
    """Runtime bookkeeping for a single production replica."""

    replica_index: int
    md_out_dir: Path
    opes_states_dir: Path
    device_index: int | None


def literature_oneopes_replica_specs(*, n_replicas: int) -> tuple[OneOpesReplicaSpec, ...]:
    """Return the built-in OneOPES ladder metadata for *n_replicas* in ``{2, 8}``.

    Two-replica slice (paper ladder indices 0 and 1):

    * Replica 0 — main ``OPES_METAD_EXPLORE`` on leading OneOPES CVs only.
    * Replica 1 — main OPES plus the first auxiliary hydration OPES site.

    Eight-replica ladder (indices ``0 … 7``):

    * ``0`` — main only (no auxiliary hydration OPES).
    * ``1–3`` — progressive inclusion of up to 1, 2, and 3 auxiliary hydration sites.
    * ``4–7`` — same auxiliary count as tier 3 plus ``OPES_EXPANDED`` /
      ``ECV_MULTITHERMAL`` with ``TEMP_MAX`` increasing toward replica 7 (see
      :func:`multithermal_temp_max_k_for_replica`).
    """
    if n_replicas not in (2, 8):
        raise ValueError("Built-in ladder supports n_replicas in {2, 8}.")
    if n_replicas == 2:
        return (
            OneOpesReplicaSpec(
                0,
                "main_oneopes_only",
                force_empty_oneopes_hydration=True,
                oneopes_hydration_site_cap=None,
                multithermal=False,
            ),
            OneOpesReplicaSpec(
                1,
                "main_plus_first_hydration",
                force_empty_oneopes_hydration=False,
                oneopes_hydration_site_cap=1,
                multithermal=False,
            ),
        )
    specs: list[OneOpesReplicaSpec] = []
    for rep in range(8):
        if rep == 0:
            specs.append(
                OneOpesReplicaSpec(
                    rep,
                    "main_only",
                    True,
                    None,
                    False,
                )
            )
        elif rep <= 3:
            specs.append(
                OneOpesReplicaSpec(
                    rep,
                    f"main_plus_{rep}_hydration_sites",
                    False,
                    rep,
                    False,
                )
            )
        else:
            specs.append(
                OneOpesReplicaSpec(
                    rep,
                    f"multithermal_tier_{rep}",
                    False,
                    3,
                    True,
                )
            )
    return tuple(specs)


def paper_host_guest_replica_strats() -> tuple[PaperHostGuestReplicaStrat, ...]:
    """Return the eight-replica paper ladder (auxiliary count + multithermal tier)."""
    from genai_tps.simulation.oneopes_upstream_reference import pefema_multithermal_temp_max_k

    rows: list[PaperHostGuestReplicaStrat] = []
    for r in range(8):
        multithermal = r >= 4
        rows.append(
            PaperHostGuestReplicaStrat(
                replica_index=r,
                auxiliary_cv_count=r if r <= 7 else 7,
                multithermal=multithermal,
                multithermal_temp_max_k=pefema_multithermal_temp_max_k(r) if multithermal else None,
            )
        )
    return tuple(rows)


def multithermal_temp_max_k_for_replica(
    replica_index: int,
    *,
    thermostat_k: float,
    hottest_t_max_k: float | None = None,
) -> float | None:
    """Return ``TEMP_MAX`` (K) for ``OPES_EXPANDED`` / ``ECV_MULTITHERMAL`` on replicas 4–7.

    When *hottest_t_max_k* is ``None``, use a fixed increment ladder above *thermostat_k*
    (20, 40, 60, 80 K), matching the qualitative ``OPES MultiT`` row in Table 1 of
    Febrer Martinez *et al.* (bioRxiv 2024.08.23.609378) anchored at the simulation
    thermostat. When *hottest_t_max_k* is set, interpolate linearly so replica 4 is
    coolest above *thermostat_k* and replica 7 reaches *hottest_t_max_k*.
    """
    if replica_index < 4:
        return None
    t0 = float(thermostat_k)
    if hottest_t_max_k is None:
        deltas = (20.0, 40.0, 60.0, 80.0)
        return t0 + deltas[replica_index - 4]
    hot = float(hottest_t_max_k)
    if hot <= t0 + 1e-6:
        raise ValueError(
            "Multithermal TEMP_MAX must be strictly greater than the thermostat temperature."
        )
    frac = (replica_index - 3) / 4.0
    return t0 + (hot - t0) * frac


def default_expanded_temp_max_for_replica(
    replica_index: int, *, thermostat_k: float = 298.15
) -> float:
    """Backward-compatible alias: ``TEMP_MAX`` for replica *replica_index* (≥ 4)."""
    v = multithermal_temp_max_k_for_replica(
        replica_index, thermostat_k=thermostat_k, hottest_t_max_k=None
    )
    if v is None:
        raise ValueError("replica_index must be in the multithermal tier (>= 4).")
    return float(v)


def build_stratified_replica_plumed_kwargs(
    replica_index: int,
    *,
    n_replicas: int,
    user_expanded_temp_max: float | None,
    user_expanded_pace: int | None,
    thermostat_temperature_k: float,
    oneopes_protocol: str = "legacy_boltz",
    paper_multithermal_pace: int | None = None,
) -> dict[str, Any]:
    """Map stratification metadata to :func:`build_plumed_extra_forces_callback` kwargs.

    ``--opes-expanded-temp-max`` / ``--paper-oneopes-temp-max`` (when passed) is
    interpreted as the **hottest** multithermal ceiling for replica **7** only;
    replicas **4–6** are interpolated. It does **not** enable multithermal on
    replicas **0–3** (legacy Boltz ladder).

    *oneopes_protocol* ``"paper_host_guest"`` requires *n_replicas* ``8`` and uses
    the Pefema fixed ``TEMP_MAX`` ladder (310–370 K) unless a hottest ceiling is
    provided for interpolation.
    """
    from genai_tps.simulation.oneopes_upstream_reference import pefema_multithermal_temp_max_k

    if oneopes_protocol == "paper_host_guest":
        if n_replicas != 8:
            raise ValueError('paper_host_guest stratification requires n_replicas=8.')
        strat = paper_host_guest_replica_strats()[replica_index]
        exp_max: float | None = None
        pace_ov: int | None = None
        if strat.multithermal:
            if user_expanded_temp_max is not None:
                exp_max = multithermal_temp_max_k_for_replica(
                    replica_index,
                    thermostat_k=float(thermostat_temperature_k),
                    hottest_t_max_k=float(user_expanded_temp_max),
                )
            else:
                exp_max = pefema_multithermal_temp_max_k(replica_index)
            pace_ov = (
                int(user_expanded_pace)
                if user_expanded_pace is not None
                else (
                    int(paper_multithermal_pace)
                    if paper_multithermal_pace is not None
                    else LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS
                )
            )
        return {
            "oneopes_protocol": "paper_host_guest",
            "force_empty_oneopes_hydration": False,
            "oneopes_hydration_site_cap": None,
            "opes_expanded_temp_max_override": exp_max,
            "opes_expanded_pace_override": pace_ov,
            "plumed_state_rfile_override": None,
            "paper_replica_index": int(replica_index),
        }

    specs = literature_oneopes_replica_specs(n_replicas=n_replicas)
    spec = specs[replica_index]
    exp_max = None
    pace_ov = None
    if spec.multithermal:
        exp_max = multithermal_temp_max_k_for_replica(
            replica_index,
            thermostat_k=float(thermostat_temperature_k),
            hottest_t_max_k=float(user_expanded_temp_max)
            if user_expanded_temp_max is not None
            else None,
        )
        pace_ov = (
            int(user_expanded_pace)
            if user_expanded_pace is not None
            else LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS
        )
    return {
        "oneopes_protocol": "legacy_boltz",
        "force_empty_oneopes_hydration": spec.force_empty_oneopes_hydration,
        "oneopes_hydration_site_cap": spec.oneopes_hydration_site_cap,
        "opes_expanded_temp_max_override": exp_max,
        "opes_expanded_pace_override": pace_ov,
        "plumed_state_rfile_override": None,
    }


def assign_replicas_to_devices(
    n_replicas: int,
    devices: Sequence[int],
    policy: ReplicaDevicePolicy,
    *,
    explicit_map: Sequence[int] | None = None,
) -> list[int]:
    """Assign each replica index to a CUDA/OpenCL ``DeviceIndex`` ordinal.

    Parameters
    ----------
    n_replicas
        Number of replicas (must be positive).
    devices
        Non-empty list of non-negative device ordinals selected by the user.
    policy
        ``"round-robin"`` cycles ``0,1,…,|devices|-1,0,…``;
        ``"packed"`` fills device ``0`` until ``ceil(n/|devices|)`` slots, then device ``1``, …;
        ``"explicit"`` requires *explicit_map* of length *n_replicas*.
    explicit_map
        Per-replica device ordinals (must be subsets of *devices* when *devices* is
        treated as the allowed set — caller should validate).

    Returns
    -------
    list[int]
        Length ``n_replicas`` device assignment.
    """
    if n_replicas < 1:
        raise ValueError("n_replicas must be >= 1.")
    devs = [int(d) for d in devices]
    if not devs:
        raise ValueError("devices must be non-empty.")
    if min(devs) < 0:
        raise ValueError("device ordinals must be non-negative.")
    if policy == "explicit":
        if explicit_map is None or len(explicit_map) != n_replicas:
            raise ValueError("explicit policy requires explicit_map of length n_replicas.")
        return [int(x) for x in explicit_map]
    if explicit_map is not None:
        raise ValueError("explicit_map is only used when policy='explicit'.")
    if policy == "round-robin":
        return [devs[i % len(devs)] for i in range(n_replicas)]
    if policy == "packed":
        n_dev = len(devs)
        cap = int(math.ceil(float(n_replicas) / float(n_dev)))
        out: list[int] = []
        d_i = 0
        while len(out) < n_replicas:
            d_ord = devs[d_i % n_dev]
            chunk = min(cap, n_replicas - len(out))
            out.extend([d_ord] * chunk)
            d_i += 1
        return out
    raise ValueError(f"unknown policy {policy!r}")


def plan_evaluator_device_assignments(
    replica_devices: Sequence[int],
    *,
    placement: EvaluatorPlacement,
    n_hamiltonians: int | None = None,
) -> list[int | None]:
    """Return a suggested evaluator device row per replica index.

    ``"same-device"`` pins evaluator *i* to the production device of replica *i*.
    ``"dedicated"`` returns ``None`` for each entry (caller may pick a free GPU).
    ``"serial"`` returns a single shared device (minimum replica device) for every row.

    *n_hamiltonians* is ignored; kept for backward compatibility with older call sites.
    """
    del n_hamiltonians  # API compatibility
    n = len(replica_devices)
    if n < 2:
        raise ValueError("replica_devices must cover at least two replicas.")
    if placement == "same-device":
        return [int(replica_devices[i]) for i in range(n)]
    if placement == "dedicated":
        return [None] * n
    if placement == "serial":
        shared = min(int(d) for d in replica_devices)
        return [shared] * n
    raise ValueError(f"unknown placement {placement!r}")


def plan_evaluator_devices_for_neighbor_pair(
    replica_devices: Sequence[int],
    replica_i: int,
    replica_j: int,
    *,
    placement: EvaluatorPlacement,
) -> tuple[int | None, int | None]:
    """Pick evaluator device ordinals for Hamiltonians *i* and *j* on a neighbor swap."""
    if int(replica_i) == int(replica_j):
        raise ValueError("replica_i and replica_j must differ.")
    if int(replica_i) < 0 or int(replica_j) < 0:
        raise ValueError("replica indices must be non-negative.")
    if int(replica_i) >= len(replica_devices) or int(replica_j) >= len(replica_devices):
        raise ValueError("replica indices out of range for replica_devices.")
    if placement == "same-device":
        return int(replica_devices[replica_i]), int(replica_devices[replica_j])
    if placement == "dedicated":
        return None, None
    if placement == "serial":
        shared = min(int(replica_devices[replica_i]), int(replica_devices[replica_j]))
        return shared, shared
    raise ValueError(f"unknown placement {placement!r}")


def count_active_contexts_per_device(
    replica_devices: Sequence[int],
    evaluator_devices: Sequence[int | None],
) -> dict[int, int]:
    """Count production + evaluator contexts touching each device ordinal."""
    counts: dict[int, int] = {}
    for d in replica_devices:
        dd = int(d)
        counts[dd] = counts.get(dd, 0) + 1
    for ev in evaluator_devices:
        if ev is None:
            continue
        dd = int(ev)
        counts[dd] = counts.get(dd, 0) + 1
    return counts


def assert_context_budget(
    counts: dict[int, int],
    *,
    max_active_contexts_per_device: int,
) -> None:
    """Raise ``RuntimeError`` if any device exceeds the configured context budget."""
    for dev, n_ctx in counts.items():
        if n_ctx > int(max_active_contexts_per_device):
            raise RuntimeError(
                f"Device {dev} would host {n_ctx} OpenMM contexts but "
                f"--max-active-contexts-per-device={max_active_contexts_per_device}."
            )


def log_acceptance_two_replica_hrex(u00: float, u01: float, u10: float, u11: float) -> float:
    """Natural log of the Metropolis acceptance ratio for two-replica H-REX.

    All inputs are **reduced** energies :math:`u_i(x_j) = U_i(x_j) / (k_B T)` with
    identical temperature across replicas so the kinetic part cancels in the
    ratio when only PLUMED biases differ; callers should pass consistent totals
    (e.g. MM+bias) if Hamiltonians differ in the MM part as well.
    """
    return float(-(u01 + u10 - u00 - u11))


def metropolis_accept_two_replica_hrex(
    u00: float,
    u01: float,
    u10: float,
    u11: float,
    rng: random.Random,
) -> tuple[bool, float, float]:
    """Draw uniform ``u`` and return ``(accepted, log_alpha, u)``."""
    for x in (u00, u01, u10, u11):
        if not math.isfinite(float(x)):
            return False, float("nan"), float(rng.random())
    log_a = log_acceptance_two_replica_hrex(u00, u01, u10, u11)
    if not math.isfinite(log_a):
        return False, log_a, float(rng.random())
    u = rng.random()
    accepted = bool(log_a >= 0.0 or math.log(u) < log_a)
    return accepted, log_a, float(u)


def relocate_plumed_deck_paths(script_text: str, src_root: Path, dst_root: Path) -> str:
    """Replace absolute *src_root* paths with *dst_root* inside a PLUMED deck."""
    a = str(src_root.expanduser().resolve())
    b = str(dst_root.expanduser().resolve())
    return script_text.replace(a, b)


def prepare_evaluator_scratch_tree(
    *,
    production_rep_root: Path,
    scratch_root: Path,
) -> Path:
    """Copy RMSD reference + relocated ``plumed_opes.dat`` into *scratch_root*.

    The caller should already have populated ``scratch_root/opes_states`` via
    :func:`copy_opes_state_snapshot`.
    """
    scratch_root = scratch_root.expanduser().resolve()
    production_rep_root = production_rep_root.expanduser().resolve()
    raw = (production_rep_root / "plumed_opes.dat").read_text(encoding="utf-8")
    fixed = relocate_plumed_deck_paths(raw, production_rep_root, scratch_root)
    sp = scratch_root / "plumed_opes.dat"
    sp.write_text(fixed, encoding="utf-8")
    ref_src = production_rep_root / "plumed_rmsd_reference.pdb"
    if ref_src.is_file():
        shutil.copy2(ref_src, scratch_root / "plumed_rmsd_reference.pdb")
    return sp


def copy_opes_state_snapshot(src: Path, dst: Path) -> None:
    """Copy OPES file artifacts from production *src* into scratch *dst*.

    Copies ``KERNELS``, ``STATE``, optional ``STATE_EXPANDED`` / ``OPES_EXPANDED_DELTAFS``,
    and auxiliary ``KERNELS_HYDR_*`` / ``STATE_HYDR_*`` pairs when present.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"missing opes_states directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    names = [
        "KERNELS",
        "STATE",
        "STATE_EXPANDED",
        "OPES_EXPANDED_DELTAFS",
    ]
    for name in names:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
    for p in sorted(src.glob("KERNELS_HYDR_*")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("STATE_HYDR_*")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("KERNELS_AUX_*")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("STATE_AUX_*")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("Kernels*.data")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("compressed_Kernels*.data")):
        shutil.copy2(p, dst / p.name)
    for p in sorted(src.glob("DeltaFs.data")):
        shutil.copy2(p, dst / p.name)


def opes_states_fingerprint(opes_states_dir: Path) -> dict[str, str]:
    """Return ``{relative_path: sha256_hex}`` for every file under *opes_states_dir*."""
    root = opes_states_dir.expanduser().resolve()
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[rel] = digest
    return out


def write_repex_config_json(
    path: Path,
    *,
    replica_devices: Sequence[int],
    evaluator_devices: Sequence[int | None],
    max_active_contexts_per_device: int,
    exchange_every: int,
    n_replicas: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write resolved placement and bookkeeping fields for reproducibility."""
    counts = count_active_contexts_per_device(replica_devices, evaluator_devices)
    payload = {
        "n_replicas": int(n_replicas),
        "exchange_every": int(exchange_every),
        "replica_devices": [int(x) for x in replica_devices],
        "evaluator_devices": [int(x) if x is not None else None for x in evaluator_devices],
        "active_context_counts_by_device": {str(k): v for k, v in sorted(counts.items())},
        "max_active_contexts_per_device": int(max_active_contexts_per_device),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _serialize_system(system: Any) -> str:
    import openmm as mm

    return mm.XmlSerializer.serialize(system)


def _deserialize_system(xml: str) -> Any:
    import openmm as mm

    return mm.XmlSerializer.deserialize(xml)


def refresh_evaluator_context(
    *,
    template_system_xml: str,
    plumed_force_index: int,
    plumed_script_path: Path,
    temperature_k: float,
    force_group: int,
    platform: Any,
    platform_properties: dict[str, str] | None,
    positions: Any,
    box_vectors: Any | None,
) -> Any:
    """Build a fresh OpenMM ``Context`` whose PLUMED force reads the on-disk script.

    The caller must refresh ``opes_states`` files under the paths referenced in the
    script **before** calling this function so PLUMED opens consistent kernels/state.
    """
    import openmm as mm
    import openmm.unit as u

    from genai_tps.simulation.plumed_opes import add_plumed_opes_to_system

    system = _deserialize_system(template_system_xml)
    system.removeForce(int(plumed_force_index))
    script = plumed_script_path.read_text(encoding="utf-8")
    add_plumed_opes_to_system(
        system,
        script,
        temperature=float(temperature_k),
        force_group=int(force_group),
        restart=True,
    )
    integrator = mm.LangevinMiddleIntegrator(
        float(temperature_k) * u.kelvin,
        1.0 / u.picosecond,
        2.0 * u.femtoseconds,
    )
    ctx = mm.Context(system, integrator, platform, platform_properties or {})
    ctx.setPositions(positions)
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)
    return ctx


def _bias_energy_kj_mol(context: Any, force_group: int) -> float:
    import openmm.unit as uu

    st = context.getState(getEnergy=True, groups={int(force_group)})
    return float(st.getPotentialEnergy().value_in_unit(uu.kilojoule_per_mole))


def _reduced_u(
    bias_kj_mol: float,
    *,
    kbt_kjmol: float,
    mm_bias_kj_mol: float | None = None,
) -> float:
    """Reduced potential contribution for Metropolis ratio (dimensionless)."""
    if mm_bias_kj_mol is not None:
        return float((mm_bias_kj_mol + bias_kj_mol) / kbt_kjmol)
    return float(bias_kj_mol / kbt_kjmol)


def run_two_replica_oneopes_repex(
    *,
    args: Any,
    out_root: Path,
    sim0: Any,
    sim1: Any,
    meta0: dict[str, Any],
    meta1: dict[str, Any],
    plumed_ctx0: dict[str, Any],
    plumed_ctx1: dict[str, Any],
    exchange_every: int,
    scratch_root: Path,
    rng: random.Random,
    replica_devices: Sequence[int],
    evaluator_devices: Sequence[int | None],
    max_active_contexts_per_device: int,
    mm_energy_in_exchange: bool = False,
) -> None:
    """Run two-replica MD with periodic Hamiltonian exchange attempts.

    *sim0* / *sim1* must already be minimized and share identical MM Hamiltonians
    (PLUMED is the only differing force). When *mm_energy_in_exchange* is ``False``
    (default), only PLUMED bias energies enter the Metropolis ratio (valid when the
    OpenMM valence+nonbonded energy is identical for the same coordinates).

    Writes ``<out_root>/exchange_log.csv`` and uses *scratch_root* for evaluator
    PLUMED file snapshots.
    """
    import openmm as mm
    import openmm.unit as u

    from genai_tps.utils.compute_device import openmm_device_index_properties

    assert_context_budget(
        count_active_contexts_per_device(replica_devices, evaluator_devices),
        max_active_contexts_per_device=max_active_contexts_per_device,
    )

    out_root = out_root.expanduser().resolve()
    scratch_root = scratch_root.expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    kbt = kbt_kj_per_mol(float(args.temperature))

    rep_dirs = [out_root / "rep000", out_root / "rep001"]

    # Serialize production systems for evaluator cloning.
    sys_xml0 = _serialize_system(sim0.system)
    sys_xml1 = _serialize_system(sim1.system)
    idx0 = int(plumed_ctx0["force_index"])
    idx1 = int(plumed_ctx1["force_index"])
    fg0 = int(plumed_ctx0["force_group"])
    fg1 = int(plumed_ctx1["force_group"])
    script0 = Path(plumed_ctx0["script_path"])
    script1 = Path(plumed_ctx1["script_path"])

    def _make_platform_for_device(dev: int | None) -> tuple[Any, dict[str, str] | None]:
        platform = mm.Platform.getPlatformByName(str(args.platform))
        if dev is None or str(args.platform) == "CPU":
            return platform, openmm_device_index_properties(str(args.platform), None)
        return platform, openmm_device_index_properties(str(args.platform), int(dev))

    exchange_log = out_root / "exchange_log.csv"
    exchange_log.parent.mkdir(parents=True, exist_ok=True)
    n_steps = int(args.n_steps)
    step = 0
    t_run0 = time.time()
    with exchange_log.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "md_step",
                "u00",
                "u01",
                "u10",
                "u11",
                "log_accept",
                "accepted",
                "rng_u",
                "elapsed_s",
            ]
        )

        while step < n_steps:
            chunk = min(int(exchange_every), n_steps - step)
            sim0.step(chunk)
            sim1.step(chunk)
            step += chunk

            st0 = sim0.context.getState(getPositions=True, getEnergy=True)
            st1 = sim1.context.getState(getPositions=True, getEnergy=True)
            pos0 = st0.getPositions()
            pos1 = st1.getPositions()
            bv0 = sim0.context.getState(getEnergy=True, groups={fg0}).getPotentialEnergy().value_in_unit(
                u.kilojoule_per_mole
            )
            bv1 = sim1.context.getState(getEnergy=True, groups={fg1}).getPotentialEnergy().value_in_unit(
                u.kilojoule_per_mole
            )

            scratch0 = scratch_root / f"eval_h0_step{step:010d}"
            scratch1 = scratch_root / f"eval_h1_step{step:010d}"
            for s in (scratch0, scratch1):
                if s.exists():
                    shutil.rmtree(s)
            scratch0.mkdir(parents=True)
            scratch1.mkdir(parents=True)
            (scratch0 / "opes_states").mkdir(parents=True)
            (scratch1 / "opes_states").mkdir(parents=True)
            copy_opes_state_snapshot(rep_dirs[0] / "opes_states", scratch0 / "opes_states")
            copy_opes_state_snapshot(rep_dirs[1] / "opes_states", scratch1 / "opes_states")
            script_scratch0 = prepare_evaluator_scratch_tree(
                production_rep_root=rep_dirs[0],
                scratch_root=scratch0,
            )
            script_scratch1 = prepare_evaluator_scratch_tree(
                production_rep_root=rep_dirs[1],
                scratch_root=scratch1,
            )

            plat0, prop0 = _make_platform_for_device(evaluator_devices[0])
            plat1, prop1 = _make_platform_for_device(evaluator_devices[1])
            try:
                boxv1 = st1.getPeriodicBoxVectors()
            except Exception:
                boxv1 = None
            try:
                boxv0 = st0.getPeriodicBoxVectors()
            except Exception:
                boxv0 = None
            ctx_cross01 = refresh_evaluator_context(
                template_system_xml=sys_xml0,
                plumed_force_index=idx0,
                plumed_script_path=script_scratch0,
                temperature_k=float(args.temperature),
                force_group=fg0,
                platform=plat0,
                platform_properties=prop0,
                positions=pos1,
                box_vectors=boxv1,
            )
            ctx_cross10 = refresh_evaluator_context(
                template_system_xml=sys_xml1,
                plumed_force_index=idx1,
                plumed_script_path=script_scratch1,
                temperature_k=float(args.temperature),
                force_group=fg1,
                platform=plat1,
                platform_properties=prop1,
                positions=pos0,
                box_vectors=boxv0,
            )

            eb01 = _bias_energy_kj_mol(ctx_cross01, fg0)
            eb10 = _bias_energy_kj_mol(ctx_cross10, fg1)
            del ctx_cross01
            del ctx_cross10

            if mm_energy_in_exchange:
                # Total MM+PLUMED — requires cloning full contexts; not implemented fast path.
                raise NotImplementedError(
                    "mm_energy_in_exchange=True requires consistent total-energy evaluation; "
                    "use bias-only exchange (default) when MM Hamiltonians match."
                )
            u00 = _reduced_u(bv0, kbt_kjmol=kbt)
            u11 = _reduced_u(bv1, kbt_kjmol=kbt)
            u01 = _reduced_u(eb01, kbt_kjmol=kbt)
            u10 = _reduced_u(eb10, kbt_kjmol=kbt)

            accepted, log_a, ru = metropolis_accept_two_replica_hrex(u00, u01, u10, u11, rng)
            if accepted:
                sim0.context.setPositions(pos1)
                sim1.context.setPositions(pos0)
                v0 = st0.getVelocities()
                v1 = st1.getVelocities()
                sim0.context.setVelocities(v1)
                sim1.context.setVelocities(v0)

            w.writerow(
                [
                    step,
                    f"{u00:.8g}",
                    f"{u01:.8g}",
                    f"{u10:.8g}",
                    f"{u11:.8g}",
                    f"{log_a:.8g}",
                    int(accepted),
                    f"{ru:.8g}",
                    f"{time.time() - t_run0:.3f}",
                ]
            )
            fcsv.flush()


def run_multi_replica_neighbor_oneopes_hrex(
    *,
    args: Any,
    out_root: Path,
    sims: Sequence[Any],
    _metas: Sequence[dict[str, Any]],
    plumed_contexts: Sequence[dict[str, Any]],
    exchange_every: int,
    scratch_root: Path,
    rng: random.Random,
    replica_devices: Sequence[int],
    evaluator_placement: EvaluatorPlacement,
    max_active_contexts_per_device: int,
    mm_energy_in_exchange: bool = False,
) -> None:
    """Run eight-replica MD with alternating neighbor Hamiltonian exchange.

    Independent neighbor pairs within a phase are attempted **sequentially** so
    at most two scratch evaluator contexts exist at once (lower GPU memory peak
    than fully parallel pair evaluation).
    """
    import openmm as mm
    import openmm.unit as u

    from genai_tps.utils.compute_device import openmm_device_index_properties

    n = len(sims)
    if n != 8:
        raise ValueError("run_multi_replica_neighbor_oneopes_hrex requires eight replicas.")
    if len(_metas) != n or len(plumed_contexts) != n or len(replica_devices) != n:
        raise ValueError(
            "sims, _metas, plumed_contexts, and replica_devices must have length 8."
        )

    base_counts = count_active_contexts_per_device(replica_devices, ())
    peak_counts = {dev: cnt + 2 for dev, cnt in base_counts.items()}
    assert_context_budget(
        peak_counts,
        max_active_contexts_per_device=max_active_contexts_per_device,
    )

    out_root = out_root.expanduser().resolve()
    scratch_root = scratch_root.expanduser().resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    kbt = kbt_kj_per_mol(float(args.temperature))
    rep_dirs = [out_root / f"rep{k:03d}" for k in range(n)]
    sims_list = list(sims)

    sys_xmls = [_serialize_system(s.system) for s in sims_list]
    idxs = [int(pc["force_index"]) for pc in plumed_contexts]
    fgs = [int(pc["force_group"]) for pc in plumed_contexts]
    scripts = [Path(pc["script_path"]) for pc in plumed_contexts]

    def _make_platform_for_device(dev: int | None) -> tuple[Any, dict[str, str] | None]:
        platform = mm.Platform.getPlatformByName(str(args.platform))
        if dev is None or str(args.platform) == "CPU":
            return platform, openmm_device_index_properties(str(args.platform), None)
        return platform, openmm_device_index_properties(str(args.platform), int(dev))

    exchange_log = out_root / "exchange_log.csv"
    exchange_log.parent.mkdir(parents=True, exist_ok=True)
    n_steps = int(args.n_steps)
    step = 0
    t_run0 = time.time()
    with exchange_log.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "md_step",
                "phase_mod_2",
                "pair_i",
                "pair_j",
                "u_ii",
                "u_jj",
                "u_ij",
                "u_ji",
                "log_accept",
                "accepted",
                "rng_u",
                "elapsed_s",
            ]
        )

        while step < n_steps:
            chunk = min(int(exchange_every), n_steps - step)
            for s in sims_list:
                s.step(chunk)
            step += chunk

            phase = ((step // int(exchange_every)) - 1) % 2
            pairs = neighbor_hrex_pairs_for_phase(phase)

            for i, j in pairs:
                st_i = sims_list[i].context.getState(getPositions=True, getEnergy=True)
                st_j = sims_list[j].context.getState(getPositions=True, getEnergy=True)
                pos_i = st_i.getPositions()
                pos_j = st_j.getPositions()
                bv_i = (
                    sims_list[i]
                    .context.getState(getEnergy=True, groups={fgs[i]})
                    .getPotentialEnergy()
                    .value_in_unit(u.kilojoule_per_mole)
                )
                bv_j = (
                    sims_list[j]
                    .context.getState(getEnergy=True, groups={fgs[j]})
                    .getPotentialEnergy()
                    .value_in_unit(u.kilojoule_per_mole)
                )

                ev_i, ev_j = plan_evaluator_devices_for_neighbor_pair(
                    replica_devices,
                    i,
                    j,
                    placement=evaluator_placement,
                )

                scratch_i = scratch_root / f"eval_step{step:010d}_h{i}_coords{j}"
                scratch_j = scratch_root / f"eval_step{step:010d}_h{j}_coords{i}"
                for sdir in (scratch_i, scratch_j):
                    if sdir.exists():
                        shutil.rmtree(sdir)
                    sdir.mkdir(parents=True)
                    (sdir / "opes_states").mkdir(parents=True)
                copy_opes_state_snapshot(rep_dirs[i] / "opes_states", scratch_i / "opes_states")
                copy_opes_state_snapshot(rep_dirs[j] / "opes_states", scratch_j / "opes_states")
                script_scratch_i = prepare_evaluator_scratch_tree(
                    production_rep_root=rep_dirs[i],
                    scratch_root=scratch_i,
                )
                script_scratch_j = prepare_evaluator_scratch_tree(
                    production_rep_root=rep_dirs[j],
                    scratch_root=scratch_j,
                )

                plat_i, prop_i = _make_platform_for_device(ev_i)
                plat_j, prop_j = _make_platform_for_device(ev_j)
                try:
                    box_j = st_j.getPeriodicBoxVectors()
                except Exception:
                    box_j = None
                try:
                    box_i = st_i.getPeriodicBoxVectors()
                except Exception:
                    box_i = None

                ctx_ij = refresh_evaluator_context(
                    template_system_xml=sys_xmls[i],
                    plumed_force_index=idxs[i],
                    plumed_script_path=script_scratch_i,
                    temperature_k=float(args.temperature),
                    force_group=fgs[i],
                    platform=plat_i,
                    platform_properties=prop_i,
                    positions=pos_j,
                    box_vectors=box_j,
                )
                ctx_ji = refresh_evaluator_context(
                    template_system_xml=sys_xmls[j],
                    plumed_force_index=idxs[j],
                    plumed_script_path=script_scratch_j,
                    temperature_k=float(args.temperature),
                    force_group=fgs[j],
                    platform=plat_j,
                    platform_properties=prop_j,
                    positions=pos_i,
                    box_vectors=box_i,
                )

                eb_ij = _bias_energy_kj_mol(ctx_ij, fgs[i])
                eb_ji = _bias_energy_kj_mol(ctx_ji, fgs[j])
                del ctx_ij
                del ctx_ji

                if mm_energy_in_exchange:
                    raise NotImplementedError(
                        "mm_energy_in_exchange=True is not implemented for eight-replica H-REX."
                    )

                u_ii = _reduced_u(bv_i, kbt_kjmol=kbt)
                u_jj = _reduced_u(bv_j, kbt_kjmol=kbt)
                u_ij = _reduced_u(eb_ij, kbt_kjmol=kbt)
                u_ji = _reduced_u(eb_ji, kbt_kjmol=kbt)

                accepted, log_a, ru = metropolis_accept_two_replica_hrex(
                    u_ii, u_ij, u_ji, u_jj, rng
                )
                if accepted:
                    sims_list[i].context.setPositions(pos_j)
                    sims_list[j].context.setPositions(pos_i)
                    vi = st_i.getVelocities()
                    vj = st_j.getVelocities()
                    sims_list[i].context.setVelocities(vj)
                    sims_list[j].context.setVelocities(vi)

                w.writerow(
                    [
                        step,
                        int(phase),
                        int(i),
                        int(j),
                        f"{u_ii:.8g}",
                        f"{u_jj:.8g}",
                        f"{u_ij:.8g}",
                        f"{u_ji:.8g}",
                        f"{log_a:.8g}",
                        int(accepted),
                        f"{ru:.8g}",
                        f"{time.time() - t_run0:.3f}",
                    ]
                )
                fcsv.flush()

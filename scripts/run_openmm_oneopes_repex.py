#!/usr/bin/env python3
"""OneOPES Hamiltonian replica exchange driver (OpenMM + PLUMED).

Supports two-replica ``legacy-boltz`` stratification and eight-replica
``paper-host-guest`` neighbor exchange (alternating bond phases).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

from genai_tps.simulation.openmm_md_runner import build_opes_md_argument_parser, run_opes_md
from genai_tps.simulation.oneopes_repex import (
    _parse_devices_csv,
    assert_context_budget,
    assign_replicas_to_devices,
    build_stratified_replica_plumed_kwargs,
    count_active_contexts_per_device,
    plan_evaluator_device_assignments,
    register_oneopes_repex_arguments,
    resolve_replica_step_workers,
    run_multi_replica_neighbor_oneopes_hrex,
    run_two_replica_oneopes_repex,
    validate_oneopes_repex_cli_args,
    write_repex_config_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = build_opes_md_argument_parser()
    register_oneopes_repex_arguments(parser)
    return parser


def _normalized_protocol(args: argparse.Namespace) -> str:
    return str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace("-", "_")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    validate_oneopes_repex_cli_args(args, parser)

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    proto = _normalized_protocol(args)

    devices = _parse_devices_csv(str(args.devices))
    explicit_map: list[int] | None = None
    if args.replica_device_map == "explicit":
        if not args.replica_devices:
            parser.error("--replica-devices is required when --replica-device-map explicit")
        explicit_map = _parse_devices_csv(str(args.replica_devices))
        if len(explicit_map) != int(args.n_replicas):
            parser.error("--replica-devices length must match --n-replicas")

    rep_devs = assign_replicas_to_devices(
        int(args.n_replicas),
        devices,
        args.replica_device_map,  # type: ignore[arg-type]
        explicit_map=explicit_map,
    )
    eval_devs = plan_evaluator_device_assignments(
        rep_devs,
        placement=args.evaluator_placement,  # type: ignore[arg-type]
    )
    eval_budget = [None] * int(args.n_replicas) if int(args.n_replicas) > 2 else eval_devs
    resolved_step_workers = resolve_replica_step_workers(
        int(getattr(args, "replica_step_workers", 0)),
        n_replicas=int(args.n_replicas),
    )
    assert_context_budget(
        count_active_contexts_per_device(rep_devs, eval_budget),
        max_active_contexts_per_device=int(args.max_active_contexts_per_device),
    )

    user_tm = getattr(args, "paper_oneopes_temp_max", None)
    if user_tm is None:
        user_tm = args.opes_expanded_temp_max

    write_repex_config_json(
        out_root / "repex_config.json",
        replica_devices=rep_devs,
        evaluator_devices=eval_devs,
        max_active_contexts_per_device=int(args.max_active_contexts_per_device),
        exchange_every=int(args.exchange_every),
        n_replicas=int(args.n_replicas),
        extra={
            "devices": devices,
            "replica_device_map": str(args.replica_device_map),
            "replica_step_workers": resolved_step_workers,
            "oneopes_protocol": proto,
        },
    )

    rng = random.Random(0)
    n_rep = int(args.n_replicas)
    if proto == "paper_host_guest" and n_rep == 8:
        packs: list[dict] = []
        for rep in range(8):
            rargs = copy.copy(args)
            rargs.platform = str(args.platform)
            rargs.openmm_device_index = int(rep_devs[rep])
            strat = build_stratified_replica_plumed_kwargs(
                rep,
                n_replicas=8,
                user_expanded_temp_max=user_tm,
                user_expanded_pace=args.opes_expanded_pace,
                thermostat_temperature_k=float(args.temperature),
                oneopes_protocol="paper_host_guest",
                paper_multithermal_pace=int(getattr(args, "paper_oneopes_multithermal_pace", 100)),
            )
            pack = run_opes_md(
                rargs,
                md_out_dir=out_root / f"rep{rep:03d}",
                plumed_factory_extra_kwargs=strat,
                return_after_initialization=True,
            )
            assert isinstance(pack, dict)
            packs.append(pack)

        run_multi_replica_neighbor_oneopes_hrex(
            args=args,
            out_root=out_root,
            sims=[p["sim"] for p in packs],
            _metas=[p["meta"] for p in packs],
            plumed_contexts=[p["plumed_context"] for p in packs],
            exchange_every=int(args.exchange_every),
            scratch_root=out_root / "eval_scratch",
            rng=rng,
            replica_devices=rep_devs,
            evaluator_placement=args.evaluator_placement,  # type: ignore[arg-type]
            max_active_contexts_per_device=int(args.max_active_contexts_per_device),
        )
    else:
        packs = []
        for rep in range(2):
            rargs = copy.copy(args)
            rargs.platform = str(args.platform)
            rargs.openmm_device_index = int(rep_devs[rep])
            strat = build_stratified_replica_plumed_kwargs(
                rep,
                n_replicas=2,
                user_expanded_temp_max=user_tm,
                user_expanded_pace=args.opes_expanded_pace,
                thermostat_temperature_k=float(args.temperature),
                oneopes_protocol="legacy_boltz",
            )
            pack = run_opes_md(
                rargs,
                md_out_dir=out_root / f"rep{rep:03d}",
                plumed_factory_extra_kwargs=strat,
                return_after_initialization=True,
            )
            assert isinstance(pack, dict)
            packs.append(pack)

        run_two_replica_oneopes_repex(
            args=args,
            out_root=out_root,
            sim0=packs[0]["sim"],
            sim1=packs[1]["sim"],
            meta0=packs[0]["meta"],
            meta1=packs[1]["meta"],
            plumed_ctx0=packs[0]["plumed_context"],
            plumed_ctx1=packs[1]["plumed_context"],
            exchange_every=int(args.exchange_every),
            scratch_root=out_root / "eval_scratch",
            rng=rng,
            replica_devices=rep_devs,
            evaluator_devices=eval_devs,
            max_active_contexts_per_device=int(args.max_active_contexts_per_device),
        )
    print(f"[ONEOPES-REX] run complete; see {out_root}/exchange_log.csv", flush=True)


if __name__ == "__main__":
    main()
